//! Fixed representative TP16 pipeline replay, not exact trace reconstruction.
//! Qualification readback and counter snapshots run on separate no-collective
//! graph boundaries. Measured decode callbacks do not copy source/output bytes.

#[path = "allocation_watch.rs"]
mod allocation_watch;
mod pipe_replay_reference;
#[path = "pipe_replay_guard.rs"]
mod collective_replay_guard;
use collective_replay_guard::{Step, schedule, finish_transfer};

use flare_helios::{
    execution::{self, BufferDescriptor, CollectiveHealth, ExecutionManifest, ExecutionWorker,
        GraphKind, RankStream, rendezvous, roce::{RoceTransport, PROXY_ABI}, roce_fabric},
    runtime_core::{KvCacheLimit, LoadOptions, NativeModel, VerifiedCore, core_cluster, test_support},
    startup::StartupProgress,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{ffi::c_void, fs, path::{Path, PathBuf}, sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH}, sync::atomic::Ordering,
    mem::ManuallyDrop};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const IMAGE: &str = "sha256:4d668f11127fe0f245b23bb26737951d6cbbcb0ca2ec6f17b33ef9745f12bed6";

fn failure(message: impl Into<String>) -> flare_helios::Error {
    flare_helios::Error::execution(flare_helios::ExecutionError::Worker, message.into())
}

#[derive(Clone, Deserialize)]
struct Operation {
    index: usize, kind: String, graph: String, shape: [usize; 2], output_shape: [usize; 2],
    transport_shape: [usize; 2], dtype: String, bytes: usize,
}

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(fields) => {
            let sorted: std::collections::BTreeMap<_, _> =
                fields.iter().map(|(k, v)| (k.clone(), canonical(v))).collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
        _ => value.clone(),
    }
}

fn read_manifest(path: &Path, digest: &str) -> Result<Vec<Operation>> {
    let mut body: Value = serde_json::from_slice(&fs::read(path)?)?;
    if body["world"] != 16 || body["digest"] != digest {
        return Err("manifest world or requested digest mismatch".into());
    }
    body.as_object_mut().ok_or("manifest must be an object")?.remove("digest");
    let bytes = serde_json::to_vec(&canonical(&body))?;
    let actual = ring::digest::digest(&ring::digest::SHA256, &bytes).as_ref().iter()
        .map(|b| format!("{b:02x}")).collect::<String>();
    if actual != digest { return Err("manifest digest mismatch".into()); }
    let ops: Vec<Operation> = serde_json::from_value(body["operations"].clone())?;
    if ops.len() != 9 || body["fixture"] != "representative-pipeline-v1"
        || body["qualification_graph"] != "init" || body["measured_graph"] != "decode"
        || body["blocks"] != 8 || body["pipe_blocks"] != 4 || body["threads"] != 512 || body["rails"] != 2 {
        return Err("not the fixed representative pipeline fixture".into());
    }
    let specs = [("AR",1,8,"init"),("RS",16,7168,"init"),("RS",16,7168,"init"),
        ("AR",2048,7168,"init"),("SG",1,7168,"init"),("SG",1,7168,"init"),
        ("AR",1,8,"init"),("AR",2048,7168,"decode"),("AR",4096,7168,"decode")];
    for (index, (op, &(kind,rows,columns,graph))) in ops.iter().zip(&specs).enumerate() {
        let output_rows = if kind=="RS" {rows/16} else if kind=="SG" {rows*16} else {rows};
        let message = rows*columns*2 * if kind=="SG" {16} else {1};
        let pipe = kind!="AR" || message >= 1<<20;
        let entry = &body["operations"][index];
        if op.index != index || op.kind != kind || op.graph != graph || op.dtype != "bf16"
            || op.shape != [rows,columns] || op.transport_shape != op.shape
            || op.bytes != rows*columns*2 || op.output_shape != [output_rows,columns]
            || entry["message_bytes"] != message || entry["grid"] != if pipe {4} else {1}
            || entry["chunk_packs"] != if pipe {32768} else {0}
            || entry["chunks"] != if pipe {(message/16).div_ceil(524288)} else {0} {
            return Err("pipeline matrix or chunk contract differs".into());
        }
    }
    for (key,value) in [("max_tokens",1024),("pipe_tokens",4096),("hidden_size",7168),
        ("gather_bytes",16777216),("varlen_bytes",150994944),("pipe_chunk_bytes",524288)] {
        if body["capacities"][key] != value { return Err("pipeline capacity differs".into()); }
    }
    Ok(ops)
}

fn source(op: &Operation, rank: usize) -> Vec<u8> {
    pipe_replay_reference::source(op.index, op.bytes, rank)
}

fn expected(op: &Operation, rank: usize) -> Vec<u8> {
    pipe_replay_reference::expected(op.index, &op.kind, op.bytes, rank)
}

fn preparation_exited(metadata: &Value) -> Result<()> {
    let p = &metadata["process"];
    if p["boot_id"].as_str() != Some(fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim()) {
        return Err("preparation receipt belongs to another host boot".into());
    }
    let pid = p["pid"].as_u64().ok_or("missing preparation PID")?;
    let start = p["start_ticks"].as_u64().ok_or("missing preparation start time")?;
    match fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => {
            let ticks: u64 = stat.rsplit_once(") ").ok_or("invalid proc stat")?.1
                .split_whitespace().nth(19).ok_or("missing proc start time")?.parse()?;
            if ticks == start { return Err("preparation process is still alive".into()); }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

enum Members { Zero(Vec<RankStream>), Peer(RankStream) }
impl Members {
    fn exchange(&mut self, local: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        Ok(match self {
            Self::Zero(peers) => rendezvous::share_records(peers, local)?,
            Self::Peer(peer) => rendezvous::exchange_record(peer, &local, 16)?,
        })
    }
}

struct Driver {
    _library: libloading::Library,
    context: unsafe extern "C" fn(*mut c_void) -> i32,
    upload: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    download: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    synchronize: unsafe extern "C" fn() -> i32,
}
impl Driver {
    fn new() -> Result<Self> {
        unsafe {
            let library = libloading::Library::new("libcuda.so.1")?;
            Ok(Self { context: *library.get(b"cuCtxSetCurrent\0")?,
                upload: *library.get(b"cuMemcpyHtoD_v2\0")?,
                download: *library.get(b"cuMemcpyDtoH_v2\0")?,
                synchronize: *library.get(b"cuCtxSynchronize\0")?, _library: library })
        }
    }
    fn check(status: i32) -> flare_helios::Result<()> {
        if status == 0 { Ok(()) } else { Err(failure(format!("CUDA fixture status {status}"))) }
    }
    fn select(&self, context: usize) -> flare_helios::Result<()> {
        unsafe { Self::check((self.context)(context as *mut c_void)) }
    }
    fn write(&self, buffer: &BufferDescriptor, bytes: Vec<u8>) -> flare_helios::Result<()> {
        if buffer.nbytes != bytes.len() { return Err(failure("fixture upload size mismatch")); }
        let copied = unsafe { (self.upload)(buffer.address as u64, bytes.as_ptr().cast(), bytes.len()) };
        // A pageable upload's return does not order nonblocking graph streams.
        // Synchronize even on copy failure before releasing its staging bytes.
        let drained = unsafe { (self.synchronize)() };
        finish_transfer(bytes, copied == 0, drained == 0)
            .map(drop).map_err(|e| failure(format!("{e}; copy={copied}, sync={drained}")))
    }
    fn read(&self, buffer: &BufferDescriptor) -> flare_helios::Result<Vec<u8>> {
        let mut bytes = vec![0; buffer.nbytes];
        let copied = unsafe { (self.download)(bytes.as_mut_ptr().cast(), buffer.address as u64, bytes.len()) };
        let drained = unsafe { (self.synchronize)() };
        finish_transfer(bytes, copied == 0, drained == 0)
            .map_err(|e| failure(format!("{e}; copy={copied}, sync={drained}")))
    }
}

struct Check { source: BufferDescriptor, output: BufferDescriptor, input: Vec<u8>, expected: Vec<u8> }
fn checks(manifest: &ExecutionManifest, ops: &[Operation], rank: usize) -> Result<Vec<Check>> {
    let buffer = |name: String| manifest.buffers.iter().find(|b| b.name == name)
        .cloned().ok_or_else(|| format!("missing fixture buffer {name}"));
    ops.iter().map(|op| {
        let input = source(op, rank);
        let expected = expected(op, rank);
        let source = buffer(format!("source-{:03}", op.index))?;
        let output = buffer(format!("output-{:03}", op.index))?;
        if source.nbytes != input.len() || output.nbytes != expected.len() {
            return Err("native buffer does not match manifest".into());
        }
        Ok(Check { source, output, input, expected })
    }).collect()
}

fn nic_snapshot() -> Value {
    let mut counters = serde_json::Map::new();
    for hca in ["roceP2p1s0f1", "rocep1s0f1"] {
        for directory in ["counters", "hw_counters"] {
            let path = format!("/sys/class/infiniband/{hca}/ports/1/{directory}");
            match fs::read_dir(&path) {
                Err(e) => { counters.insert(path, json!({"error":e.to_string()})); }
                Ok(entries) => for entry in entries {
                    match entry {
                        Err(e) => { counters.insert(path.clone(), json!({"error":e.to_string()})); }
                        Ok(entry) => {
                            let filename = entry.path();
                            let value = match fs::read_to_string(&filename) {
                                Ok(v) => json!({"raw":v.trim()}),
                                Err(e) => json!({"error":e.to_string()}),
                            };
                            counters.insert(filename.display().to_string(), value);
                        }
                    }
                }
            }
        }
    }
    Value::Object(counters)
}

fn timestamp_ns() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos().try_into()?)
}

fn run(args: &[String]) -> Result<()> {
    let get = |flag: &str| -> Result<String> {
        let i = args.iter().position(|a| a == flag).ok_or_else(|| format!("missing {flag}"))?;
        args.get(i + 1).cloned().ok_or_else(|| format!("missing value for {flag}").into())
    };
    let rank: usize = get("--rank")?.parse()?;
    let world: usize = get("--world")?.parse()?;
    let device: i32 = get("--device")?.parse()?;
    let rounds: usize = get("--rounds")?.parse()?;
    let warmup: usize = get("--warmup")?.parse()?;
    if world != 16 || rank >= 16 || device != 0 || !(1..=28).contains(&rounds)
        || !(1..=4).contains(&warmup) || get("--image-id")? != IMAGE
        || !args.iter().any(|a| a == "--exclusive-fleet-confirmed") {
        return Err("requires TP16, rank 0..15, device 0, rounds 1..28, warmup 1..4, pinned image and explicit fleet release".into());
    }
    let root = PathBuf::from(get("--directory")?);
    let digest = get("--digest")?;
    let local = root.join(format!("rank-{rank}"));
    let ops = read_manifest(&local.join("manifest.json"), &digest)?;
    let metadata: Value = serde_json::from_slice(&fs::read(local.join("replay.json"))?)?;
    if metadata["rank"] != rank || metadata["world"] != world || metadata["digest"] != digest {
        return Err("local preparation metadata mismatch".into());
    }
    preparation_exited(&metadata)?;
    // Authenticate the selected rank and assets before any GPU access.
    let authorized = VerifiedCore::open(&root.join("core"), &[test_support::publisher()], Some(rank))?
        .authorize("flare-tests", false)?;
    let (core_world, export_id) = core_cluster(&authorized).ok_or("core has no cluster identity")?;
    if core_world != 16 || metadata["export_id"] != export_id { return Err("core export mismatch".into()); }
    let address = get("--rendezvous")?;
    let mut members = if rank == 0 {
        Members::Zero(rendezvous::accept(&address, 16, &export_id, Duration::from_secs(180), |_| {})?)
    } else {
        Members::Peer(rendezvous::knock(&address, rank, 16, &export_id, Duration::from_secs(180))?)
    };
    let settings = serde_json::to_vec(&json!({"digest":digest, "rounds":rounds, "warmup":warmup,
        "shape":"mesh", "image":IMAGE, "preparation_exited":true}))?;
    if members.exchange(settings.clone())?.iter().any(|r| r != &settings) {
        return Err("ranks disagree on replay settings".into());
    }
    let allocations = allocation_watch::AllocationWatch::new()?;
    let options = LoadOptions { cluster_rank: Some(rank), kv_cache_limit: KvCacheLimit::Context,
        ..LoadOptions::default() };
    let owner = unsafe { NativeModel::load(authorized, &[device], &options, &StartupProgress::silent())? };
    let mut manifests = owner.manifests()?;
    if manifests.len() != 1 { return Err("expected one local rank".into()); }
    let mut manifest = manifests.remove(0);
    let spec = owner.roce(0).ok_or("core has no RoCE region")?;
    if spec.world != 16 || spec.rank != rank || spec.region_bytes != 5_583_155_200
        || spec.varlen_bytes != 150_994_944 || spec.pipe_bytes != 58_720_256
        || spec.slot_bytes != 16_777_216 {
        return Err(format!("production transport reservation mismatch: {spec:?}").into());
    }
    let fabric = roce_fabric::discover()?;
    let endpoints = members.exchange(roce_fabric::encode(&fabric.endpoints))?;
    let endpoints = endpoints.iter().map(|b| roce_fabric::decode(b)).collect::<flare_helios::Result<Vec<_>>>()?;
    let topology = fabric.topology(&endpoints, rank, roce_fabric::Shape::Mesh)?;
    if topology.rails != 2 { return Err("fixture requires two rails".into()); }
    // open performs the existing ABI9/10/11/14 and exact region-layout checks.
    let mut transport = unsafe { RoceTransport::open(spec, &topology)? };
    let records = members.exchange(transport.local_record()?)?;
    if records.iter().any(|r| r.get(..4) != Some(PROXY_ABI.to_le_bytes().as_slice())) {
        return Err("every native peer must advertise ABI14".into());
    }
    transport.connect(&records)?;
    let mut guard = collective_replay_guard::Guard::default();
    guard.connect(&records)?;
    match &mut members {
        Members::Zero(peers) => rendezvous::await_connected(peers)?,
        Members::Peer(peer) => rendezvous::report_connected(peer)?,
    }
    let core_proxy_abi = spec.proxy_abi;
    // From the first upload onward, every error path retains native resources,
    // including startup failures before an ExecutionWorker exists. Only the
    // successful coordinated-drain path below explicitly releases these owners.
    let mut owner = ManuallyDrop::new(owner);
    let mut transport = ManuallyDrop::new(Arc::new(transport));
    let driver = Driver::new()?;
    driver.select(manifest.context)?;
    let checks = checks(&manifest, &ops, rank)?;
    for check in &checks {
        driver.write(&check.source, check.input.clone())?;
        driver.write(&check.output, vec![0xff; check.output.nbytes])?;
    }
    let context = manifest.context;
    let steps = schedule(warmup, rounds)?;
    let callback_steps = steps.clone();
    let count = Arc::new(Mutex::new(0usize));
    let observed = count.clone();
    let health = Arc::clone(&transport);
    let allocation_counter = allocations.counter();
    let allocation_baseline = Mutex::new(None);
    manifest.collectives = Some(CollectiveHealth(Arc::new(move |kind| {
        health.health()?;
        let Some(kind) = kind else { return Ok(()); };
        let mut completed = observed.lock().map_err(|_| failure("phase counter poisoned"))?;
        let step = *callback_steps.get(*completed).ok_or_else(|| failure("unexpected extra completion"))?;
        let expected_kind = match step.graph() {
            "init" => GraphKind::Init,
            "decode" => GraphKind::Decode,
            "restore" => GraphKind::Restore,
            "snapshot" => GraphKind::Snapshot,
            _ => unreachable!(),
        };
        if kind != expected_kind { return Err(failure("fixture completion phase mismatch")); }
        if step.readback() {
            // Rank zero submitted this Restore only after all preceding graph
            // completions. No rank can still be executing a measured decode.
            driver.select(context)?;
            for (index, check) in checks.iter().enumerate() {
                if driver.read(&check.source)? != check.input
                    || (step.checks_output(index) && driver.read(&check.output)? != check.expected) {
                    return Err(failure(format!("rank {rank} parity/source failure at operation {index}")));
                }
            }
            println!("{}", json!({"kind":"qualification", "phase":format!("{step:?}"),
                "rank":rank, "outputs":(0..9).filter(|&i| step.checks_output(i)).count(),
                "sources_unchanged":true}));
        }
        let allocations = allocation_counter.load(Ordering::Relaxed);
        let mut baseline = allocation_baseline.lock().map_err(|_| failure("allocation counter poisoned"))?;
        if step == Step::Begin { *baseline = Some(allocations); }
        if matches!(step, Step::Measure | Step::End) && *baseline != Some(allocations) {
            return Err(failure(format!("rank {rank}: CUDA allocation during measured replay")));
        }
        if matches!(step, Step::Begin | Step::End) {
            let started = timestamp_ns().map_err(|e| failure(e.to_string()))?;
            let nic = nic_snapshot();
            let proxy = format!("{:?}", health.kind_stats());
            let finished = timestamp_ns().map_err(|e| failure(e.to_string()))?;
            println!("{}", json!({"kind":"measurement_boundary", "phase":format!("{step:?}"),
                "rank":rank, "snapshot_start_unix_ns":started, "snapshot_end_unix_ns":finished,
                "nic":nic, "proxy":proxy, "boundary_graph_collectives":0}));
        }
        *completed += 1;
        Ok(())
    })));
    println!("{}", json!({"kind":"connected", "rank":rank, "core_proxy_abi":core_proxy_abi,
        "native_proxy_abi":PROXY_ABI, "image_asserted":IMAGE,
        "binary":"separate native test executable; source history corroborated, not cryptographically attested"}));
    let shutdown = match members {
        Members::Peer(peer) => {
            unsafe { execution::serve_rank(manifest, peer, rank, 16) }
        }
        Members::Zero(peers) => {
            let graph_indices = steps.iter().map(|step|
                manifest.graphs.iter().position(|g| g.name == step.graph()).ok_or("fixture graph missing"))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let worker = unsafe { ExecutionWorker::new_parallel(manifest, peers)? };
            let result = (|| -> Result<()> {
                let mut measured = 0;
                for (step, graph) in steps.iter().zip(graph_indices) {
                    guard.submit()?;
                    let ticket = worker.submit(graph, vec![], false)?.ok_or("queue full")?;
                    let started = Instant::now();
                    let mut warned = false;
                    let completion = loop {
                        if let Some(completion) = worker.poll()? { break completion; }
                        // poll and close are synchronous and can block inside
                        // native execution. This is not a process time bound.
                        if !warned && started.elapsed() > Duration::from_secs(120) {
                            eprintln!("advisory progress deadline exceeded; awaiting native completion/drain, no kill or retry");
                            warned = true;
                        }
                        std::thread::yield_now();
                    };
                    if completion.fence != ticket || completion.timing.gpu_ns.iter().any(Option::is_none) {
                        return Err("completion identity or rank timing missing".into());
                    }
                    guard.complete(true)?;
                    if *step == Step::Measure {
                        println!("{}", json!({"kind":"measurement", "round":measured,
                            "rank_gpu_ns":completion.timing.gpu_ns,
                            "max_rank_gpu_ns":completion.timing.gpu_ns.iter().flatten().max(),
                            "host_dispatch_through_all_rank_completion_ns":completion.timing.wall_ns}));
                        measured += 1;
                    } else {
                        println!("{}", json!({"kind":"untimed_phase", "phase":format!("{step:?}"),
                            "collectives":if step.graph() == "init" {7} else if step.graph() == "decode" {2} else {0}}));
                    }
                }
                if guard.completed() != steps.len() { return Err("coordinated phase sequence incomplete".into()); }
                Ok(())
            })();
            let shutdown = worker.close();
            if let Err(error) = result {
                return Err(error);
            }
            shutdown
        }
    };
    if !shutdown.drained {
        return Err("native drain unconfirmed; owners retained until process exit".into());
    }
    guard.release(shutdown.drained)?;
    if let Some(error) = shutdown.error { return Err(error.into()); }
    if *count.lock().map_err(|_| "phase counter poisoned")? != steps.len() {
        return Err("qualification/measurement sequence incomplete".into());
    }
    println!("{}", json!({"kind":"drained", "rank":rank, "measured_rounds":rounds,
        "warmup_rounds":warmup, "qualification_rounds":1, "transition_graphs":1, "parity_checks":3,
        "transition_outputs":7, "measured_outputs":2, "no_collective_boundary_graphs":5,
        "maintenance_operations":0}));
    // No accepted transfer or graph remains outstanding on this success path.
    unsafe { ManuallyDrop::drop(&mut transport); ManuallyDrop::drop(&mut owner); }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help") {
        println!("test_collective_replay_pipe_<baseline|variant> --directory PATH --rank N --world 16 --device 0 \
            --rendezvous HOST:PORT --digest SHA256 --rounds 28 --warmup 2 \
            --image-id {IMAGE} --exclusive-fleet-confirmed\n\
            CPU-only help. Execution requires signed fixture and explicit GPU release. No launcher.");
        return Ok(());
    }
    run(&args)
}
