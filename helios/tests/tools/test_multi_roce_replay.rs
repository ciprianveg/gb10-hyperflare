//! Weightless plural-transport replay using the serving CLI's actual bootstrap.
//! CPU plan checking never opens a CUDA library or a rendezvous socket.
#[path = "../../../flare-cli/src/cluster.rs"]
mod cluster;
mod allocation_watch;
mod collective_replay_guard;

use flare_helios::{
    execution::{self, BufferDescriptor, CollectiveHealth, ExecutionWorker, GraphKind, rendezvous},
    runtime_core::{KvCacheLimit, LoadOptions, NativeModel, VerifiedCore, core_cluster, test_support},
    startup::StartupProgress,
};
use serde::Deserialize;
use serde_json::json;
use std::{ffi::c_void, fs, mem::ManuallyDrop, path::Path, sync::{Arc, Mutex, atomic::Ordering}, time::Duration};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn failure(message: impl Into<String>) -> flare_helios::Error {
    flare_helios::Error::execution(flare_helios::ExecutionError::Worker, message.into())
}

#[derive(Deserialize)]
struct Check {
    source: String,
    output: String,
    input: Vec<u8>,
    expected: Vec<u8>,
}
#[derive(Deserialize)]
struct Transport {
    id: String,
    members: Vec<usize>,
    region_address: u64,
    region_bytes: u64,
    counters_address: u64,
    counters_bytes: u64,
}
#[derive(Deserialize)]
struct Plan {
    rank: usize,
    export_id: String,
    preparation_pid: u32,
    preparation_start_ticks: u64,
    boot_id: String,
    transports: Vec<Transport>,
    checks: Vec<Check>,
}
impl Plan {
    fn validate(&self) -> Result<()> {
        if self.rank >= 16 || self.export_id.is_empty() || self.transports.len() != 2
            || self.checks.is_empty() {
            return Err("expected one rank of TP16 plus local DCP8 and nonempty checks".into());
        }
        let subgroup = self.rank / 8;
        let expected = [("tp", (0..16).collect::<Vec<_>>()),
            (if subgroup == 0 {"dcp-a"} else {"dcp-b"}, (subgroup*8..subgroup*8+8).collect())];
        let mut ranges = Vec::new();
        for (id, members) in expected {
            let t = self.transports.iter().find(|t| t.id == id).ok_or("missing communicator")?;
            if t.members != members { return Err("communicator membership mismatch".into()); }
            // Mapped host regions are page-aligned; device counters are int32
            // tensors and need natural atomic alignment, not a page boundary.
            for (address, bytes, alignment) in [(t.region_address,t.region_bytes,4096),(t.counters_address,t.counters_bytes,4)] {
                let end = address.checked_add(bytes).filter(|_| bytes > 0 && address % alignment == 0)
                    .ok_or("invalid protocol range")?;
                if ranges.iter().any(|&(a,b)| address < b && a < end) {
                    return Err("protocol ranges overlap".into());
                }
                ranges.push((address,end));
            }
        }
        let mut names = std::collections::BTreeSet::new();
        for c in &self.checks {
            if c.input.is_empty() || c.expected.is_empty() || c.input.len()%2 != 0 || c.expected.len()%2 != 0
                || !names.insert(&c.source) || !names.insert(&c.output) {
                return Err("checks require distinct source/output names and nonempty BF16 byte arrays".into());
            }
        }
        Ok(())
    }
    fn preparation_exited(&self) -> Result<()> {
        if fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim() != self.boot_id {
            return Err("preparation belongs to a different host boot".into());
        }
        match fs::read_to_string(format!("/proc/{}/stat",self.preparation_pid)) {
            Ok(stat) => {
                let start: u64 = stat.rsplit_once(") ").ok_or("invalid proc stat")?.1
                    .split_whitespace().nth(19).ok_or("missing start ticks")?.parse()?;
                if start == self.preparation_start_ticks { return Err("preparation still owns GPU resources".into()); }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        Ok(())
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
            Ok(Self { context:*library.get(b"cuCtxSetCurrent\0")?,
                upload:*library.get(b"cuMemcpyHtoD_v2\0")?, download:*library.get(b"cuMemcpyDtoH_v2\0")?,
                synchronize:*library.get(b"cuCtxSynchronize\0")?, _library:library })
        }
    }
    fn select(&self, context: usize) -> flare_helios::Result<()> {
        if unsafe {(self.context)(context as *mut c_void)} != 0 { return Err(failure("CUDA context selection failed")); }
        Ok(())
    }
    fn write(&self, b: &BufferDescriptor, bytes: Vec<u8>) -> flare_helios::Result<()> {
        if b.nbytes != bytes.len() { return Err(failure("upload size mismatch")); }
        let copied = unsafe {(self.upload)(b.address as u64, bytes.as_ptr().cast(), bytes.len())};
        let drained = unsafe {(self.synchronize)()};
        collective_replay_guard::finish_transfer(bytes, copied == 0, drained == 0)
            .map(drop).map_err(|e|failure(e.to_string()))
    }
    fn read(&self, b: &BufferDescriptor) -> flare_helios::Result<Vec<u8>> {
        let mut bytes = vec![0;b.nbytes];
        let copied = unsafe {(self.download)(bytes.as_mut_ptr().cast(),b.address as u64,bytes.len())};
        let drained = unsafe {(self.synchronize)()};
        collective_replay_guard::finish_transfer(bytes,copied == 0,drained == 0)
            .map_err(|e|failure(e.to_string()))
    }
}

fn run(args: &[String]) -> Result<()> {
    let get = |flag: &str| -> Result<&str> {
        let i = args.iter().position(|a| a == flag).ok_or_else(||format!("missing {flag}"))?;
        args.get(i+1).map(String::as_str).ok_or_else(||format!("missing value for {flag}").into())
    };
    let plan: Plan = serde_json::from_slice(&fs::read(get("--plan")?)?)?;
    plan.validate()?;
    if args.iter().any(|a| a == "--check-plan") {
        println!("{}",json!({"cpu_plan_valid":true,"rank":plan.rank,"communicators":2}));
        return Ok(());
    }
    if !args.iter().any(|a| a == "--exclusive-fleet-confirmed") {
        return Err("explicit parent fleet release required".into());
    }
    let rank: usize = get("--rank")?.parse()?;
    let rounds: usize = get("--rounds")?.parse()?;
    if rank != plan.rank || !(2..=16).contains(&rounds) { return Err("rank mismatch or rounds outside 2..16".into()); }
    plan.preparation_exited()?;
    let authorized = VerifiedCore::open(Path::new(get("--core")?), &[test_support::publisher()], Some(rank))?
        .authorize("flare-tests", false)?;
    if core_cluster(&authorized) != Some((16,plan.export_id.clone())) { return Err("signed cluster identity mismatch".into()); }
    let address = get("--rendezvous")?;
    let mut membership = if rank == 0 {
        cluster::Membership::Zero(rendezvous::accept(address,16,&plan.export_id,Duration::from_secs(180), |_|{})?)
    } else {
        cluster::Membership::Peer(rendezvous::knock(address,rank,16,&plan.export_id,Duration::from_secs(180))?)
    };
    let settings = serde_json::to_vec(&json!({"rounds":rounds,"export_id":plan.export_id,"world":16}))?;
    let agreed = match &mut membership {
        cluster::Membership::Zero(peers) => rendezvous::share_records(peers,settings.clone())?,
        cluster::Membership::Peer(peer) => rendezvous::exchange_record(peer,&settings,16)?,
    };
    if agreed.iter().any(|record|record != &settings) {return Err("replay settings differ between ranks".into());}
    let watch = allocation_watch::AllocationWatch::new()?;
    let owner = unsafe {NativeModel::load(authorized,&[0],&LoadOptions {
        cluster_rank:Some(rank),kv_cache_limit:KvCacheLimit::Context,..LoadOptions::default()
    },&StartupProgress::silent())?};
    let comms = owner.roce_communicators(0).ok_or("missing restored communicators")?;
    if comms.len() != 2 { return Err("expected exactly two restored communicators".into()); }
    for c in comms {
        let t = plan.transports.iter().find(|t|t.id == c.id).ok_or("unexpected restored communicator")?;
        let s = &c.transport;
        if c.members != t.members || c.members.get(s.rank) != Some(&rank)
            || s.world != t.members.len() || s.region_address != t.region_address
            || s.region_bytes != t.region_bytes || s.counters_address != t.counters_address
            || s.counters_bytes != t.counters_bytes {
            return Err("restored transport differs from prepared owner".into());
        }
    }
    let progress = StartupProgress::silent();
    let (membership,transport) = match membership {
        cluster::Membership::Zero(streams) => {
            let joined = cluster::rank_zero(&owner,streams,&progress,cluster::Shape::Mesh)?;
            (cluster::Membership::Zero(joined.streams),joined.transport)
        }
        cluster::Membership::Peer(stream) => {
            let joined = cluster::peer(&owner,owner.cluster().ok_or("cluster absent")?,stream,&progress,cluster::Shape::Mesh)?;
            if joined.records.is_some() { return Err("weightless fixture must not own prefix records".into()); }
            (cluster::Membership::Peer(joined.streams),joined.transport)
        }
    };
    let mut manifests = owner.manifests()?;
    if manifests.len() != 1 { return Err("expected one manifest".into()); }
    let mut manifest = manifests.remove(0);
    // Once uploads start, uncertain drain intentionally retains both ownership sets.
    let mut owner = ManuallyDrop::new(owner);
    let mut transport = ManuallyDrop::new(transport);
    let driver = Driver::new()?;
    driver.select(manifest.context)?;
    let mut checks = Vec::new();
    let mut ranges = Vec::new();
    for c in plan.checks {
        let find = |name: &str| manifest.buffers.iter().find(|b|b.name == name).cloned().ok_or("missing check buffer");
        let source = find(&c.source)?;
        let output = find(&c.output)?;
        if source.nbytes != c.input.len() || output.nbytes != c.expected.len() { return Err("check extent mismatch".into()); }
        for b in [&source,&output] {
            let end = b.address.checked_add(b.nbytes).ok_or("buffer overflow")?;
            if ranges.iter().any(|&(a,z)|b.address < z && a < end) { return Err("fixture scratch aliases".into()); }
            ranges.push((b.address,end));
        }
        driver.write(&source,c.input.clone())?;
        driver.write(&output,vec![0xff;output.nbytes])?;
        checks.push((source,output,c));
    }
    let context = manifest.context;
    let health = cluster::health(&transport);
    let counter = watch.counter();
    let baseline = Mutex::new(None);
    let completed = Arc::new(Mutex::new(0usize));
    let observed = completed.clone();
    manifest.collectives = Some(CollectiveHealth(Arc::new(move |kind| {
        (health.0)(kind)?;
        let Some(kind) = kind else {return Ok(())};
        let mut n = observed.lock().map_err(|_|failure("completion counter poisoned"))?;
        let expected = if *n % 2 == 0 {GraphKind::Decode} else {GraphKind::Restore};
        if kind != expected { return Err(failure("fixture phase mismatch")); }
        if kind == GraphKind::Restore {
            driver.select(context)?;
            for (source,output,c) in &checks {
                if driver.read(source)? != c.input || driver.read(output)? != c.expected {
                    return Err(failure(format!("rank {rank}: parity or source preservation failed for {}",c.output)));
                }
            }
            let allocations = counter.load(Ordering::Relaxed);
            let mut baseline = baseline.lock().map_err(|_|failure("allocation baseline poisoned"))?;
            if let Some(previous) = *baseline {
                if previous != allocations {return Err(failure("CUDA allocation during repeated replay"));}
            } else {
                *baseline = Some(allocations);
            }
            println!("{}",json!({"rank":rank,"round":*n/2,"parity":true,"sources_preserved":true}));
        }
        *n += 1;
        Ok(())
    })));
    let shutdown = match membership {
        cluster::Membership::Peer(stream) => unsafe {execution::serve_rank(manifest,stream,rank,16)},
        cluster::Membership::Zero(peers) => {
            let indices = ["decode","restore"].iter().map(|name|manifest.graphs.iter().position(|g|g.name == *name)
                .ok_or("missing fixture graph")).collect::<std::result::Result<Vec<_>,_>>()?;
            let worker = unsafe {ExecutionWorker::new_parallel(manifest,peers)?};
            let result = (|| -> Result<()> {
                for _ in 0..rounds {
                    for &graph in &indices {
                        let ticket = worker.submit(graph,vec![],false)?.ok_or("queue full")?;
                        let done = loop {
                            if let Some(done) = worker.poll()? {break done;}
                            std::thread::yield_now();
                        };
                        if done.fence != ticket {return Err("completion identity mismatch".into());}
                    }
                }
                Ok(())
            })();
            let shutdown = worker.close();
            result?;
            shutdown
        }
    };
    if !shutdown.drained {return Err("drain unconfirmed; native owners retained".into());}
    if let Some(error) = shutdown.error {return Err(error.into());}
    if *completed.lock().map_err(|_|"completion lock poisoned")? != 2*rounds {return Err("incomplete schedule".into());}
    unsafe {ManuallyDrop::drop(&mut transport);ManuallyDrop::drop(&mut owner);}
    println!("{}",json!({"rank":rank,"drained":true,"rounds":rounds,"performance_claim":false}));
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a|a == "--help") {
        println!("test_multi_roce_replay --plan PATH --check-plan\n\
            test_multi_roce_replay --plan PATH --core PATH --rank N --rounds 4 \
            --rendezvous HOST:PORT --exclusive-fleet-confirmed\n\
            CPU help/check-plan only; replay requires separate fleet authorization.");
        return Ok(());
    }
    run(&args)
}
