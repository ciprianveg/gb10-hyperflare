/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ranks of one model on separate hosts: rendezvous at rank zero and connect their RoCE transport.

pub use flare_helios::execution::roce_fabric::Shape;
use std::sync::Arc;

use flare_helios::{
    execution::{CollectiveHealth, RankStream},
    runtime_core::{ClusterRank, NativeModel},
    startup::StartupProgress,
};

type Failure = Box<dyn std::error::Error>;

const NO_TRANSPORT: &str = "this flare build has no RoCE transport for multi-host cores; on Linux x86_64, rebuild with --features roce";

/// Reject unsupported serving before loading resources or sizing caches.
pub fn require_transport() -> Result<(), Failure> {
    if cfg!(all(
        target_os = "linux",
        any(target_arch = "aarch64", feature = "roce")
    )) {
        Ok(())
    } else {
        Err(NO_TRANSPORT.into())
    }
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
pub use flare_helios::execution::roce::RoceTransport;

#[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
pub struct RoceTransport;

/// A rank's connection to the rest of its cluster; the transport must outlive every replay.
pub struct Joined<S> {
    pub streams: S,
    pub transport: Arc<Vec<(String, RoceTransport)>>,
    /// A peer's host prefix record service, which rank zero drives until it closes its store.
    pub records: Option<flare_helios::offload::shard::Service>,
}

/// One completed frame's proxy-side decomposition, or None when no collective ran in the
/// window since the previous frame: per op kind its total wait (dispatch to completion in
/// the posting thread), op count, average wait and transfer bytes; compute is the frame's
/// wall time minus every kind's wait.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn frame_summary(
    label: &str,
    rank: usize,
    elapsed: std::time::Duration,
    before: &[flare_helios::execution::roce::KindStat;
              flare_helios::execution::roce::KIND_STATS],
    after: &[flare_helios::execution::roce::KindStat;
             flare_helios::execution::roce::KIND_STATS],
) -> Option<String> {
    use flare_helios::execution::roce::{self, KindStat, KIND_STATS};
    let kinds = [
        (roce::KIND_ONESHOT, "oneshot"),
        (roce::KIND_TWOSHOT, "pipe"),
        (roce::KIND_REDUCE_SCATTER, "pipe-rs"),
        (roce::KIND_GATHER_SHARDS, "pipe-ag"),
        (roce::KIND_VARLEN_RECORDS, "varlen-records"),
        (roce::KIND_VARLEN_HISTORY, "varlen-history"),
    ];
    let mut waited = 0u64;
    let mut parts = Vec::new();
    for (kind, name) in kinds {
        let ops = after[kind].ops - before[kind].ops;
        if ops == 0 {
            continue;
        }
        let nanos = after[kind].nanos - before[kind].nanos;
        let bytes = after[kind].bytes - before[kind].bytes;
        waited += nanos;
        parts.push(format!(
            "{name} {:.1}s ({} ops, avg {:.1} ms, {:.2} GB)",
            nanos as f64 / 1e9,
            ops,
            nanos as f64 / ops as f64 / 1e6,
            bytes as f64 / 1e9,
        ));
    }
    if parts.is_empty() {
        return None;
    }
    let seconds = elapsed.as_secs_f64();
    let compute = (seconds - waited as f64 / 1e9).max(0.0);
    Some(format!(
        "flare-roce rank {rank} {label}: {seconds:.1}s | compute ~{compute:.1}s | {}",
        parts.join(" | "),
    ))
}

/// The check every completed frame passes: a failed transport poisons the rank.
pub fn health(transport: &Arc<Vec<(String, RoceTransport)>>) -> CollectiveHealth {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        use flare_helios::execution::roce::{KindStat, KIND_STATS};
        use std::sync::Mutex;

        let transport = transport.clone();
        let frames: Mutex<Vec<Option<(std::time::Instant, [KindStat; KIND_STATS])>>> =
            Mutex::new(vec![None; transport.len()]);
        CollectiveHealth(Arc::new(move |kind| {
            let now = std::time::Instant::now();
            for (index, (id, channel)) in transport.iter().enumerate() {
              let stats = channel.kind_stats();
              if let Ok(mut frames) = frames.lock()
                  && let Some((started, before)) = frames[index].replace((now, stats))
                  && let Some(kind) = kind
              {
                let verbose =
                    std::env::var("FLARE_ROCE_FRAME_STATS").ok().as_deref() == Some("1");
                let label = if kind.is_prefill() {
                    "prefill frame"
                } else if matches!(kind, flare_helios::execution::GraphKind::Decode
                    | flare_helios::execution::GraphKind::Speculative)
                {
                    "decode frame"
                } else {
                    "frame"
                };
                if kind.is_prefill() || verbose {
                    if let Some(line) = frame_summary(label, channel.spec().rank, now - started, &before, &stats) {
                        if transport.len() == 1 { eprintln!("{line}"); }
                        else { eprintln!("communicator {id}: {line}"); }
                    }
                }
              }
              channel.health()?;
            }
            Ok(())
        }))
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = transport;
        unreachable!("no transport without RoCE support")
    }
}

/// Open this rank's transport over the fabric every rank discovered, from their endpoint lists in
/// rank order.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn declarations(owner: &NativeModel) -> Result<Vec<flare_helios::runtime_core::roce_communicators::Declaration>, Failure> {
    use flare_helios::runtime_core::roce_communicators::{Declaration, require_cluster_transport};
    let comms = require_cluster_transport(owner.roce_communicators(0).filter(|v| !v.is_empty()))?;
    Ok(comms.iter().map(Declaration::from).collect())
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn agree_communicators(records: &[Vec<u8>]) -> Result<std::collections::BTreeMap<String,
    flare_helios::runtime_core::roce_communicators::Declaration>, Failure> {
    let records = records.iter().map(|r| serde_json::from_slice(r)).collect::<Result<Vec<_>, _>>()?;
    Ok(flare_helios::runtime_core::roce_communicators::agree(&records)?)
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn open(
    owner: &NativeModel,
    fabric: &flare_helios::execution::roce_fabric::Fabric,
    endpoints: &[Vec<u8>],
    shape: Shape,
) -> Result<Vec<(String, RoceTransport)>, Failure> {
    use flare_helios::execution::roce_fabric;
    use flare_helios::runtime_core::roce_communicators::require_cluster_transport;
    let comms = require_cluster_transport(owner.roce_communicators(0).filter(|v| !v.is_empty()))?;
    let mut opened = Vec::new();
    // Normalization sorts IDs, so independently owned channels open deterministically.
    // A failure drops every already-opened transport before the borrowed owner.
    for comm in comms {
        let selected = comm.members.iter().map(|&rank| {
            let bytes = endpoints.get(rank).ok_or("communicator endpoint missing")?;
            Ok(roce_fabric::decode(bytes)?)
        }).collect::<Result<Vec<_>, Failure>>()?;
        let topology = fabric.topology(&selected, comm.transport.rank, shape)?;
        // SAFETY: Signed disjoint storage is restored and owner outlives every transport.
        // No model graph runs until all communicators on every rank have connected.
        opened.push((comm.id.clone(), unsafe { RoceTransport::open(&comm.transport, &topology)? }));
    }
    Ok(opened)
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn local_records(transports: &[(String, RoceTransport)]) -> Result<Vec<u8>, Failure> {
    use flare_helios::runtime_core::roce_communicators::Connection;
    let mut records = Vec::new();
    for (id, transport) in transports {
        records.push(Connection { id:id.clone(),local_rank:transport.spec().rank,bytes:transport.local_record()? });
    }
    Ok(serde_json::to_vec(&records)?)
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn connect_all(
    transports: &mut [(String, RoceTransport)],
    catalog: &std::collections::BTreeMap<String, flare_helios::runtime_core::roce_communicators::Declaration>,
    records: &[Vec<u8>],
) -> Result<(), Failure> {
    let records = records.iter().map(|r| serde_json::from_slice(r)).collect::<Result<Vec<_>, _>>()?;
    let routed = flare_helios::runtime_core::roce_communicators::connection_records(catalog, &records)?;
    for (id, transport) in transports {
        transport.connect(routed.get(id).ok_or("communicator connection records missing")?)?;
    }
    Ok(())
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn connected(transport: &RoceTransport) -> String {
    format!(
        "{} rails per {} over {}",
        transport.topology().rails,
        match transport.topology().shape {
            Shape::Mesh => "peer",
            Shape::Ring => "neighbor (experimental CPU relay ring)",
            Shape::Chain => "neighbor (experimental CPU relay chain)",
        },
        transport.topology().hcas.join(",")
    )
}

/// The transport shape a rank was started with, as the ranks exchange it before planning routes.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn shape_record(shape: Shape) -> Vec<u8> {
    shape.name().as_bytes().to_vec()
}

/// Every rank must select the same shape: a chain rank plans no route to the far end that a ring
/// rank would wait for.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn agree_shape(records: &[Vec<u8>], shape: Shape) -> Result<(), Failure> {
    let local = shape_record(shape);
    if let Some(rank) = records.iter().position(|record| *record != local) {
        return Err(format!(
            "rank {rank} selected RoCE transport shape {} where this rank selected {}",
            String::from_utf8_lossy(&records[rank]),
            String::from_utf8_lossy(&local)
        )
        .into());
    }
    Ok(())
}

/// A rank's admitted connections to the rest of its cluster, made before it loads.
pub enum Membership {
    /// Rank zero's connection to each peer, ranks 1 onward.
    Zero(Vec<RankStream>),
    /// A peer's connection to rank zero.
    Peer(RankStream),
}

impl flare_helios::runtime_core::CacheAgreement for Membership {
    fn caches(&mut self, local: (u64, u64)) -> flare_helios::Result<(u64, u64)> {
        use flare_helios::execution::rendezvous;
        match self {
            Membership::Zero(peers) => rendezvous::agree_capacity(peers, local),
            Membership::Peer(stream) => rendezvous::propose_capacity(stream, local),
        }
    }

    fn pool(
        &mut self,
        offer: flare_helios::offload::shard::PoolOffer,
    ) -> flare_helios::Result<Option<flare_helios::offload::shard::PoolPeers>> {
        use flare_helios::offload::shard;
        match self {
            Membership::Zero(peers) => Ok(shard::pool_rank_zero(offer, peers)?),
            Membership::Peer(stream) => Ok(shard::pool_peer(offer, stream)?),
        }
    }

    fn pool_approved(&mut self, approved: bool, world: usize) -> flare_helios::Result<bool> {
        use flare_helios::execution::rendezvous;
        let records = match self {
            Membership::Zero(peers) => rendezvous::share_records(peers, vec![approved as u8])?,
            Membership::Peer(stream) => {
                rendezvous::exchange_record(stream, &[approved as u8], world)?
            }
        };
        Ok(records.iter().all(|record| record.as_slice() == [1]))
    }
}

/// Rank zero: exchange transport records with every admitted peer and wait until every rank's
/// transport is connected.
pub fn rank_zero(
    owner: &NativeModel,
    streams: Vec<RankStream>,
    progress: &StartupProgress,
    shape: Shape,
) -> Result<Joined<Vec<RankStream>>, Failure> {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        use flare_helios::execution::{rendezvous, roce_fabric};
        let mut streams = streams;

        let phase = progress.phase("Connect RoCE transport", None);
        // Every rank learns every rank's devices and the shape, then plans the same routes from them.
        let fabric = roce_fabric::discover()?;
        let shapes = rendezvous::share_records(&mut streams, shape_record(shape))?;
        agree_shape(&shapes, shape)?;
        let declared = rendezvous::share_records(&mut streams, serde_json::to_vec(&declarations(owner)?)?)?;
        let catalog = agree_communicators(&declared)?;
        let endpoints =
            rendezvous::share_records(&mut streams, roce_fabric::encode(&fabric.endpoints))?;
        let mut transport = open(owner, &fabric, &endpoints, shape)?;
        let records = rendezvous::share_records(&mut streams, local_records(&transport)?)?;
        connect_all(&mut transport, &catalog, &records)?;
        rendezvous::await_connected(&mut streams)?;
        phase.finish(transport.iter().map(|(id,t)| format!("{id}: {}", connected(t))).collect::<Vec<_>>().join("; "));
        let phase = progress.phase("Join host prefix stores", None);
        match flare_helios::offload::shard::join_rank_zero(owner.offload(), &mut streams)? {
            Some(summary) => phase.finish(summary),
            None => phase.finish("none"),
        }
        Ok(Joined {
            streams,
            transport: Arc::new(transport),
            records: None,
        })
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = (owner, streams, progress, shape);
        Err(NO_TRANSPORT.into())
    }
}

/// A peer rank: exchange transport records with rank zero over its admitted stream, and connect.
pub fn peer(
    owner: &NativeModel,
    cluster: &ClusterRank,
    stream: RankStream,
    progress: &StartupProgress,
    shape: Shape,
) -> Result<Joined<RankStream>, Failure> {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        use flare_helios::execution::{rendezvous, roce_fabric};
        let mut stream = stream;

        let phase = progress.phase("Connect RoCE transport", None);
        let fabric = roce_fabric::discover()?;
        let shapes = rendezvous::exchange_record(&mut stream, &shape_record(shape), cluster.world)?;
        agree_shape(&shapes, shape)?;
        let declared = rendezvous::exchange_record(&mut stream, &serde_json::to_vec(&declarations(owner)?)?, cluster.world)?;
        let catalog = agree_communicators(&declared)?;
        let endpoints = rendezvous::exchange_record(
            &mut stream,
            &roce_fabric::encode(&fabric.endpoints),
            cluster.world,
        )?;
        let mut transport = open(owner, &fabric, &endpoints, shape)?;
        let records =
            rendezvous::exchange_record(&mut stream, &local_records(&transport)?, cluster.world)?;
        connect_all(&mut transport, &catalog, &records)?;
        rendezvous::report_connected(&mut stream)?;
        phase.finish(transport.iter().map(|(id,t)| format!("{id}: {}", connected(t))).collect::<Vec<_>>().join("; "));
        let phase = progress.phase("Join host prefix stores", None);
        let records = flare_helios::offload::shard::join_peer(
            owner.offload(),
            &mut stream,
            cluster.rank,
            cluster.world,
        )?;
        phase.finish(match &records {
            Some(service) => service.summary.clone(),
            None => "none".to_owned(),
        });
        Ok(Joined {
            streams: stream,
            transport: Arc::new(transport),
            records,
        })
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = (owner, cluster, stream, progress, shape);
        Err(NO_TRANSPORT.into())
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    use flare_helios::runtime_core::CacheAgreement;
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn pool_approval_is_unanimous_and_waits_for_every_rank() {
        for peer_approved in [true, false] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let zero = listener.accept().unwrap().0;
            let (ready, started) = std::sync::mpsc::channel();
            let zero = std::thread::spawn(move || {
                let approved = Membership::Zero(vec![RankStream::Tcp(zero)])
                    .pool_approved(true, 2)
                    .unwrap();
                ready.send(approved).unwrap();
                approved
            });
            assert!(
                started
                    .recv_timeout(std::time::Duration::from_millis(50))
                    .is_err()
            );
            let approved = Membership::Peer(RankStream::Tcp(peer))
                .pool_approved(peer_approved, 2)
                .unwrap();
            assert_eq!(approved, peer_approved);
            assert_eq!(started.recv().unwrap(), peer_approved);
            assert_eq!(zero.join().unwrap(), peer_approved);
        }
    }
}

#[cfg(all(test, target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
mod communicator_tests {
    use super::*;
    use flare_helios::runtime_core::{RoceSpec, roce_communicators::{Declaration,RoceCommunicator}};

    #[test]
    fn production_declaration_decoder_accepts_matching_records_and_rejects_missing_members() {
        let records=(0..2).map(|rank| {
            let c=RoceCommunicator { id:"tp".into(),members:vec![0,1],
                transport:RoceSpec {proxy_abi:14,world:2,rank,region_address:0x10000,
                    region_bytes:8192,counters_address:0x14000,counters_bytes:4096,
                    slot_bytes:16,pipe_bytes:16,varlen_bytes:0,ctrl_offset:4096} };
            serde_json::to_vec(&vec![Declaration::from(&c)]).unwrap()
        }).collect::<Vec<_>>();
        let catalog=agree_communicators(&records).unwrap();
        assert_eq!(catalog["tp"].members,vec![0,1]);
        let mut missing=records.clone();
        missing[1]=b"[]".to_vec();
        assert!(agree_communicators(&missing).is_err());
        let mut malformed=records;
        malformed[0]=b"not json".to_vec();
        assert!(agree_communicators(&malformed).is_err());
    }
}

#[cfg(all(
    test,
    all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))
))]
mod frame_summary_tests {
    use super::*;
    use flare_helios::execution::roce::{KindStat, KIND_STATS};

    fn stat(ops: u64, nanos: u64, bytes: u64) -> KindStat {
        KindStat { ops, nanos, bytes }
    }

    #[test]
    fn frame_summary_decomposes_one_prefill_chunk() {
        let mut before = [KindStat::default(); KIND_STATS];
        let mut after = [KindStat::default(); KIND_STATS];
        after[flare_helios::execution::roce::KIND_TWOSHOT] = stat(110, 41_300_000_000, 6_160_000_000);
        after[flare_helios::execution::roce::KIND_VARLEN_RECORDS] =
            stat(1760, 96_200_000_000, 265_000_000_000);
        after[flare_helios::execution::roce::KIND_VARLEN_HISTORY] =
            stat(220, 3_200_000_000, 1_900_000_000);
        let line = frame_summary(
            "prefill frame",
            3,
            std::time::Duration::from_millis(152_800),
            &before,
            &after,
        )
        .unwrap();
        assert!(line.starts_with("flare-roce rank 3 prefill frame: 152.8s"));
        assert!(line.contains("compute ~12.1s"));
        assert!(line.contains("pipe 41.3s (110 ops, avg 375.5 ms"));
        assert!(line.contains("varlen-records 96.2s (1760 ops, avg 54.7 ms"));
        assert!(line.contains("varlen-history 3.2s (220 ops"));
        // Zero ops in the window produce no line.
        assert!(frame_summary(
            "decode frame",
            0,
            std::time::Duration::from_millis(80),
            &after,
            &after,
        )
        .is_none());
    }
}

#[cfg(all(
    test,
    not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))
))]
mod transport_tests {
    use super::*;

    #[test]
    fn unsupported_build_rejects_multi_host_serving() {
        assert!(
            require_transport()
                .unwrap_err()
                .to_string()
                .contains("--features roce")
        );
    }
}
