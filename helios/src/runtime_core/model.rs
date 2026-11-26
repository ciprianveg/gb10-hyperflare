/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! A model core owns a disjoint resource bundle for every tensor-parallel rank.

use super::{
    AuthorizedCore, LoadOptions, NativeResources,
    driver::{Access, Location},
    format::{Memory, Core, Reservation, invalid},
    payload::Files,
    protection::AuthorizedKind,
    protection::FilesRef,
    protobuf,
};
use crate::{Result, execution::ExecutionManifest, startup::StartupProgress};
use blake2::{Blake2b, Digest, digest::consts::U32};
use std::{collections::HashSet, path::Path, thread};

pub struct NativeModel {
    pub(super) ranks: Vec<NativeResources>,
    _core_files: Vec<std::sync::Arc<std::fs::File>>,
    pub(super) server_config: serde_json::Value,
    engram: Vec<(std::sync::Arc<crate::engram::Shared>, usize)>,
    pub(super) offload: flare_selene::offload::Startup,
    cluster: Option<ClusterRank>,
    roce: Vec<Vec<super::roce_communicators::RoceCommunicator>>,
}

/// The rank this host serves of a model whose ranks run on separate hosts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterRank {
    pub world: usize,
    pub rank: usize,
    /// Names the export; every host must load the same one.
    pub export_id: String,
}

/// A RoCEnante transport a rank's graphs launch: the pinned region and device counters (restored
/// at these addresses and zeroed), the transport geometry the kernels were compiled for, and the
/// HCAs and per-peer rails preparation verified.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoceSpec {
    pub proxy_abi: u32,
    pub world: usize,
    pub rank: usize,
    pub region_address: u64,
    pub region_bytes: u64,
    pub counters_address: u64,
    pub counters_bytes: u64,
    pub slot_bytes: u64,
    pub pipe_bytes: u64,
    /// Per-rank capacity of the variable-length all-gather area (0 without one).
    pub varlen_bytes: u64,
    pub ctrl_offset: u64,
}

fn roce_spec(
    wire: protobuf::wire::RoceTransport,
    core: &super::format::Core,
) -> Result<RoceSpec> {
    let spec = RoceSpec {
        proxy_abi: wire.proxy_abi,
        world: wire.world as usize,
        rank: wire.rank as usize,
        region_address: wire.region_address,
        region_bytes: wire.region_bytes,
        counters_address: wire.counters_address,
        counters_bytes: wire.counters_bytes,
        slot_bytes: wire.slot_bytes,
        pipe_bytes: wire.pipe_bytes,
        varlen_bytes: wire.varlen_bytes,
        ctrl_offset: wire.ctrl_offset,
    };
    let within = |address: u64, nbytes: u64, memory: super::format::Memory| {
        core.allocations.iter().any(|a| {
            a.sizing.is_none()
                && std::mem::discriminant(&a.memory) == std::mem::discriminant(&memory)
                && a.address.is_some_and(|base| {
                    base <= address
                        && address
                            .checked_add(nbytes)
                            .is_some_and(|end| base.checked_add(a.nbytes as u64).is_some_and(|limit| end <= limit))
                })
        })
    };
    if !(2..=16).contains(&spec.world) || spec.rank >= spec.world {
        return Err(invalid("RoCE transport geometry is inconsistent"));
    }
    if spec.region_bytes == 0
        || spec
            .ctrl_offset
            .checked_add(4096)
            .is_none_or(|end| end > spec.region_bytes)
        || !within(
            spec.region_address,
            spec.region_bytes,
            super::format::Memory::MappedHost,
        )
        || !within(
            spec.counters_address,
            spec.counters_bytes,
            super::format::Memory::Device,
        )
    {
        return Err(invalid(
            "RoCE transport memory must lie in the rank's mapped host and device allocations",
        ));
    }
    Ok(spec)
}

pub(super) fn load_roce(
    bundle: &protobuf::wire::BundleIndex,
    core: &super::format::Core,
    cluster: Option<&ClusterRank>,
) -> Result<Vec<super::roce_communicators::RoceCommunicator>> {
    let normalized = super::roce_communicators::normalize(
        bundle.roce.clone(), bundle.roce_communicators.clone(), cluster,
    )?;
    // Payload descriptors carry topology only. The selected decoded core owns
    // the allocation inventory for both payload and legacy formats.
    normalized.into_iter().map(|comm| Ok(super::roce_communicators::RoceCommunicator {
        id: comm.id,
        members: comm.members.into_iter().map(|r| r as usize).collect(),
        transport: roce_spec(comm.transport.ok_or_else(|| invalid("missing transport"))?, core)?,
    })).collect()
}

#[cfg(test)]
mod roce_load_tests {
    use super::*;
    use super::super::{protection::payload_transport_descriptor, protobuf, format::Placements};
    use crate::core::schema::wire as payload;
    use protobuf::wire;

    fn transport(world: u32, base: u64) -> wire::RoceTransport {
        wire::RoceTransport {
            proxy_abi: 14, world, rank: 0, region_address: base, region_bytes: 8192,
            counters_address: base + 0x4000, counters_bytes: 4096,
            slot_bytes: 16, pipe_bytes: 16, ctrl_offset: 4096, varlen_bytes: 0,
        }
    }

    fn rank(repeated: bool) -> payload::RankBody {
        let mut rank = payload::RankBody::default();
        rank.context = Some(Default::default());
        let tp = transport(16, 0x10000);
        if repeated {
            rank.roce_communicators = vec![
                wire::RoceCommunicator { id: "tp".into(), members: (0..16).collect(), transport: Some(tp) },
                wire::RoceCommunicator { id: "dcp-a".into(), members: (0..8).collect(), transport: Some(transport(8, 0x20000)) },
            ];
        } else {
            rank.roce = Some(tp);
        }
        for base in if repeated { vec![0x10000, 0x20000] } else { vec![0x10000] } {
            for (address, nbytes, memory) in [
                (base, 8192, wire::Memory::MappedHost),
                (base + 0x4000, 4096, wire::Memory::Device),
            ] {
                rank.allocations.push(payload::Allocation {
                    name: format!("storage-{address}"), address: Some(address), nbytes,
                    alignment: 4096, memory: memory as i32, ..Default::default()
                });
            }
        }
        rank
    }

    fn cluster() -> ClusterRank {
        ClusterRank { world: 16, rank: 0, export_id: "load-regression".into() }
    }

    fn load(rank: &payload::RankBody) -> Result<Vec<super::super::roce_communicators::RoceCommunicator>> {
        // These are the production payload topology and ready_cores conversions.
        let descriptor = payload_transport_descriptor(rank, vec![12, 1]);
        assert!(descriptor.allocations.is_empty());
        let core = protobuf::payload_rank(rank, [12, 1], &Placements::default())?;
        load_roce(&descriptor, &core, Some(&cluster()))
    }

    #[test]
    fn payload_singular_and_tp16_dcp8_use_decoded_allocations() {
        assert_eq!(load(&rank(false)).unwrap().len(), 1);
        assert_eq!(load(&rank(true)).unwrap().len(), 2);
    }

    #[test]
    fn legacy_bundle_and_payload_have_equivalent_transport_ownership() {
        let rank = rank(false);
        let mut bundle = payload_transport_descriptor(&rank, vec![12, 1]);
        bundle.context = rank.context.clone();
        bundle.allocations = rank.allocations.iter().map(|a| wire::Allocation {
            name: a.name.clone(), address: a.address, nbytes: a.nbytes,
            alignment: a.alignment, memory: a.memory, ..Default::default()
        }).collect();
        let core = protobuf::decode(bundle.clone()).unwrap();
        let legacy = load_roce(&bundle, &core, Some(&cluster())).unwrap();
        assert_eq!(legacy[0].transport, load(&rank).unwrap()[0].transport);
    }

    #[test]
    fn payload_rejects_invalid_owned_storage_and_protocol_ranges() {
        for case in 0..8 {
            let mut rank = rank(false);
            match case {
                0 => rank.allocations[0].memory = wire::Memory::Device as i32,
                1 => rank.allocations[0].sizing = Some(wire::Sizing {
                    capacity: 0, unit_bytes: 4096, maximum: 2,
                }),
                2 => rank.allocations[1].nbytes -= 1,
                3 => rank.allocations[0].nbytes = u64::MAX,
                4 => rank.roce.as_mut().unwrap().region_address = u64::MAX - 4096,
                5 => rank.roce.as_mut().unwrap().counters_address = 0x11000,
                6 => rank.roce.as_mut().unwrap().ctrl_offset = u64::MAX,
                7 => rank.allocations[0].address = None,
                _ => unreachable!(),
            }
            assert!(load(&rank).is_err(), "invalid storage case {case}");
        }
        let mut rank = rank(true);
        rank.roce_communicators[1].transport = Some(transport(8, 0x10000));
        assert!(load(&rank).is_err());
    }
}

/// The cluster an authorized core's ranks form: (world, export id), or None
/// when every rank runs on one host.
pub fn core_cluster(core: &AuthorizedCore) -> Option<(usize, String)> {
    match &core.kind {
        AuthorizedKind::Legacy { index, .. } => index
            .cluster
            .as_ref()
            .map(|c| (c.world as usize, c.export_id.clone())),
        AuthorizedKind::Payload(payload) => {
            payload.cluster().map(|c| (c.world as usize, c.export_id))
        }
    }
}

/// What one rank's bundle needs from its GPU before its KV cache is allocated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankNeeds {
    /// The compute capability the bundle was prepared for, as (major, minor).
    pub architecture: [i32; 2],
    /// Device bytes of the bundle's fixed allocations; capacity-sized allocations add their KV
    /// and state storage at load.
    pub device_bytes: u64,
}

/// Every rank bundle's needs, in index order: one entry per host of a multi-host core, or
/// one per GPU of a single-host tensor-parallel core.
pub fn rank_needs(core: &AuthorizedCore) -> Vec<RankNeeds> {
    let index = match &core.kind {
        AuthorizedKind::Legacy { index, .. } => index,
        AuthorizedKind::Payload(payload) => return payload.rank_needs(),
    };
    index
        .ranks
        .iter()
        .map(|bundle| RankNeeds {
            architecture: [
                bundle.architecture.first().copied().unwrap_or(0),
                bundle.architecture.get(1).copied().unwrap_or(0),
            ],
            device_bytes: bundle
                .allocations
                .iter()
                .filter(|a| {
                    protobuf::wire::Memory::try_from(a.memory) == Ok(protobuf::wire::Memory::Device)
                })
                .map(|a| a.nbytes)
                .sum(),
        })
        .collect()
}

/// Bytes the core's serving contract reserves on each GPU beyond its allocations.
pub fn reserve_bytes(core: &AuthorizedCore) -> u64 {
    let index = match &core.kind {
        AuthorizedKind::Legacy { index, .. } => index,
        AuthorizedKind::Payload(payload) => return payload.reserve_bytes(),
    };
    index
        .contract
        .as_ref()
        .and_then(|c| c.reserve_bytes)
        .unwrap_or(0)
}

/// The byte ranges of an authorized multi-host core, in offset order, that the host serving
/// `rank` reads. A copy holding only these, at the same offsets and length, verifies completely
/// and loads as that rank.
pub fn rank_ranges(
    core: &AuthorizedCore,
    rank: usize,
) -> Result<Vec<super::archive::CoreFile>> {
    let (archive, index) = match &core.kind {
        AuthorizedKind::Legacy { archive, index } => (archive, index),
        AuthorizedKind::Payload(payload) => return payload.rank_ranges(rank),
    };
    if index.cluster.is_none() {
        return Err(invalid(
            "only a multi-host core has ranks for separate hosts",
        ));
    }
    archive.rank_ranges(index, rank)
}

/// Startup agreement between ranks: register and approve pool memory before choosing the
/// smallest cache capacity every host can back.
pub trait CacheAgreement {
    fn caches(&mut self, local: (u64, u64)) -> Result<(u64, u64)>;

    fn pool(
        &mut self,
        offer: crate::offload::shard::PoolOffer,
    ) -> Result<Option<crate::offload::shard::PoolPeers>> {
        if offer.enabled {
            return Err(invalid("a multi-host pool requires a rank rendezvous"));
        }
        Ok(None)
    }

    fn pool_approved(&mut self, approved: bool, _world: usize) -> Result<bool> {
        Ok(approved)
    }
}

impl<F: FnMut((u64, u64)) -> Result<(u64, u64)>> CacheAgreement for F {
    fn caches(&mut self, local: (u64, u64)) -> Result<(u64, u64)> {
        self(local)
    }
}

impl NativeModel {
    /// # Safety
    /// All rank bundles must be trusted and obey the CUDA execution contract.
    pub unsafe fn load(
        authorized: AuthorizedCore,
        devices: &[i32],
        options: &LoadOptions,
        progress: &StartupProgress,
    ) -> Result<Self> {
        unsafe { Self::load_agreeing(authorized, devices, options, progress, &mut Ok) }
    }

    /// Load an accepted core from either format, sizing caches through `agree`.
    ///
    /// # Safety
    /// All rank cores must be trusted and obey the CUDA execution contract.
    pub unsafe fn load_model_agreeing(
        core: super::ModelCore,
        devices: &[i32],
        options: &LoadOptions,
        progress: &StartupProgress,
        agree: &mut dyn CacheAgreement,
    ) -> Result<Self> {
        match core {
            super::ModelCore::Legacy(authorized) => unsafe {
                Self::load_agreeing(authorized, devices, options, progress, agree)
            },
            super::ModelCore::Payload(payload) => unsafe {
                Self::load_agreeing(
                    AuthorizedCore {
                        kind: AuthorizedKind::Payload(payload),
                    },
                    devices,
                    options,
                    progress,
                    agree,
                )
            },
        }
    }

    /// Load as [`NativeModel::load`] does, sizing caches through `agree`.
    ///
    /// # Safety
    /// All rank bundles must be trusted and obey the CUDA execution contract.
    pub unsafe fn load_agreeing(
        authorized: AuthorizedCore,
        devices: &[i32],
        options: &LoadOptions,
        progress: &StartupProgress,
        agree: &mut dyn CacheAgreement,
    ) -> Result<Self> {
        let device_labels = options.device_labels.as_deref().unwrap_or(devices);
        if device_labels.len() != devices.len() {
            return Err(invalid("GPU label count must match selected devices"));
        }
        let phase = progress.phase("Prepare model", None);
        let (mut descriptor, mut ready_cores, files) = match authorized.kind {
            AuthorizedKind::Legacy { archive, index } => (index, None, FilesRef::Legacy(archive)),
            AuthorizedKind::Payload(core) => {
                let descriptor = core.descriptor()?;
                let (source, cores) = core.into_parts();
                (descriptor, Some(cores), FilesRef::Payload(source))
            }
        };
        let root = files.path();
        // A multi-host core holds every rank; each host keeps only the bundle it serves, so
        // it never maps another rank's storage.
        let cluster = match (descriptor.cluster.take(), options.cluster_rank) {
            (Some(c), rank) => {
                let rank = rank.unwrap_or(0);
                // Verification checked that the core holds every rank of its cluster.
                if !(2..=16).contains(&(c.world as usize)) || c.export_id.is_empty() {
                    return Err(invalid(
                        "a multi-host core must hold every rank of its cluster",
                    ));
                }
                let position = files.rank_position(rank)?;
                if rank >= c.world as usize || position >= descriptor.ranks.len() {
                    return Err(invalid(format!(
                        "rank {rank} is outside the core's {} ranks",
                        c.world
                    )));
                }
                let bundle = descriptor.ranks.swap_remove(position);
                descriptor.ranks = vec![bundle];
                if let Some(cores) = ready_cores.as_mut()
                    && position < cores.len()
                {
                    // A cluster host serves exactly the rank it selected.
                    let selected = cores.swap_remove(position);
                    *cores = vec![selected];
                }
                Some(ClusterRank {
                    world: c.world as usize,
                    rank,
                    export_id: c.export_id,
                })
            }
            (None, Some(_)) => {
                return Err(invalid("--rank applies only to a multi-host core"));
            }
            (None, None) => None,
        };
        let transports: Vec<_> = descriptor
            .ranks
            .iter_mut()
            .map(|bundle| protobuf::wire::BundleIndex {
                roce: bundle.roce.take(),
                roce_communicators: std::mem::take(&mut bundle.roce_communicators),
                ..Default::default()
            })
            .collect();
        if devices.len() != descriptor.ranks.len()
            || devices.iter().collect::<HashSet<_>>().len() != devices.len()
        {
            return Err(invalid(
                "model rank count must match distinct selected devices",
            ));
        }
        let mut cores = match ready_cores {
            Some(cores) => cores,
            None => descriptor
                .ranks
                .drain(..)
                .map(protobuf::decode)
                .collect::<Result<Vec<_>>>()?,
        };
        if cores.is_empty() || cores.len() != devices.len() {
            return Err(invalid("model rank count differs from its cores"));
        }
        for core in &cores {
            core.validate()?;
            Files::validate(&files, core)?;
        }
        let roce = transports
            .into_iter()
            .zip(&cores)
            .map(|(bundle, core)| load_roce(&bundle, core, cluster.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        let mut server_config = protobuf::serving_contract(
            descriptor
                .contract
                .take()
                .ok_or_else(|| invalid("model core lacks a typed serving contract"))?,
            cores
                .first()
                .and_then(|core| core.execution.as_ref())
                .ok_or_else(|| invalid("model core has no serving contract"))?
                .max_active,
        )?;
        server_config["sampling_defaults"] = protobuf::sampling_defaults(
            descriptor
                .sampling_defaults
                .take()
                .ok_or_else(|| invalid("model core lacks sampling defaults"))?,
        )?;
        server_config["frontend"] = protobuf::frontend_settings(
            descriptor
                .frontend
                .take()
                .ok_or_else(|| invalid("model core lacks frontend settings"))?,
        )?;
        let mut assets = serde_json::Map::new();
        for asset in descriptor.assets {
            if !matches!(asset.name.as_str(), "tokenizer" | "chat_template") {
                return Err(invalid("unsupported runtime frontend asset"));
            }
            let span = asset
                .payload
                .ok_or_else(|| invalid("frontend asset lacks a payload"))?;
            let (_, offset) = files.source(&span)?;
            let asset_path = files.asset_path(&span)?;
            let value = serde_json::json!({ "path": asset_path, "offset": offset,
                "nbytes": span.nbytes, "sha256": asset.sha256 });
            if assets.insert(asset.name, value).is_some() {
                return Err(invalid("duplicate frontend asset"));
            }
        }
        server_config["assets"] = assets.into();
        let active = options.select(&mut cores, &mut server_config)?;
        let table_count = descriptor
            .engram
            .as_ref()
            .map_or(0, |catalog| catalog.tables.len());
        let mut engram: Vec<(crate::engram::Index, Vec<usize>)> = Vec::new();
        if let Some(catalog) = descriptor.engram.take() {
            for (local, core) in cores.iter().enumerate() {
                let (rank, world) = cluster
                    .as_ref()
                    .map_or((local, cores.len()), |c| (c.rank, c.world));
                let index = super::engram::decode(
                    catalog.clone(),
                    &files,
                    std::slice::from_ref(core),
                    rank,
                    world,
                )?;
                if let Some((_, ranks)) = engram.iter_mut().find(|(existing, _)| existing == &index)
                {
                    ranks.push(local);
                } else {
                    engram.push((index, vec![local]));
                }
            }
        }
        if options
            .embedding_table_cache_bytes
            .is_some_and(|bytes| bytes != 0)
            && engram.is_empty()
        {
            return Err(invalid(
                "embedding row cache requires an embedding table catalog",
            ));
        }
        if options.embedding_table_resident_scales == Some(true) && engram.is_empty() {
            return Err(invalid(
                "resident embedding scales require an embedding table catalog",
            ));
        }
        let ram = options.embedding_table_storage == Some(crate::engram::Storage::Ram);
        let engram_tokens = engram.iter().map(|(_, ranks)| {
            ranks.iter().flat_map(|rank| cores[*rank].execution.as_ref().unwrap().graphs.iter())
                .filter(|graph| crate::execution::GraphKind::try_from(graph.kind.as_str())
                    .is_ok_and(crate::engram::consumes))
                .map(|graph| super::protobuf::size(graph.packed_tokens()))
                .collect::<Result<Vec<_>>>()?.into_iter().max()
                .ok_or_else(|| invalid("embedding tables lack a consuming graph"))
        }).collect::<Result<Vec<_>>>()?;
        let staging_bytes = engram.iter().zip(&engram_tokens).try_fold(0u64,
            |total, ((index, _), tokens)| total.checked_add(crate::engram::staging_bytes(index, *tokens)?)
                .ok_or_else(|| invalid("embedding staging exceeds the memory budget")))?;
        let prepared: crate::model::PreparedContract =
            serde_json::from_value(server_config["model_contract"].clone())
                .map_err(|error| invalid(format!("invalid prepared contract: {error}")))?;
        if prepared
            .graphs
            .iter()
            .flat_map(|g| &g.embedding_tables)
            .any(|table| *table as usize >= table_count)
        {
            return Err(invalid("graph requires an absent embedding-table resource"));
        }
        let state_rows = server_config["state_rows"]
            .as_u64()
            .filter(|rows| *rows > 0 && *rows < i32::MAX as u64)
            .ok_or_else(|| invalid("model configuration requires a shared state-row pool"))?;
        for core in &cores {
            core.validate()?;
            let pool = core
                .execution
                .as_ref()
                .unwrap()
                .buffers
                .iter()
                .find(|buffer| buffer.name == "state_pool")
                .ok_or_else(|| invalid("rank lacks continuation pool geometry"))?;
            if pool.dtype != "uint8"
                || pool.shape.len() != 2
                || pool.shape[0] != state_rows + 1
                || pool.shape[1] == 0
                || pool.strides != [pool.shape[1], 1]
                || pool.nbytes != pool.shape[0].checked_mul(pool.shape[1]).unwrap_or(0)
            {
                return Err(invalid(
                    "rank continuation pool differs from its serving capacity",
                ));
            }
            let reference = pool
                .reference
                .as_ref()
                .filter(|r| r.offset == 0)
                .ok_or_else(|| invalid("continuation storage must start at its allocation base"))?;
            if !core.allocations.iter().any(|a| {
                a.name == reference.allocation
                    && a.sizing.as_ref().is_some_and(|s| {
                        s.capacity == 0
                            && s.unit_bytes == pool.shape[1]
                            && s.maximum >= pool.shape[0]
                    })
            }) {
                return Err(invalid(
                    "continuation storage requires a matching runtime-sized allocation",
                ));
            }
            let capacity = core
                .execution
                .as_ref()
                .unwrap()
                .buffers
                .iter()
                .find(|b| b.name == "memory_capacity")
                .filter(|b| {
                    b.dtype == "int64" && b.shape == [2] && b.strides == [1] && b.nbytes == 16
                })
                .and_then(|b| b.reference.as_ref())
                .ok_or_else(|| invalid("model requires a runtime memory capacity binding"))?;
            if !core.allocations.iter().any(|a| {
                a.name == capacity.allocation
                    && a.sizing.is_none()
                    && capacity
                        .offset
                        .checked_add(16)
                        .is_some_and(|n| n <= a.nbytes as u64)
            }) {
                return Err(invalid(
                    "runtime memory capacity must fit a fixed allocation",
                ));
            }
        }
        let mut allocations = Vec::new();
        let mut ranges = Vec::new();
        for core in &mut cores {
            // Peer allocations name storage another rank owns and are checked against the model index below.
            for allocation in core
                .allocations
                .iter()
                .filter(|a| !matches!(a.memory, Memory::Peer))
            {
                let address = allocation
                    .address
                    .ok_or_else(|| invalid("model allocations require recorded addresses"))?;
                let end = address
                    .checked_add(allocation.extent() as u64)
                    .ok_or_else(|| invalid("allocation range overflow"))?;
                allocations.push((address, end));
            }
            for reservation in core.reservations.drain(..) {
                ranges.push((
                    reservation.address,
                    reservation.address + reservation.nbytes as u64,
                ));
            }
        }
        allocations.sort_unstable();
        if allocations.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(invalid("rank allocation addresses overlap"));
        }
        ranges.sort_unstable();
        let mut merged: Vec<Reservation> = Vec::new();
        for (start, end) in ranges {
            if let Some(previous) = merged
                .last_mut()
                .filter(|r| start <= r.address + r.nbytes as u64)
            {
                previous.nbytes = (end.max(previous.address + previous.nbytes as u64)
                    - previous.address) as usize;
            } else {
                merged.push(Reservation {
                    address: start,
                    nbytes: (end - start) as usize,
                });
            }
        }
        cores[0].reservations = merged;
        validate_peer_memory(&cores, &descriptor.peer_memory)?;
        // Distribution and verification can fill shared DRAM before CUDA creates its context.
        Files::release_page_cache(&files);
        phase.finish(format!(
            "{} rank{}",
            devices.len(),
            if devices.len() == 1 { "" } else { "s" }
        ));
        // Windows direct reads can stall while any buffered handle remains open.
        // Keep every shard's write/delete exclusion through unbuffered guards;
        // the buffered startup readers close when loading returns.
        let core_files = files.retained_files()?;
        let mut model = Self {
            _core_files: core_files,
            ranks: Vec::new(),
            server_config,
            engram: Vec::new(),
            offload: Default::default(),
            cluster,
            roce,
        };
        unsafe {
            let before_contexts = if matches!(
                options.memory_limit,
                Some(super::MemoryLimit::Bytes(_) | super::MemoryLimit::Fraction(_))
            ) {
                Some(
                    devices
                        .iter()
                        .map(|&device| NativeResources::memory_before_context(device))
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                None
            };
            let phase = progress.phase(
                if devices.len() == 1 {
                    "Initialize GPU"
                } else {
                    "Initialize GPUs"
                },
                None,
            );
            for (core, device) in cores.iter().zip(devices) {
                model
                    .ranks
                    .push(NativeResources::initialize(core, *device)?);
            }
            phase.finish(
                cores
                    .iter()
                    .zip(device_labels)
                    .map(|(core, device)| {
                        format!(
                            "GPU {device} · SM{}{}",
                            core.architecture[0], core.architecture[1]
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" / "),
            );
            let embedding_table_resident_scales = options.embedding_resident_scales(
                !engram.is_empty(),
                model.ranks.iter().any(|rank| rank.integrated),
            );
            let resident_bytes = engram
                .iter()
                .flat_map(|(index, _)| &index.tables)
                .flat_map(|table| table.planes.iter().enumerate())
                .filter(|(ordinal, _)| {
                    ram || (embedding_table_resident_scales && *ordinal == 1)
                })
                .try_fold(0u64, |bytes, (_, plane)| {
                    bytes
                        .checked_add(plane.resident_bytes()? as u64)
                        .ok_or_else(|| invalid("resident embedding tables exceed the memory budget"))
                })?;
            let embedding_table_cache_bytes = options.embedding_cache_bytes(
                !engram.is_empty(),
                model.ranks.iter().any(|rank| rank.integrated),
            );
            let embedding_host_bytes = resident_bytes
                .checked_add(staging_bytes)
                .and_then(|bytes| bytes.checked_add(embedding_table_cache_bytes))
                .ok_or_else(|| {
                    invalid("embedding tables, staging, and row cache exceed the memory budget")
                })?;
            let initial = match before_contexts {
                Some(initial) => initial,
                None => model
                    .ranks
                    .iter()
                    .map(|rank| rank.memory_snapshot())
                    .collect::<Result<Vec<_>>>()?,
            };
            model.check_memory_limit(&cores, options, &initial, embedding_host_bytes)?;
            let (mut gpu_bytes, mut host_bytes, mut payload_bytes) = (0u64, 0u64, 0u64);
            for allocation in cores.iter().flat_map(|core| &core.allocations) {
                match allocation.memory {
                    Memory::Device => gpu_bytes += allocation.nbytes as u64,
                    Memory::MappedHost => host_bytes += allocation.nbytes as u64,
                    Memory::Peer => {}
                }
                payload_bytes += allocation.stored_bytes() as u64;
            }
            let phase = progress.phase("Allocate model memory", Some(gpu_bytes + host_bytes));
            // Reservations cover all ranks and must exist before parallel mapping.
            model.ranks[0].reserve(&cores[0])?;
            model.parallel(&cores, &files, |rank, core, _| {
                rank.allocate(core, &phase)
            })?;
            phase.finish(format!(
                "{:.1} GiB GPU · {:.1} GiB host",
                gpu_bytes as f64 / (1u64 << 30) as f64,
                host_bytes as f64 / (1u64 << 30) as f64,
            ));
            let phase = progress.phase("Load weights", Some(payload_bytes));
            phase.advance_bytes(0);
            model.parallel(&cores, &files, |rank, core, path| {
                rank.load_payload(path, core, &phase, devices.len())
            })?;
            phase.finish(format!(
                "{:.1} GiB",
                payload_bytes as f64 / (1u64 << 30) as f64
            ));
            model.grant_peer_access(&descriptor.peer_memory)?;
            let kernels = cores
                .iter()
                .map(|p| p.functions.len() as u64)
                .sum::<u64>();
            let modules = cores.iter().map(|p| p.modules.len() as u64).sum::<u64>();
            let phase = progress.phase("Prepare kernels", Some(kernels + modules));
            model.parallel(&cores, &files, |rank, core, path| {
                rank.prepare_kernels(path, core, &phase)
            })?;
            phase.finish(format!("{kernels} kernels"));
            let nodes = cores
                .iter()
                .flat_map(|p| &p.graphs)
                .map(|graph| graph.node_count as u64 + 1)
                .sum();
            let phase = progress.phase("Prepare execution", Some(nodes));
            model.parallel(&cores, &files, |rank, core, files| {
                rank.prepare_graphs(files, core, &phase)
            })?;
            phase.finish(format!("concurrency {active}"));
            if !engram.is_empty() {
                let phase = progress.phase("Prepare embedding tables", None);
                let storage = options.embedding_table_storage.unwrap_or_default();
                let group_budget = embedding_table_cache_bytes / engram.len() as u64;
                // Open direct-I/O handles only for containers that actually carry embedding
                // planes rather than every weight companion in the deployment.
                let mut mapped_files = std::collections::BTreeMap::new();
                let mut direct_files = Vec::new();
                for (index, _) in &mut engram {
                    for table in &mut index.tables {
                        for plane in &mut table.planes {
                            let target = match mapped_files.entry(plane.file) {
                                std::collections::btree_map::Entry::Occupied(entry) => {
                                    *entry.get()
                                }
                                std::collections::btree_map::Entry::Vacant(slot) => {
                                    let next = direct_files.len();
                                    direct_files.push(files.direct_file(plane.file)?);
                                    *slot.insert(next)
                                }
                            };
                            plane.file = target;
                        }
                    }
                }
                let mut bindings = Vec::new();
                for ((index, ranks), tokens) in engram.into_iter().zip(engram_tokens) {
                    let first = ranks[0];
                    let shared = crate::engram::Shared::new(
                        index,
                        &direct_files,
                        ranks.len(),
                        tokens,
                        model.ranks[first].context as usize,
                        devices[first],
                        storage,
                        group_budget,
                        options.embedding_workers(),
                        embedding_table_resident_scales,
                    )?;
                    for (delivery_rank, local) in ranks.into_iter().enumerate() {
                        bindings.push((local, shared.clone(), delivery_rank));
                    }
                }
                bindings.sort_by_key(|(local, _, _)| *local);
                model.engram = bindings
                    .into_iter()
                    .map(|(_, shared, rank)| (shared, rank))
                    .collect();
                let label = match storage {
                    crate::engram::Storage::Disk if embedding_table_resident_scales => {
                        "disk weights, RAM scales"
                    }
                    crate::engram::Storage::Disk => "disk",
                    crate::engram::Storage::Ram => "RAM",
                };
                let cache_bytes: u64 = model
                    .engram
                    .iter()
                    .filter(|(_, rank)| *rank == 0)
                    .map(|(shared, _)| shared.cache_bytes())
                    .sum();
                phase.finish(if cache_bytes == 0 {
                    label.to_owned()
                } else {
                    format!(
                        "{label} · {:.0} MiB row cache",
                        cache_bytes as f64 / (1u64 << 20) as f64
                    )
                });
            }
            Files::release_page_cache(&files);
            model.allocate_caches(
                &mut cores,
                progress,
                active,
                agree,
                options,
                &initial,
                embedding_host_bytes,
            )?;
            model.grant_peer_access(&descriptor.peer_memory)?;
            model.prepare_offload(root, options, progress, active)?;
        }
        Ok(model)
    }

    /// Resolve the host prefix store configuration, map and register its
    /// pools, and scan its record directory.
    unsafe fn prepare_offload(
        &mut self,
        root: &Path,
        options: &LoadOptions,
        progress: &StartupProgress,
        max_active: usize,
    ) -> Result<()> {
        if !options.prefix_cache.enabled() {
            return Ok(());
        }
        let phase = progress.phase("Prepare host prefix cache", None);
        let identity = offload_identity(&self.server_config)?;
        let geometry = self
            .ranks
            .iter()
            .map(|rank| {
                rank.cache_geometry()
                    .cloned()
                    .ok_or_else(|| invalid("rank lacks cache geometry"))
            })
            .collect::<Result<Vec<_>>>()?;
        let contexts: Vec<_> = self
            .ranks
            .iter()
            .map(|rank| rank.context_handle())
            .collect();
        let contract: crate::model::PreparedContract =
            serde_json::from_value(self.server_config["model_contract"].clone())
                .map_err(|error| invalid(format!("invalid prepared contract: {error}")))?;
        let manifest = self.ranks[0].execution_manifest()?;
        let staging = crate::offload::input_staging_bytes(&contract, |name| {
            manifest
                .buffers
                .iter()
                .find(|buffer| buffer.name == name)
                .map(|buffer| buffer.nbytes as u64)
        })?;
        let shard = self
            .cluster
            .as_ref()
            .map(|cluster| (cluster.rank, cluster.world));
        let prepared = crate::offload::prepare_with_inputs(
            &options.prefix_cache,
            root,
            identity,
            geometry,
            contexts,
            max_active,
            staging,
            shard,
        )?;
        let summary = self.offload.attach(prepared, &crate::offload::transfer::Copiers)?;
        phase.finish(summary);
        Ok(())
    }

    unsafe fn grant_peer_access(&self, peers: &[protobuf::wire::PeerMemory]) -> Result<()> {
        for peer in peers {
            let rank = &self.ranks[peer.owner as usize];
            let allocation = rank
                .allocations
                .values()
                .find(|a| a.address == peer.address && matches!(a.memory, Memory::Device))
                .ok_or_else(|| invalid("shared allocation is missing from its owner"))?;
            if allocation.nbytes == 0 {
                continue;
            }
            let access: Vec<_> = peer
                .readers
                .iter()
                .map(|reader| Access {
                    location: Location {
                        kind: 1,
                        device: self.ranks[*reader as usize].device,
                    },
                    flags: 3,
                })
                .collect();
            unsafe {
                rank.driver
                    .check("cuCtxSetCurrent", (rank.driver.context_set)(rank.context))?;
                rank.driver.check(
                    "cuMemSetAccess",
                    (rank.driver.access)(
                        peer.address,
                        allocation.nbytes,
                        access.as_ptr(),
                        access.len(),
                    ),
                )?;
            }
        }
        Ok(())
    }

    fn parallel<F: super::payload::Files>(
        &mut self,
        cores: &[Core],
        archive: &F,
        operation: impl Fn(&mut NativeResources, &Core, &F) -> Result<()> + Sync,
    ) -> Result<()> {
        thread::scope(|scope| {
            let mut workers = Vec::new();
            for (index, (rank, core)) in self.ranks.iter_mut().zip(cores).enumerate() {
                let operation = &operation;
                workers.push(
                    thread::Builder::new()
                        .name(format!("flare-load-{index}"))
                        .spawn_scoped(scope, move || {
                            operation(rank, core, archive)
                                .map_err(|error| invalid(format!("rank {index}: {error}")))
                        })
                        .map_err(|error| error.to_string())?,
                );
            }
            // Join every rank before any error can release shared reservations.
            let results: Vec<_> = workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|_| Err(invalid("rank loader panicked")))
                })
                .collect();
            results.into_iter().collect()
        })
    }

    pub fn manifests(&self) -> Result<Vec<ExecutionManifest>> {
        self.ranks
            .iter()
            .enumerate()
            .map(|(index, rank)| {
                let mut manifest = rank.execution_manifest()?;
                manifest.offload = self.offload.binding(index);
                if let Some((shared, delivery_rank)) = self.engram.get(index) {
                    let events = shared
                        .index
                        .tables
                        .iter()
                        .map(|t| {
                            rank.events
                                .get(&t.event)
                                .map(|event| *event as usize)
                                .ok_or_else(|| invalid("Engram readiness event missing"))
                        })
                        .collect::<Result<_>>()?;
                    manifest.engram = Some(crate::engram::Binding {
                        shared: shared.clone(),
                        rank: *delivery_rank,
                        events,
                    });
                }
                Ok(manifest)
            })
            .collect()
    }

    pub fn server_config(&self) -> &serde_json::Value {
        &self.server_config
    }

    /// This file's rank of a multi-host model, or None when every rank is local.
    pub fn cluster(&self) -> Option<&ClusterRank> {
        self.cluster.as_ref()
    }

    /// The host prefix store this process prepared, before the model worker opens it.
    pub fn offload(&self) -> Option<&crate::offload::Shared> {
        self.offload.shared()
    }

    /// Compatibility accessor for one full-cluster transport; never select the
    /// first of several communicators or reinterpret a subgroup as the cluster.
    pub fn roce(&self, rank: usize) -> Option<&RoceSpec> {
        let comms = self.roce.get(rank)?;
        let cluster = self.cluster.as_ref()?;
        if comms.len() != 1 || comms[0].members != (0..cluster.world).collect::<Vec<_>>() {
            return None;
        }
        Some(&comms[0].transport)
    }

    /// All independently owned communicators; cluster bootstrap must connect every entry.
    pub fn roce_communicators(&self, rank: usize) -> Option<&[super::roce_communicators::RoceCommunicator]> {
        self.roce.get(rank).map(Vec::as_slice)
    }
}

impl Drop for NativeModel {
    fn drop(&mut self) {
        self.offload.reset();
        self.engram.clear();
        // Rank zero owns the shared reservations and releases them after every peer.
        while self.ranks.pop().is_some() {}
    }
}

/// Namespace host records by model and the continuation state maintained by selected graphs.
pub(super) fn offload_identity(config: &serde_json::Value) -> Result<[u8; 32]> {
    let text = config["identity"]
        .as_str()
        .ok_or_else(|| invalid("model core lacks an identity"))?;
    if text.len() != 64 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid("model identity must be 64 hexadecimal characters"));
    }
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .map_err(|_| invalid("model identity must be hexadecimal"))?;
    }
    let widths = config["speculative_widths"]
        .as_array()
        .ok_or_else(|| invalid("selected speculative_widths must be an array"))?;
    let state = if widths.is_empty() {
        0u8
    } else {
        match config["speculative_proposal_source"]
            .as_str()
            .unwrap_or("mtp")
        {
            "mtp" => 1,
            "copy" => 2,
            _ => return Err(invalid("unknown speculative proposal source")),
        }
    };
    let mut hash = Blake2b::<U32>::new();
    hash.update(b"flare-prefix-state-v1\0");
    hash.update(digest);
    hash.update([state]);
    Ok(hash.finalize().into())
}

fn validate_peer_memory(cores: &[Core], peers: &[protobuf::wire::PeerMemory]) -> Result<()> {
    let mut shared = HashSet::new();
    for peer in peers {
        let owner = cores
            .get(peer.owner as usize)
            .ok_or_else(|| invalid("invalid peer-memory owner"))?;
        if !shared.insert(peer.address)
            || !peer.readers.contains(&peer.owner)
            || peer.readers.iter().collect::<HashSet<_>>().len() != peer.readers.len()
            || peer.readers.is_empty()
            || peer.readers.iter().any(|r| *r as usize >= cores.len())
            || !owner.allocations.iter().any(|a| {
                a.address == Some(peer.address)
                    && a.extent() as u64 == peer.nbytes
                    && matches!(a.memory, Memory::Device)
            })
        {
            return Err(invalid(
                "peer memory must name one owned device allocation and valid readers",
            ));
        }
    }
    for (rank, core) in cores.iter().enumerate() {
        for allocation in core
            .allocations
            .iter()
            .filter(|a| matches!(a.memory, Memory::Peer))
        {
            if !peers.iter().any(|peer| {
                peer.owner as usize != rank
                    && Some(peer.address) == allocation.address
                    && peer.nbytes == allocation.nbytes as u64
                    && peer.readers.contains(&(rank as u32))
            }) {
                return Err(invalid(
                    "peer allocation must name another rank's shared memory this rank reads",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use protobuf::wire;

    #[test]
    fn disabled_offload_needs_no_model_metadata_or_cuda_resources() {
        let progress = StartupProgress::silent();
        let mut model = NativeModel {
            _core_files: Vec::new(),
            ranks: Vec::new(),
            server_config: serde_json::Value::Null,
            engram: Vec::new(),
            offload: Default::default(),
            cluster: None,
            roce: Vec::new(),
        };
        for prefix_cache in [
            crate::offload::PrefixCacheOffload::default(),
            crate::offload::PrefixCacheOffload {
                memory_bytes: Some(0),
                disk: Some("unopened-prefix-cache".into()),
                disk_limit: Some(0),
                drain_on_shutdown: Default::default(),
                max_age: std::time::Duration::ZERO,
                pool: None,
            },
        ] {
            let options = LoadOptions {
                prefix_cache,
                ..Default::default()
            };
            unsafe {
                model
                    .prepare_offload(Path::new("unopened-core"), &options, &progress, 1)
                    .unwrap();
            }
            assert!(model.offload.shared().is_none());
        }
        let options = LoadOptions {
            prefix_cache: crate::offload::PrefixCacheOffload {
                memory_bytes: Some(1 << 30),
                ..Default::default()
            },
            ..Default::default()
        };
        let error =
            unsafe { model.prepare_offload(Path::new("unopened-core"), &options, &progress, 1) }
                .unwrap_err();
        assert!(error.to_string().contains("lacks an identity"));
    }

    #[test]
    fn peer_ownership_covers_virtual_extents_before_cache_backing() {
        let address = 0x200000000000;
        for dynamic in [false, true] {
            let extent = if dynamic { 8 << 20 } else { 2 << 20 };
            let allocation = wire::Allocation {
                name: "cache".into(),
                address: Some(address),
                alignment: 65536,
                nbytes: if dynamic { 0 } else { extent },
                sizing: dynamic.then_some(wire::Sizing {
                    capacity: 1,
                    unit_bytes: 2 << 20,
                    maximum: 4,
                }),
                ..Default::default()
            };
            let import = wire::Allocation {
                name: "peer-cache".into(),
                address: Some(address),
                alignment: 65536,
                nbytes: extent,
                memory: wire::Memory::Peer as i32,
                ..Default::default()
            };
            let cores: Vec<_> = [allocation, import]
                .into_iter()
                .map(|allocation| {
                    protobuf::decode(wire::BundleIndex {
                        architecture: vec![12, 0],
                        context: Some(Default::default()),
                        allocations: vec![allocation],
                        ..Default::default()
                    })
                    .unwrap()
                })
                .collect();
            let mut peers = vec![wire::PeerMemory {
                owner: 0,
                address,
                nbytes: extent,
                readers: vec![0, 1],
            }];
            validate_peer_memory(&cores, &peers).unwrap();
            peers[0].nbytes /= 2;
            assert!(validate_peer_memory(&cores, &peers).is_err());
            peers[0].nbytes = extent;
            peers[0].readers.pop();
            assert!(validate_peer_memory(&cores, &peers).is_err());
        }
    }
}

/// Verified identity of an authorized core, taken from signed metadata only.
///
/// Distribution uses this to place a core by its actual model, hardware, and
/// tensor-parallel deployment rather than by file names or unverified claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreIdentity {
    /// Signed model identity, e.g. `deepseek-v4.1-flash`.
    pub model: String,
    /// Signed hardware identity when the core records one.
    pub hardware: String,
    /// Selected deployment's tensor-parallel size.
    pub tensor_parallel: u32,
    /// Selected deployment's compute capabilities as `[major, minor]`.
    pub architecture: Vec<i32>,
    /// `true` when the core uses the payload format.
    pub payload_format: bool,
    /// Signed multi-host topology, when the core declares a cluster.
    pub cluster: Option<ClusterIdentity>,
}

/// Signed cluster topology of a prepared deployment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterIdentity {
    pub world: u32,
    pub export: String,
}

/// Identity of an authenticated legacy model index.
///
/// Only signed index fields are read: the contract's model id, the rank count as
/// the tensor-parallel size, and the first rank's compute capability. Legacy
/// cores carry no separate hardware identity, so the field stays empty rather
/// than borrowing the core identity string.
pub(crate) fn legacy_identity(
    index: &crate::runtime_core::protobuf::wire::ModelIndex,
) -> crate::Result<CoreIdentity> {
    let contract = index.contract.as_ref();
    let model = contract
        .map(|contract| contract.model_id.clone())
        .filter(|model| !model.is_empty())
        .ok_or_else(|| crate::Error::from("legacy core has no model identity"))?;
    let architecture = index
        .ranks
        .first()
        .map(|rank| rank.architecture.clone())
        .unwrap_or_default();
    // A copy holding one separately served rank of a multi-host core carries
    // only that rank's bundle, so the signed cluster world size is the deployment's
    // tensor-parallel size. Cores without a cluster keep the rank count.
    let tensor_parallel = match index.cluster.as_ref() {
        Some(cluster) if cluster.world != 0 => cluster.world,
        _ => u32::try_from(index.ranks.len())
            .map_err(|_| crate::Error::from("legacy core rank count overflow"))?,
    };
    Ok(CoreIdentity {
        model,
        hardware: String::new(),
        tensor_parallel,
        architecture,
        payload_format: false,
        cluster: index.cluster.as_ref().map(|cluster| ClusterIdentity {
            world: cluster.world,
            export: cluster.export_id.clone(),
        }),
    })
}

impl AuthorizedCore {
    /// The verified identity of this core's selected deployment.
    pub fn identity(&self) -> crate::Result<CoreIdentity> {
        match &self.kind {
            crate::runtime_core::protection::AuthorizedKind::Payload(core) => {
                let (model, hardware, tensor_parallel, architecture) = core.identity_parts()?;
                Ok(CoreIdentity {
                    model,
                    hardware,
                    tensor_parallel,
                    architecture,
                    payload_format: true,
                    cluster: core.cluster_identity(),
                })
            }
            crate::runtime_core::protection::AuthorizedKind::Legacy { index, .. } => {
                legacy_identity(index)
            }
        }
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use crate::core::{PayloadId, container, export, schema::wire};
    use crate::runtime_core::Publisher;
    use ring::signature::{Ed25519KeyPair, KeyPair};

    fn published(
        dir: &std::path::Path,
        model: &str,
        hardware: &str,
        tp: u32,
    ) -> (Publisher, Vec<u8>) {
        let payload = b"identity payload".to_vec();
        let id = PayloadId::of(&payload);
        let raw = dir.join("weights.bin");
        std::fs::write(&raw, &payload).unwrap();
        let graph = dir.join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Identity".into(),
            namespace: "identity".into(),
            public_key,
            development: false,
        };
        let contract = crate::runtime_core::wire::ServingContract {
            model_id: model.into(),
            served_names: vec!["recipe-alias".into()],
            identity: format!("{model}-{hardware}"),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(crate::runtime_core::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![crate::runtime_core::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: format!("{hardware}-tp{tp}"),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: payload.len() as u64,
                alignment: 4096,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: payload.len() as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = export::PublishRequest {
            core: wire::CoreBody {
                model_identity: model.into(),
                hardware_identity: hardware.into(),
                frontend: Some(crate::runtime_core::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    crate::runtime_core::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "identity".into(),
                    official_source: "https://huggingface.co/identity/model".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![export::PublishDeployment {
                payloads: vec![],
                body: wire::DeploymentBody {
                    deployment_id: format!("{hardware}-tp{tp}"),
                    architecture: vec![12, 0],
                    tensor_parallel: tp,
                    contract: Some(contract),
                    ..Default::default()
                },
                ranks: (0..tp)
                    .map(|index| export::PublishRank {
                        body: wire::RankBody {
                            rank: index,
                            ..rank.clone()
                        },
                        payloads: if index == 0 {
                            vec![export::PublishPayload {
                                role: wire::PayloadRole::Weights,
                                relative: "weights/layer-000.safetensors".into(),
                                tensor: Some(container::COMPANION_TENSOR.into()),
                                source: container::Slice::new(
                                    std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                                    0,
                                    payload.len() as u64,
                                )
                                .unwrap(),
                                id,
                            }]
                        } else {
                            Vec::new()
                        },
                        resources: vec![export::PublishResource {
                            kind: wire::ExecutableKind::Graph,
                            path: graph.clone(),
                            nbytes: 11,
                        }],
                    })
                    .collect(),
            }],
            core_id: [21u8; 32],
            public_key,
            accepted: "identity".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(dir)).unwrap();
        let metadata = crate::core::reader::Metadata::open(dir).unwrap();
        (publisher, metadata.envelope)
    }

    #[test]
    fn legacy_identity_reads_the_signed_index_fields() {
        let mut index = crate::runtime_core::protobuf::wire::ModelIndex {
            version: 4,
            ..Default::default()
        };
        index.contract = Some(crate::runtime_core::wire::ServingContract {
            model_id: "qwen3.8-flash".into(),
            identity: "core-identity".into(),
            ..Default::default()
        });
        for (architecture, rank) in [(vec![12, 0], 0u32), (vec![12, 0], 1)] {
            let mut bundle = crate::runtime_core::protobuf::wire::BundleIndex::default();
            bundle.architecture = architecture;
            bundle.execution = None;
            index.ranks.push(bundle);
            let _ = rank;
        }
        // Architecture is recorded per rank's bundle header in the index itself.
        let identity = legacy_identity(&index).unwrap();
        assert_eq!(identity.model, "qwen3.8-flash");
        assert!(!identity.payload_format);
        assert_eq!(identity.tensor_parallel, 2);
        assert!(
            identity.hardware.is_empty(),
            "legacy has no hardware identity"
        );

        // A separately served rank of a multi-host legacy core holds one rank
        // bundle; the signed cluster world size is the deployment's TP.
        let mut single = index.clone();
        single.ranks.truncate(1);
        single.cluster = Some(crate::runtime_core::protobuf::wire::Cluster {
            world: 4,
            export_id: "cluster".into(),
        });
        let identity = legacy_identity(&single).unwrap();
        assert_eq!(identity.tensor_parallel, 4, "single-host copy reported 1");

        // A legacy index without a signed contract model id is rejected, never guessed.
        index.contract = None;
        assert!(legacy_identity(&index).is_err());
    }

    #[test]
    fn payload_identity_comes_from_signed_metadata() {
        let scratch = std::env::temp_dir().join(format!("helios-identity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch).unwrap();
        let (publisher, _envelope) = published(&scratch, "deepseek-v4.1-flash", "gb202", 2);
        let verified = crate::runtime_core::VerifiedCore::open(
            &scratch,
            std::slice::from_ref(&publisher),
            None,
        )
        .unwrap();
        let authorized = verified.authorize("identity", false).unwrap();
        assert_eq!(authorized.model_names().unwrap(), vec!["deepseek-v4.1-flash", "recipe-alias"]);
        let identity = authorized.identity().unwrap();
        assert_eq!(identity.model, "deepseek-v4.1-flash");
        assert_eq!(identity.hardware, "gb202");
        assert_eq!(identity.tensor_parallel, 2);
        assert_eq!(identity.architecture, vec![12, 0]);
        assert!(identity.payload_format);

        // A single served rank of a multi-host deployment still reports the full
        // deployment TP, both before and after authorization.
        let single = crate::runtime_core::VerifiedCore::open(
            &scratch,
            std::slice::from_ref(&publisher),
            Some(1),
        )
        .unwrap();
        let before = single.identity().unwrap();
        assert_eq!(before.model, "deepseek-v4.1-flash");
        assert_eq!(before.tensor_parallel, 2);
        assert!(before.cluster.is_none());
        let single = single.authorize("identity", false).unwrap();
        let after = single.identity().unwrap();
        assert_eq!(after.tensor_parallel, 2);
        assert_eq!(
            after,
            CoreIdentity {
                cluster: None,
                ..before
            }
        );
        let _ = std::fs::remove_dir_all(&scratch);
    }
}
