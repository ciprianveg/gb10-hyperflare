//! Signed communicator membership and cluster-wide bootstrap record validation.
use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use super::{ClusterRank, RoceSpec, wire};
use super::format::invalid;
use crate::Result;

#[derive(Clone, Debug)]
pub struct RoceCommunicator {
    pub id: String,
    pub members: Vec<usize>,
    pub transport: RoceSpec,
}

/// The supported cluster bootstrap requires a transport before any replay.
/// This is also the legacy singular-descriptor missing-transport requirement.
pub fn require_cluster_transport<T>(value: Option<T>) -> Result<T> {
    value.ok_or_else(|| invalid("multi-host core rank has no RoCE transport"))
}

/// Normalize old full-cluster descriptors without changing their geometry.
pub fn normalize(
    legacy: Option<wire::RoceTransport>,
    mut explicit: Vec<wire::RoceCommunicator>,
    cluster: Option<&ClusterRank>,
) -> Result<Vec<wire::RoceCommunicator>> {
    if legacy.is_some() && !explicit.is_empty() {
        return Err(invalid("singular and repeated RoCE descriptors are mutually exclusive"));
    }
    if legacy.is_none() && explicit.is_empty() {
        return Ok(explicit);
    }
    let cluster = cluster.ok_or_else(|| invalid("RoCE communicators require a signed cluster identity"))?;
    if !(2..=16).contains(&cluster.world) || cluster.rank >= cluster.world {
        return Err(invalid("invalid signed cluster identity"));
    }
    if let Some(transport) = legacy {
        if (transport.world as usize, transport.rank as usize) != (cluster.world, cluster.rank) {
            return Err(invalid("legacy RoCE transport differs from full-cluster identity"));
        }
        explicit.push(wire::RoceCommunicator {
            id: "legacy-full-cluster".into(),
            members: (0..cluster.world as u32).collect(),
            transport: Some(transport),
        });
    }
    let mut ids = BTreeSet::new();
    let mut ranges = Vec::new();
    for comm in &explicit {
        let t = comm.transport.as_ref().ok_or_else(|| invalid("communicator lacks transport geometry"))?;
        if comm.id.trim().is_empty() || !ids.insert(&comm.id) {
            return Err(invalid("RoCE communicator IDs must be nonempty and unique per rank"));
        }
        if !(2..=16).contains(&comm.members.len())
            || t.world as usize != comm.members.len()
            || comm.members.iter().any(|&r| r as usize >= cluster.world)
            || comm.members.iter().collect::<BTreeSet<_>>().len() != comm.members.len()
            || comm.members.get(t.rank as usize).copied() != Some(cluster.rank as u32)
        {
            return Err(invalid("communicator membership/local rank differs from signed cluster identity"));
        }
        for (start, bytes) in [(t.region_address, t.region_bytes), (t.counters_address, t.counters_bytes)] {
            let end = start.checked_add(bytes).filter(|_| bytes > 0)
                .ok_or_else(|| invalid("RoCE protocol range is empty or overflows"))?;
            if ranges.iter().any(|&(a, b)| start < b && a < end) {
                return Err(invalid("RoCE communicator protocol regions overlap"));
            }
            ranges.push((start, end));
        }
    }
    explicit.sort_by(|a,b| a.id.cmp(&b.id));
    Ok(explicit)
}

/// Protocol storage must be owned, fixed-address mapped-host/device storage.
pub fn validate_owned_ranges(comms: &[wire::RoceCommunicator], allocations: &[wire::Allocation]) -> Result<()> {
    for comm in comms {
        let t = comm.transport.as_ref().ok_or_else(|| invalid("communicator lacks transport geometry"))?;
        for (address, bytes, memory) in [
            (t.region_address,t.region_bytes,wire::Memory::MappedHost as i32),
            (t.counters_address,t.counters_bytes,wire::Memory::Device as i32),
        ] {
            let end = address.checked_add(bytes).filter(|_| bytes > 0)
                .ok_or_else(|| invalid("invalid communicator protocol bounds"))?;
            if !allocations.iter().any(|a| a.memory == memory && a.sizing.is_none()
                && a.address.is_some_and(|base| base <= address
                    && base.checked_add(a.nbytes).is_some_and(|limit| end <= limit))) {
                return Err(invalid("RoCE protocol range is outside fixed owned allocations"));
            }
        }
        if t.ctrl_offset.checked_add(4096).is_none_or(|end| end > t.region_bytes) {
            return Err(invalid("RoCE control page is outside the region"));
        }
    }
    Ok(())
}

/// Address-independent geometry must agree among every communicator's members.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    proxy_abi: u32,
    world: usize,
    region_bytes: u64,
    counters_bytes: u64,
    slot_bytes: u64,
    pipe_bytes: u64,
    varlen_bytes: u64,
    ctrl_offset: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Declaration {
    pub id: String,
    pub members: Vec<usize>,
    pub local_rank: usize,
    pub geometry: Geometry,
}

impl From<&RoceCommunicator> for Declaration {
    fn from(comm: &RoceCommunicator) -> Self {
        let t = &comm.transport;
        Self { id: comm.id.clone(), members: comm.members.clone(), local_rank: t.rank,
            geometry: Geometry { proxy_abi:t.proxy_abi, world:t.world,
                region_bytes:t.region_bytes,counters_bytes:t.counters_bytes,
                slot_bytes:t.slot_bytes,pipe_bytes:t.pipe_bytes,varlen_bytes:t.varlen_bytes,ctrl_offset:t.ctrl_offset } }
    }
}

/// Every rank participates in the existing full-cluster exchange, even when it
/// does not belong to a subgroup. No independently waiting subgroup is created.
pub fn agree(records: &[Vec<Declaration>]) -> Result<BTreeMap<String, Declaration>> {
    let world = records.len();
    if !(2..=16).contains(&world) {
        return Err(invalid("communicator exchange requires the full cluster"));
    }
    let mut catalog = BTreeMap::<String, Declaration>::new();
    for (rank, declarations) in records.iter().enumerate() {
        require_cluster_transport((!declarations.is_empty()).then_some(()))?;
        let mut ids = BTreeSet::new();
        for d in declarations {
            if d.id.trim().is_empty() || !ids.insert(&d.id) || d.geometry.world != d.members.len()
                || !(2..=world).contains(&d.members.len())
                || d.members.iter().any(|&r| r >= world)
                || d.members.iter().collect::<BTreeSet<_>>().len() != d.members.len()
                || d.members.get(d.local_rank) != Some(&rank)
            {
                return Err(invalid("invalid communicator bootstrap declaration"));
            }
            if let Some(old) = catalog.get(&d.id) {
                if old.members != d.members || old.geometry != d.geometry {
                    return Err(invalid(format!("communicator {} membership or geometry differs", d.id)));
                }
            } else {
                catalog.insert(d.id.clone(), d.clone());
            }
        }
    }
    for d in catalog.values() {
        for (rank, row) in records.iter().enumerate() {
            if row.iter().any(|r| r.id == d.id) != d.members.contains(&rank) {
                return Err(invalid(format!("communicator {} has missing or nonmember declarations", d.id)));
            }
        }
    }
    Ok(catalog)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Connection {
    pub id: String,
    pub local_rank: usize,
    pub bytes: Vec<u8>,
}

/// Route each communicator's records in its signed membership order.
pub fn connection_records(
    catalog: &BTreeMap<String, Declaration>, records: &[Vec<Connection>],
) -> Result<BTreeMap<String, Vec<Vec<u8>>>> {
    let world = catalog.values().flat_map(|d| d.members.iter()).max().map_or(0, |r| r + 1);
    if records.len() != world {
        return Err(invalid("connection exchange does not cover the full cluster"));
    }
    for (rank, row) in records.iter().enumerate() {
        let mut ids = BTreeSet::new();
        for r in row {
            let d = catalog.get(&r.id).ok_or_else(|| invalid("unknown communicator connection"))?;
            if !ids.insert(&r.id) || d.members.get(r.local_rank) != Some(&rank) || r.bytes.is_empty() {
                return Err(invalid("invalid communicator connection identity or record"));
            }
        }
    }
    let mut routed = BTreeMap::new();
    for d in catalog.values() {
        let mut peers = Vec::new();
        for (local, &global) in d.members.iter().enumerate() {
            let r = records.get(global).and_then(|row| row.iter().find(|r| r.id == d.id))
                .ok_or_else(|| invalid(format!("communicator {} connection missing", d.id)))?;
            if r.local_rank != local {
                return Err(invalid("communicator connection order differs from membership"));
            }
            peers.push(r.bytes.clone());
        }
        routed.insert(d.id.clone(), peers);
    }
    Ok(routed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    // Frozen legacy fields, independent of the generated extended rank messages.
    #[derive(Clone, PartialEq, Message)]
    struct LegacyTransport {
        #[prost(uint32, tag="1")] proxy_abi: u32,
        #[prost(uint32, tag="2")] world: u32,
        #[prost(uint32, tag="3")] rank: u32,
        #[prost(uint64, tag="4")] region_address: u64,
        #[prost(uint64, tag="5")] region_bytes: u64,
        #[prost(uint64, tag="6")] counters_address: u64,
        #[prost(uint64, tag="7")] counters_bytes: u64,
        #[prost(uint64, tag="8")] slot_bytes: u64,
        #[prost(uint64, tag="9")] pipe_bytes: u64,
        #[prost(uint64, tag="10")] ctrl_offset: u64,
        #[prost(uint64, tag="16")] varlen_bytes: u64,
    }
    #[derive(Clone, PartialEq, Message)]
    struct LegacyCluster {
        #[prost(uint32, tag="1")] world: u32,
        #[prost(string, tag="3")] export_id: String,
    }
    #[derive(Clone, PartialEq, Message)]
    struct LegacyBundle {
        #[prost(message, optional, tag="11")] roce: Option<LegacyTransport>,
    }
    #[derive(Clone, PartialEq, Message)]
    struct LegacyIndex {
        #[prost(message, repeated, tag="2")] ranks: Vec<LegacyBundle>,
        #[prost(message, optional, tag="9")] cluster: Option<LegacyCluster>,
    }
    #[derive(Clone, PartialEq, Message)]
    struct LegacyRankBody {
        #[prost(uint32, tag="1")] rank: u32,
        #[prost(string, tag="2")] deployment_id: String,
        #[prost(message, optional, tag="15")] roce: Option<LegacyTransport>,
    }
    #[derive(Clone, PartialEq, Message)]
    struct LegacyDeployment {
        #[prost(string, tag="1")] deployment_id: String,
        #[prost(message, optional, tag="5")] cluster: Option<LegacyCluster>,
    }

    fn cluster(rank: usize) -> ClusterRank {
        ClusterRank { world:16,rank,export_id:"transport-codec-test".into() }
    }
    fn transport(world: usize, rank: usize, base: u64) -> wire::RoceTransport {
        wire::RoceTransport { proxy_abi:14,world:world as u32,rank:rank as u32,
            region_address:base,region_bytes:8192,counters_address:base+0x4000,
            counters_bytes:4096,slot_bytes:16,pipe_bytes:16,ctrl_offset:4096,varlen_bytes:0 }
    }
    fn comm(id: &str, members: Vec<u32>, rank: usize, base: u64) -> wire::RoceCommunicator {
        let local = members.iter().position(|&r| r as usize == rank).unwrap();
        wire::RoceCommunicator { id:id.into(), transport:Some(transport(members.len(),local,base)),members }
    }

    #[test]
    fn legacy_and_repeated_wire_roundtrip_preserve_old_serving_rejection() {
        use crate::core::schema::wire as core;
        let cluster_wire=wire::Cluster { world:16,export_id:"transport-codec-test".into() };
        for repeated in [false,true] {
            let t=transport(16,0,0x10000);
            let bundle=wire::BundleIndex {
                roce:(!repeated).then_some(t.clone()),
                roce_communicators:if repeated { vec![comm("tp",(0..16).collect(),0,0x10000)] } else { vec![] },
                ..Default::default()
            };
            let index=wire::ModelIndex { ranks:vec![bundle.clone()],cluster:Some(cluster_wire.clone()),..Default::default() };
            let bytes=index.encode_to_vec();
            let old=LegacyIndex::decode(bytes.as_slice()).unwrap();
            assert_eq!(old.cluster.as_ref().unwrap().world,16);
            assert_eq!(old.cluster.as_ref().unwrap().export_id,"transport-codec-test");
            assert_eq!(require_cluster_transport(old.ranks[0].roce.as_ref()).is_ok(),!repeated);
            if !repeated {
                assert_eq!(old.ranks[0].roce.as_ref().unwrap().encode_to_vec(),t.encode_to_vec());
            }
            let current=wire::ModelIndex::decode(bytes.as_slice()).unwrap();
            let rank=&current.ranks[0];
            let normalized=normalize(rank.roce.clone(),rank.roce_communicators.clone(),Some(&cluster(0))).unwrap();
            assert_eq!(normalized[0].transport.as_ref().unwrap(),&t);
            assert_eq!(normalized[0].members,(0..16).collect::<Vec<_>>());

            let deployment=core::DeploymentBody { deployment_id:"dep".into(),cluster:Some(cluster_wire.clone()),..Default::default() };
            let rank=core::RankBody { rank:0,deployment_id:"dep".into(),roce:bundle.roce,
                roce_communicators:bundle.roce_communicators,..Default::default() };
            let old_deployment=LegacyDeployment::decode(deployment.encode_to_vec().as_slice()).unwrap();
            let old_rank=LegacyRankBody::decode(rank.encode_to_vec().as_slice()).unwrap();
            assert_eq!(old_deployment.cluster.unwrap().world,16);
            assert_eq!(old_rank.deployment_id,old_deployment.deployment_id);
            assert_eq!(require_cluster_transport(old_rank.roce.as_ref()).is_ok(),!repeated);
            let current=core::RankBody::decode(rank.encode_to_vec().as_slice()).unwrap();
            assert_eq!(normalize(current.roce,current.roce_communicators,Some(&cluster(0))).unwrap()[0]
                .transport.as_ref().unwrap(),&t);
        }
    }

    #[test]
    fn normalization_rejects_ambiguity_membership_and_protocol_aliasing() {
        let tp=comm("tp",(0..16).collect(),0,0x10000);
        let subgroup=comm("dcp-a",(0..8).collect(),0,0x20000);
        let valid=normalize(None,vec![tp.clone(),subgroup.clone()],Some(&cluster(0))).unwrap();
        assert_eq!(valid.len(),2);
        assert!(normalize(tp.transport.clone(),vec![subgroup.clone()],Some(&cluster(0))).is_err());
        assert!(normalize(None,vec![tp.clone(),tp.clone()],Some(&cluster(0))).is_err());
        for members in [vec![0,0],vec![0,16],vec![1,0]] {
            let mut bad=comm("bad",(0..2).collect(),0,0x30000);
            bad.members=members;
            assert!(normalize(None,vec![bad],Some(&cluster(0))).is_err());
        }
        let mut bad=subgroup.clone();
        bad.transport.as_mut().unwrap().world=16;
        assert!(normalize(None,vec![bad],Some(&cluster(0))).is_err());
        let mut alias=tp.clone();alias.id="independent".into();
        assert!(normalize(None,vec![tp.clone(),alias.clone()],Some(&cluster(0))).is_err());
        alias.transport=Some(transport(16,0,0x30000));
        assert_eq!(normalize(None,vec![tp,alias],Some(&cluster(0))).unwrap().len(),2);
        assert!(normalize(None,vec![subgroup],None).is_err());
    }

    #[test]
    fn owned_ranges_reject_wrong_memory_bounds_and_control_page() {
        let c=comm("tp",(0..16).collect(),0,0x10000);
        let allocations=vec![
            wire::Allocation { address:Some(0x10000),nbytes:8192,memory:wire::Memory::MappedHost as i32,..Default::default() },
            wire::Allocation { address:Some(0x14000),nbytes:4096,memory:wire::Memory::Device as i32,..Default::default() },
        ];
        validate_owned_ranges(&[c.clone()],&allocations).unwrap();
        let mut small=allocations.clone();small[1].nbytes-=1;
        assert!(validate_owned_ranges(&[c.clone()],&small).is_err());
        let mut wrong=allocations.clone();wrong[0].memory=wire::Memory::Peer as i32;
        assert!(validate_owned_ranges(&[c.clone()],&wrong).is_err());
        let mut bad=c;bad.transport.as_mut().unwrap().ctrl_offset=8192;
        assert!(validate_owned_ranges(&[bad],&allocations).is_err());
    }

    fn declarations16() -> Vec<Vec<Declaration>> {
        (0..16).map(|rank| {
            let group=(rank/8)*8;
            let normalized=normalize(None,vec![
                comm("tp",(0..16).collect(),rank,0x10000),
                comm(if group==0 {"dcp-a"} else {"dcp-b"},(group as u32..group as u32+8).collect(),rank,0x20000),
            ],Some(&cluster(rank))).unwrap();
            normalized.into_iter().map(|c| {
                let t=c.transport.unwrap();
                Declaration { id:c.id,members:c.members.into_iter().map(|r|r as usize).collect(),local_rank:t.rank as usize,
                    geometry:Geometry {proxy_abi:t.proxy_abi,world:t.world as usize,region_bytes:t.region_bytes,
                        counters_bytes:t.counters_bytes,slot_bytes:t.slot_bytes,pipe_bytes:t.pipe_bytes,
                        varlen_bytes:t.varlen_bytes,ctrl_offset:t.ctrl_offset} }
            }).collect()
        }).collect()
    }

    #[test]
    fn serialized_bootstrap_routes_tp16_and_two_subgroups_without_collapsing_ids() {
        let rows=declarations16();
        let serialized=rows.iter().map(|r|serde_json::to_vec(r).unwrap()).collect::<Vec<_>>();
        let decoded=serialized.iter().map(|r|serde_json::from_slice(r).unwrap()).collect::<Vec<Vec<Declaration>>>();
        let agreed=agree(&decoded).unwrap();
        assert_eq!(agreed.keys().cloned().collect::<Vec<_>>(),["dcp-a","dcp-b","tp"]);
        let records=decoded.iter().enumerate().map(|(rank,row)|row.iter().map(|d|Connection {
            id:d.id.clone(),local_rank:d.local_rank,bytes:vec![rank as u8,d.geometry.world as u8],
        }).collect()).collect::<Vec<Vec<_>>>();
        let bytes=serde_json::to_vec(&records).unwrap();
        let routed=connection_records(&agreed,&serde_json::from_slice::<Vec<Vec<Connection>>>(&bytes).unwrap()).unwrap();
        assert_eq!(routed["tp"].len(),16);
        assert_eq!(routed["dcp-a"],(0..8).map(|r|vec![r,8]).collect::<Vec<_>>());
        assert_eq!(routed["dcp-b"],(8..16).map(|r|vec![r,8]).collect::<Vec<_>>());
        let mut missing=records.clone();missing[7].retain(|r|r.id!="dcp-a");
        assert!(connection_records(&agreed,&missing).is_err());
        let mut duplicate=records.clone();duplicate[0].push(records[0][0].clone());
        assert!(connection_records(&agreed,&duplicate).is_err());
        assert!(connection_records(&agreed,&records[..15]).is_err());
    }

    #[test]
    fn bootstrap_rejects_missing_members_and_geometry_disagreement() {
        let rows=declarations16();
        let mut missing=rows.clone();missing[3].retain(|d|d.id!="dcp-a");
        assert!(agree(&missing).is_err());
        let mut wrong=rows.clone();wrong[8][0].geometry.slot_bytes+=16;
        assert!(agree(&wrong).is_err());
        let mut reversed=rows.clone();reversed[0][0].members.reverse();
        assert!(agree(&reversed).is_err());
        let mut duplicate=rows.clone();duplicate[0].push(rows[0][0].clone());
        assert!(agree(&duplicate).is_err());
    }
}
