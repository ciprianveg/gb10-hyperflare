"""CPU arithmetic for the fixed TP16/DCP8 qualification schedule.

Variable-count references include both logical prefixes and the documented
PreparedVarlenGather row layout. The future producer must bind the prescribed
zero-initialization/local-prefix copy before these can qualify GPU results.
"""
from dataclasses import dataclass
import struct

WORLD = tuple(range(16))
GROUPS = (tuple(range(8)), tuple(range(8, 16)))
HOSTS = tuple(f"192.168.177.{host}" for host in (*range(111, 119), *range(11, 19)))
CHUNK_BYTES = 256 * 1024
# Zero is an inactive contribution; the tail cases straddle one complete chunk.
VALID_BYTES = (0, 16, CHUNK_BYTES - 16, CHUNK_BYTES, CHUNK_BYTES + 16, 32, 48, 64)


@dataclass(frozen=True)
class Operation:
    index: int
    communicator: str
    kind: str
    rows: int
    columns: int


SCHEDULE = (
    Operation(0, "tp", "reduce", 17, 256),
    Operation(1, "dcp", "gather", 3, 256),
    Operation(2, "dcp", "reduce", 17, 256),
    Operation(3, "dcp", "reduce", 17, 256),
    Operation(4, "tp", "gather", 3, 256),
    Operation(5, "dcp", "selection-history", 1, (CHUNK_BYTES + 16) // 2),
    Operation(6, "tp", "reduce", 17, 256),
)


def members(operation, global_rank):
    if type(global_rank) is not int or not 0 <= global_rank < 16:
        raise ValueError("global rank must be in 0..15")
    return WORLD if operation.communicator == "tp" else GROUPS[global_rank // 8]


def bf16(value):
    bits, = struct.unpack("<I", struct.pack("<f", value))
    return struct.pack("<H", (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)


def source(operation, rank):
    """Exactly representable BF16 inputs distinguish global rank and operation."""
    if not 0 <= rank < 16:
        raise ValueError("invalid source rank")
    # All sums are small exact integers, divisible by 4 or 8 above BF16's unit band.
    return b"".join(bf16(rank + 1 + operation.index * 2 + i % 8)
                    for i in range(operation.rows * operation.columns))


def reduce_reference(operation, global_rank):
    group = members(operation, global_rank)
    return b"".join(bf16(sum(rank + 1 + operation.index * 2 + i % 8 for rank in group))
                    for i in range(operation.rows * operation.columns))


def gather_reference(operation, global_rank):
    """Column gather: concatenate members' columns independently for each row."""
    peers = [source(operation, rank) for rank in members(operation, global_rank)]
    stride = operation.columns * 2
    return b"".join(peer[row*stride:(row+1)*stride]
                    for row in range(operation.rows) for peer in peers)


def history_reference(operation, global_rank):
    """Member-ordered valid prefixes; inactive contribution is empty."""
    group = members(operation, global_rank)
    return tuple(source(operation, rank)[:count]
                 for rank, count in zip(group, VALID_BYTES, strict=True))


def history_output_reference(operation, global_rank):
    """PreparedVarlenGather [world, capacity] layout after zeroing every row.

    The mixed graph must zero output then copy only the local valid prefix into
    its row before gathering. Remote tails and inactive rows remain zero.
    """
    capacity = operation.rows * operation.columns * 2
    return b"".join(prefix + bytes(capacity - len(prefix))
                    for prefix in history_reference(operation, global_rank))


def logical_accounting(operation, global_rank):
    """Payload/chunk accounting, deliberately separate from proxy WR accounting.

    Proxy WR totals also include flags/credits and depend on rails and the
    selected collective family. They must be derived from the installed
    implementation, not inferred from these logical payload counts.
    """
    group = members(operation, global_rank)
    local = group.index(global_rank)
    counts = VALID_BYTES if operation.kind == "selection-history" else (
        operation.rows * operation.columns * 2,
    ) * len(group)
    return {
        "world": len(group),
        "local_rank": local,
        "local_payload_bytes": counts[local],
        "all_member_payload_bytes": sum(counts),
        "local_payload_chunks": (counts[local] + CHUNK_BYTES - 1) // CHUNK_BYTES,
        "all_member_payload_chunks": sum((size + CHUNK_BYTES - 1) // CHUNK_BYTES for size in counts),
    }


def protocol_accounting(operation, global_rank, rails=2):
    """Direct-mesh post_striped/run_varlen_op accounting in qualified C proxy.

    Each stripe has one signaled 4-byte flag WR and, only when nonempty, one
    unsignaled data WR. Native writes_completed counts flag completions, not all
    WRs; native kind_stats.bytes counts local dispatch bytes, not replicated
    payload bytes or wire flag bytes. Empty varlen sends one flags-only chunk.
    """
    if rails != 2:
        raise ValueError("fixture requires the qualified two-rail direct mesh")
    info = logical_accounting(operation, global_rank)
    size, peers = info["local_payload_bytes"], info["world"] - 1
    chunks = ([min(CHUNK_BYTES, size-offset) for offset in range(0,size,CHUNK_BYTES)] or [0]
              if operation.kind == "selection-history" else [size])
    flags = len(chunks)*rails*peers
    data_wrs = sum(min(chunk//16,rails) for chunk in chunks)*peers
    return dict(
        ops_posted=1,last_seq_increment=1,
        dispatch_bytes=size,kind_slot=5 if operation.kind == "selection-history" else 0,
        writes_completed=flags,payload_bytes_posted=size*peers,
        derived_data_wrs=data_wrs,derived_flag_wrs=flags,
        derived_total_wrs=data_wrs+flags,derived_flag_bytes=flags*4,
        protocol_chunks=len(chunks),
    )


def communicator_accounting(global_rank):
    result = {}
    for name in ("tp","dcp"):
        operations = [protocol_accounting(op,global_rank) for op in SCHEDULE if op.communicator == name]
        result[name] = {key:sum(op[key] for op in operations) for key in operations[0] if key != "kind_slot"}
        result[name]["kind_stats"] = {
            str(slot):dict(ops=sum(op["ops_posted"] for op in operations if op["kind_slot"] == slot),
                           bytes=sum(op["dispatch_bytes"] for op in operations if op["kind_slot"] == slot))
            for slot in (0,5)
        }
    return result
