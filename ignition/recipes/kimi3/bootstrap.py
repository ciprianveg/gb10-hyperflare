# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Kimi K3 text and image graphs with tensor-parallel decoder storage."""

import contextlib
from dataclasses import dataclass, fields as dataclass_fields, replace
import hashlib
import json
import os
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.capacity import (
    batch_capacities, prefill_capacities, request_capacity,
    speculative_token_capacity, serving_modes,
)
from ignition.components.parallel import TensorParallelShard
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
                                              create_all_reduce_runtime, create_head_exchange,
                                              create_roce_runtime)
from .routing import create_route_runtime
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import GraphBufferPool, ScratchArena, scratch_views
from ignition.preparation.selection import BenchmarkSelection
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import DSparkConfig, ModelConfig, PAGE_TOKENS
from .dcp import LSE_SLICE_ROWS
from ignition.recipes.kimi31.dflash_config import DFlash2Config
from ignition.recipes.kimi31.dspark_config import DSparkConfig as MLADSparkConfig
from .contract import prepare as prepare_contract
from .decoder import DecoderStorage, PreparedDecoder, Scratch
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights
from .vision import PATCH_CAPACITIES, FEATURES, HIDDEN, PreparedVision, VisionInputs, VisionScratch
from .vision_state import PreparedVisionGeneration
from ignition.components.vision_state import PrefillVisionInputs, VisionState


# Most rows of one all-reduce; larger prefill reductions run in chunks of this many rows.
REDUCE_ROWS = 1024

# Kimi K3's tensor-parallel path grew up CUDA-IPC native: the exchange classes below
# select experts, candidates, and attention heads inside PCIe DcpAllToAll channels,
# read peer continuation pages through IPC pointers, and stage prefill records in
# shared pages. Ranks on separate hosts can carry none of them, so multi-host
# preparation is gated PER CLASS: each port stage flips its class's entry in
# REMOTE_EXCHANGES_PORTED and the gate message shrinks with it, until stage 6 removes
# the gate. The RoCE runtime that glm53, deepseek41, minimax_m3, and qwen38 swap in
# under multi-host covers all-reduce and all-gather, not these exchanges.
HEAD_EXCHANGE_CLASS = "the latent, draft-context, convolution, prefill-route, decode-head, and LSE head exchanges"
REMOTE_EXCHANGE_STAGES = {
    "the decode route top-16 exchange": 3,
    "the DSA candidate and owner exchanges": 4,
    HEAD_EXCHANGE_CLASS: 2,
    "shared prefill staging": 5,
    "IPC continuation-page peers": 5,
}
# Stage 2 ported the head-exchange classes: latent, draft-context, convolution, and
# prefill-route gathers ride the shared RoCE backbone, decode/verify gathers carry
# E4M3 queries through a byte-exact BF16 view, and LSE combines run sliced over
# caller-owned staging with the auxiliary-stream query overlap disabled.
# Stage 3 ported the decode route top-16 exchange: both route shards pack into one
# BF16 send row over the backbone and the fused selection runs on the main stream
# afterwards with exact semantics, keeping nine-to-sixteen-row verify batches on the
# route path instead of the general path (voipmonitor #15's algorithm).
# Stage 4 ported the DSA candidate exchange: the replicated candidate and selection
# gathers ride the backbone unchanged (their FP32 score/ID pairs are already BF16-view
# byte-exact), and the PCIe owner exchange and its tuning composition are never
# constructed under multi-host — the merged alternative stays the only one.
# Stage 5 ported the two prefill classes onto the backbone's variable-length gathers:
# the windowed index history (query-sharded scoring) and the selection-driven selected
# latent records (each rank packs what its own pool holds of the bit-identical global
# top-2048), both self-synchronizing: no shared pages, staged slabs, or BF16 signals
# exist under multi-host, and no peer page is ever read.
REMOTE_EXCHANGES_PORTED = {name: True for name in REMOTE_EXCHANGE_STAGES}


def _unported_exchange_classes():
    """Every exchange class still tied to single-host CUDA IPC, as (stage, name) pairs."""
    return sorted((stage, name) for name, ported in REMOTE_EXCHANGES_PORTED.items()
                  if not ported for stage in (REMOTE_EXCHANGE_STAGES[name],))


def _require_ported_exchanges(coordinator):
    """Fail fast, per exchange class, before any cross-host CUDA-IPC object is constructed."""
    if coordinator is None or not getattr(coordinator, "spans_hosts", False):
        return
    remaining = _unported_exchange_classes()
    if remaining:
        raise NotImplementedError(
            "multi-host Kimi K3 preparation still requires the PCIe-IPC-only exchanges: "
            + "; ".join(f"{name} (port stage {stage})" for stage, name in remaining)
            + "; port them to the RoCE transport (see glm53's remote context parallelism)"
              " before preparing Kimi K3 across hosts")


def _shared_column_peers(group, core_storage, hosts):
    """CUDA-IPC column staging works only between ranks on one host.

    SharedColumnReads exchanges CUDA-IPC handles across the whole group, which fails
    opaquely across hosts. Under multi-host preparation every host reads its own
    column slices through the snapshot source instead.
    """
    if (group is None or core_storage is None or core_storage.peer_storage is None
            or (hosts is not None and len(set(hosts)) > 1)):
        return None
    return (group, core_storage.peer_storage)


# Stage 2 slices LSE-combine rows to at most this many rows per exchange (dcp.py).
# Stage 5's variable-length exchanges over the backbone: one 128-row indexer part's
# selected latent records (576 bytes each, worst case every selection owned by one
# rank) and one window of index-history positions (132 bytes each) per rank.
REMOTE_RECORD_ROWS, INDEX_WINDOW_POSITIONS = 128, 65536
# Option W's fixed window capacity is LATENT_WINDOW_RECORDS in kimi31.dcp.
# Resolve its transport bound before choosing any replica or attention consumer.


def _latent_exchange(collective, spans_hosts, group, device, *, world_size, latent, channel_rows,
                     max_tokens, ipc):
    """Latent gather runtimes with their declared geometry (stage 2).

    Under multi-host preparation the shared RoCEnante backbone carries every latent
    gather, and callers read explicit rows and the padded latent width instead of a
    PCIe channel's declared capacity. Single-host ranks keep the two b12x head
    exchanges: one sized for up to eight decode rows, one for prefill chunks.
    """
    padding = -(-latent // 8) * 8
    if spans_hosts:
        return (collective,), channel_rows, padding
    runtimes = tuple(create_head_exchange(group, device, max_rows=rows, heads=world_size,
                                         head_dim=padding, ipc=ipc)
                     for rows in sorted({min(8, max_tokens), channel_rows}))
    return runtimes, None, None


@dataclass(frozen=True)
class PrefillRecordExchange:
    mode: str
    varlen_bytes: int
    window_records: int
    attempt_replica: bool


def prefill_record_exchange(mode):
    """Resolve the existing record path and its per-rank staging capacity together."""
    from ignition.recipes.kimi31.dcp import LATENT_WINDOW_RECORDS, RECORD_BYTES, TOPK

    if mode not in ("auto", "selected", "windowed"):
        raise ValueError("prefill record exchange must be auto, selected or windowed")
    index_bytes = ((INDEX_WINDOW_POSITIONS * 132 + 15) // 16) * 16
    window = LATENT_WINDOW_RECORDS if mode == "windowed" else 0
    latent_bytes = (window if window else REMOTE_RECORD_ROWS * TOPK) * RECORD_BYTES
    return PrefillRecordExchange(mode, max(latent_bytes, index_bytes), window, mode == "auto")


def remote_record_context(context, config, exchange, model_config, pool, *, rank, shard, device, runtime):
    """Apply one resolved exchange before preparing any DSA attention consumer."""
    from ignition.recipes.kimi31.dcp import (PreparedRecordReplica, replica_capacity,
                                            RECORD_BYTES, ContextShards)
    from ignition.recipes.kimi31.indexer import SharedBuffers
    from ignition.preparation.host_memory import device_memory_info

    replica = None
    if exchange.attempt_replica and os.environ.get("FLARE_RECORDS_REPLICA", "1") != "0":
        try:
            layers = sum(kind == "dsa" for kind in model_config.layers)
            assert layers > 0, "the record replica requires at least one DSA layer"
            free = device_memory_info(device)[0]
            capacity = replica_capacity(free, layers, config.context,
                -(-config.prefill_chunk_size // shard.world_size), headroom_bytes=config.headroom_bytes)
            # Partial replicas are invalid for full-context requests. Auto retains
            # the existing conservative transport and selected-record fallback.
            if capacity < config.context:
                print(f"[rank {rank}] record replica disabled: capacity {capacity} < context "
                      f"{config.context} ({layers} layers x 708 B/token; free {free >> 30} GiB) "
                      "— stage-5 exchange applies", flush=True)
            else:
                print(f"[rank {rank}] record replica capacity {capacity} tokens "
                      f"({capacity * layers * (RECORD_BYTES + 132) >> 30} GiB of "
                      f"{free >> 30} GiB free)", flush=True)
                replica = PreparedRecordReplica(layers=layers, capacity=capacity,
                    chunk_records=-(-config.prefill_chunk_size // shard.world_size),
                    backing=pool.backing, shards=ContextShards.on(rank, shard.world_size, device),
                    storage=SharedBuffers({}, device),
                    fused_publish=os.environ.get("FLARE_RECORDS_FUSED_PUBLISH", "1") != "0")
        except (MemoryError, torch.cuda.OutOfMemoryError, ValueError):
            replica = None
    context = replace(context, replica=replica)
    if exchange.window_records:
        if replica is not None or exchange.varlen_bytes != runtime.max_varlen_bytes:
            raise ValueError("windowed exchange differs from its declared transport capacity")
        print(f"[rank {rank}] windowed latent exchange: {exchange.window_records} records/window "
              f"({exchange.window_records * RECORD_BYTES >> 20} MiB staging row per rank) "
              f"— {exchange.window_records * shard.world_size} tokens per window", flush=True)
        context = replace(context, window_records=exchange.window_records,
            staged_records=torch.zeros((config.context, RECORD_BYTES),
                                       dtype=torch.float8_e4m3fn, device=device))
    return context


def _remote_backbone_runtime(config, group, device, *, core_storage, model_config, shard,
                             channel_rows, sample_rows, slots, record_exchange=None,
                             communicator_id=None, members=None):
    """One RoCEnante runtime for every backbone reduction and gather of separate-host ranks.

    The runtime carries the exchanges the RoCE transport already mirrors — hidden
    all-reduces, the vocabulary and embedding column gathers, the router's expert
    columns — and, through `collective`, DSA prefill staging (stage_runtime) and the
    drafting context gathers. It does NOT port the gated exchange classes; their
    per-class entries stay False until stages 2-5 land.

    Decode and verification rows (`channel_rows`) reduce one-shot; prefill chunks run
    through the pipelined two-shot path (`pipe_tokens` rows of hidden columns:
    4096 x 7168 x 2 = 56 MiB). The one-shot window also plans the speculative round's
    physical rows in from construction (`sample_rows`, glm53's planned sample
    capacity): the dflash2 draft block runs every configured position per request, so
    its embedding, per-layer and head exchanges move `batch * (draft tokens + 1)` rows
    whether the round verifies two rows or eight. The gather channel covers the
    largest per-rank contribution the port stages move onto this runtime — the LSE
    combine's 128-row slices (stage 2), vocabulary partitions, DSA candidates and
    selections (stage 4) — sized upward in 16 MiB steps. Variable-length gathers
    carry stage 5's windowed history exchanges.
    """
    from ignition.components.head_exchange import pack_width
    record_exchange = (prefill_record_exchange(config.prefill_record_exchange)
                       if record_exchange is None else record_exchange)
    world = shard.world_size
    heads = model_config.text["num_attention_heads"]
    candidate_rows = max(min(config.prefill_chunk_size, LSE_SLICE_ROWS), config.decode_tokens)
    gather_bytes = max(
        LSE_SLICE_ROWS * pack_width(world, heads, 512) * 2,
        sample_rows * (model_config.vocabulary // world) * 2,
        candidate_rows * 8192 * 2,
        -(-config.prefill_chunk_size // world) * 4096 * 2,
    )
    if model_config.draft is not None:
        # The markov draft's transition bias gathers every vocabulary column as FP32
        # logits carried in BF16 pairs (one all-gather per round, up to `slots` rows),
        # and the DSpark draft's LSE combine packs `draft.heads` head shards per slice,
        # which can exceed the target attention heads the first term sizes.
        draft_heads = getattr(model_config.draft, "heads", None)
        gather_bytes = max(gather_bytes, slots * (model_config.vocabulary // world) * 4)
        if draft_heads is not None:
            gather_bytes = max(gather_bytes, LSE_SLICE_ROWS * pack_width(world, draft_heads, 512) * 2)
    options = dict(max_tokens=max(channel_rows, sample_rows),
        pipe_tokens=max(config.prefill_chunk_size, config.decode_tokens),
        hidden_size=model_config.hidden,
        gather_bytes=-(-gather_bytes // (16 << 20)) * (16 << 20),
        varlen_bytes=record_exchange.varlen_bytes,
        ring=config.roce_ring, ring_open=config.roce_chain, core_storage=core_storage)
    if config.roce_pipe_chunk_bytes:
        # The preparation config's chunk override beats create_roce_runtime's environment read.
        options["pipe_chunk_bytes"] = config.roce_pipe_chunk_bytes
    return create_roce_runtime(group, device, communicator_id=communicator_id, members=members, **options)


def context_groups(world, size):
    """Ordered contiguous global memberships; every rank creates the same groups."""
    if type(size) is not int or size < 1 or world % size:
        raise ValueError("context group size must positively divide tensor-parallel world")
    return tuple(tuple(range(start, start + size)) for start in range(0, world, size))


def prepare_context_group(group, *, rank, world, size):
    """Create ordered Gloo subgroups before loading weights; return this rank's group."""
    memberships = context_groups(world, size)
    if not 0 <= rank < world:
        raise ValueError("preparation rank is outside tensor-parallel membership")
    if size == world:
        return group, rank, memberships[0]
    if tuple(torch.distributed.get_process_group_ranks(group)) != tuple(range(world)):
        raise ValueError("DCP subgroups require ordered full-world TP membership")
    if torch.distributed.get_rank() != rank:
        raise ValueError("preparation rank does not match distributed global rank")
    selected = None
    for members in memberships:
        subgroup = torch.distributed.new_group(ranks=list(members), backend="gloo")
        if rank in members:
            selected = subgroup, members.index(rank), members
    assert selected is not None
    return selected


def allocate_serial_workspace(buffers, *, text_rows, hidden_size, residual_count,
                              auxiliary_count, vision_patches):
    """Share text and vision backing while keeping each forward's live values disjoint.

    First-fit allocation must see the largest slabs first. A smaller residual
    reservation can otherwise force another vision QKV slab even though the
    two forward families execute serially.
    """
    text_shapes = ((text_rows, hidden_size),) * 4
    if auxiliary_count:
        text_shapes += ((text_rows, auxiliary_count * hidden_size),)
    text_elements = sum(rows * columns for rows, columns in text_shapes)
    residual_shape = (text_rows, residual_count, hidden_size)
    residual_elements = text_rows * residual_count * hidden_size
    vision_first = (vision_patches > 0 and
                    VisionScratch.workspace_elements(vision_patches) >= max(text_elements, residual_elements))

    def reserve_vision():
        VisionScratch.allocate(vision_patches, buffers.device, buffers=buffers.cursor())

    if vision_first:
        reserve_vision()
    cursor = buffers.cursor()
    if text_elements >= residual_elements:
        shared = cursor.zeros((text_elements,), torch.bfloat16)
        residual = cursor.zeros(residual_shape, torch.bfloat16)
    else:
        residual = cursor.zeros(residual_shape, torch.bfloat16)
        shared = cursor.zeros((text_elements,), torch.bfloat16)
    if vision_patches and not vision_first:
        reserve_vision()
    return cursor, residual, shared, text_shapes


def allocate_prefill_histories(tokens, device):
    """Reuse index-history bytes after selection for the latent-history copy.

    Attention must complete index scoring and selected-ID exchange before it
    overwrites this backing with latent records. Neither history survives the
    layer; the paged KV and index caches retain independent persistent storage.
    """
    from .dcp import RECORD_BYTES

    records = torch.zeros((tokens, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
    indices = records.view(torch.uint8).view(-1)[:tokens * 132].view(tokens // 64, 64 * 132)
    return records, indices


def target_tuning_identity(config, headers, world_size, *, online_mxfp8_scope="all",
                           prefill_record_exchange="auto", roce_pipe_chunk_bytes=None, dcp_size=None):
    """Scope kernel selections to target geometry, independently of attached drafts.

    The target-only descriptor retains its empty draft field. Each operator's
    encoded query, invocation and candidate programs distinguish additional
    draft shapes and arithmetic contracts inside the selection cache.
    """
    value = {
        "config": config.text, "headers": headers, "vision": config.vision,
        "draft": None, "world_size": world_size,
        "contract": "kimi3.graph-inputs.v1"}
    if online_mxfp8_scope != "all":
        value["online_mxfp8_scope"] = online_mxfp8_scope
    if prefill_record_exchange != "auto":
        value["prefill_record_exchange"] = prefill_record_exchange
    if roce_pipe_chunk_bytes is not None:
        value["roce_pipe_chunk_bytes"] = roce_pipe_chunk_bytes
    if dcp_size is not None and dcp_size != world_size:
        value["dcp_size"] = dcp_size
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    image_concurrency: int | None = None
    compile_workers: int | None = None
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = tuple(range(16))
    # Each rank's host when ranks run on separate hosts; devices are then host-local ordinals.
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    roce_chain: bool = False
    # Pipelined two-shot chunk size in bytes (multiple of 16); None keeps b12x's 128 KiB chunks.
    # The environment variable FLARE_ROCE_PIPE_CHUNK_BYTES is the launch-time override.
    roce_pipe_chunk_bytes: int | None = None
    prefill_record_exchange: str = "auto"
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    # Lower bound on the KV page count the plan requires; carried for the native serving bootstrap.
    minimum_pages: int | None = None
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    benchmark_selection: BenchmarkSelection | None = None
    # Learned draft checkpoint selected by the recipe's speculation method.
    draft_checkpoint: Path | None = None
    draft_head_quantization: str | None = None
    draft_body_quantization: str | None = None
    # Verification widths count the pending input and up to seven drafts.
    speculative_widths: tuple[int, ...] = tuple(range(1, 9))
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    speculative_graph_budget: int = 64
    # Store eligible attention, shared-expert, latent MoE, and DSpark projections as MXFP8.
    online_mxfp8: bool = False
    online_mxfp8_scope: str = "all"
    # Shared experts overlap routing and routed experts on an auxiliary stream.
    stream_overlap: bool = True
    # Ranks share each page's context (decode context parallel over every rank, KV interleave 1).
    context_parallel: bool = True
    dcp_size: int | None = None
    # The MoonViT encoder and its graphs prepare beside the text model unless the plan serves text only.
    vision: bool = True
    # Recipe-owned serving metadata travels with each rank's preparation configuration.
    sampling_defaults: dict | None = None
    frontend_template: str | None = None
    mtp_only: bool = False

    def __post_init__(self):
        if self.dcp_size is not None:
            if (type(self.dcp_size) is not int or self.dcp_size not in (8, 16)
                    or len(self.devices) != 16 or not self.context_parallel
                    or self.hosts is None or len(self.hosts) != 16 or len(set(self.hosts)) != 16
                    or self.roce_ring or self.roce_chain):
                raise ValueError("explicit DCP size requires TP16 on sixteen distinct hosts in a mesh, size 8 or 16")
        if self.prefill_record_exchange not in ("auto", "selected", "windowed"):
            raise ValueError("prefill record exchange must be auto, selected or windowed")
        if self.prefill_record_exchange == "windowed" and self.prefill_chunk_size < 4096:
            raise ValueError("windowed prefill record exchange requires configured prefill chunk >= 4096")
        if self.prefill_record_exchange != "auto" and (
                len(self.devices) != 16 or not self.context_parallel
                or self.hosts is None or len(self.hosts) != 16 or len(set(self.hosts)) != 16
                or self.roce_ring or self.roce_chain):
            raise ValueError("explicit prefill record exchange requires TP16 DCP on sixteen distinct hosts in a mesh")
        if self.mtp_only and (not self.drafting
                or max(self.speculative_widths, default=1) <= 1 or self.speculative_fixed_width == 1):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.draft_head_quantization not in (None, "mxfp8", "nvfp4"):
            raise ValueError("Kimi draft head conversion must be mxfp8 or nvfp4")
        if self.draft_head_quantization is not None and not self.drafting:
            raise ValueError("Kimi draft head conversion requires draft weights")
        if self.draft_body_quantization not in (None, "mxfp8"):
            raise ValueError("Kimi draft body conversion must be omitted or mxfp8")
        if self.draft_body_quantization is not None and not self.drafting:
            raise ValueError("Kimi draft body conversion requires draft weights")
        if self.online_mxfp8_scope not in ("all", "kda-input"):
            raise ValueError("online MXFP8 scope must be all or kda-input")
        if self.online_mxfp8_scope != "all" and not self.online_mxfp8:
            raise ValueError("online MXFP8 scope requires online conversion")
        if self.speculative_widths and max(self.speculative_widths) > 8:
            raise ValueError("Kimi K3 draft model supports at most seven draft tokens")
        from ignition.components.capacity import image_capacity

        if self.vision:
            images = image_capacity(self.max_active, self.image_concurrency)
        else:
            if self.image_concurrency is not None and (type(self.image_concurrency) is not int
                                                       or self.image_concurrency != 0):
                raise ValueError("image concurrency must be zero or omitted when vision is disabled")
            images = 0
        object.__setattr__(self, "image_concurrency", images)
        from ignition.planning import host_shape_error

        error = host_shape_error(self.roce_ring, self.roce_chain, None if self.hosts is None else len(self.hosts))
        if error:
            raise ValueError(error)
        if self.roce_pipe_chunk_bytes is not None and (type(self.roce_pipe_chunk_bytes) is not int
                or self.roce_pipe_chunk_bytes <= 0 or self.roce_pipe_chunk_bytes % 16):
            raise ValueError("RoCE pipe chunk bytes must be a positive multiple of 16 bytes")
        # One host names distinct CUDA devices; ranks on separate hosts repeat each host's
        # ordinal, so uniqueness belongs to the (host, device) placement pairs.
        placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
        if (not isinstance(self.devices, tuple) or len(self.devices) not in (8, 16)
                or len(set(placements)) != len(self.devices)
                or self.hosts is not None and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
                or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("Kimi K3 preparation requires eight or sixteen distinct (host, device) placements")
        if self.minimum_pages is not None and (type(self.minimum_pages) is not int or self.minimum_pages <= 0):
            raise ValueError("minimum page capacity must be a positive integer")
        request_capacity(self.max_active)
        selection = self.benchmark_selection
        if selection is not None:
            if (not self.drafting and not selection.prefill_chunk) or self.prompt_logprobs or selection.speculation_study:
                raise ValueError("Kimi benchmark preparation requires complete unscored text graphs")
            if selection.max_batch > self.max_active or (not selection.prefill_chunk
                    and not set(selection.selected_verification_widths) <= set(self.speculative_widths)):
                raise ValueError("benchmark shapes exceed the configured request or width capacities")
            if selection.required_context > self.context:
                raise ValueError("benchmark workload and output allowance must fit the context")
            if selection.prefill_chunk and selection.prefill_chunk != self.prefill_chunk_size:
                raise ValueError("benchmark chunk must match the prepared prefill chunk")
            if (not selection.prefill_chunk and self.speculative_token_capacity is not None
                    and selection.max_batch * selection.width > self.speculative_token_capacity):
                raise ValueError("benchmark shapes exceed the speculative token capacity")
        if self.drafting:
            if not self.speculative_widths or not set(self.speculative_widths) <= set(range(1, 9)):
                raise ValueError("Kimi K3 drafts verify one through eight rows per request")
            if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
                raise ValueError("fixed verification width must belong to the captured widths")
            _ = self.speculative_round_shapes
        if not self.page_tokens <= self.context <= 1048576 or self.context % self.page_tokens:
            raise ValueError(f"Kimi K3 context must be {self.page_tokens}-aligned in [{self.page_tokens},1048576]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Kimi K3 prefill chunk must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def drafting(self):
        return self.draft_checkpoint is not None

    @property
    def page_tokens(self):
        """Global positions per KV page: PAGE_TOKENS local records on each rank that shares the context."""
        return PAGE_TOKENS * ((self.dcp_size or len(self.devices)) if self.context_parallel else 1)

    @property
    def preparation_pages(self):
        """Back tuning and selected benchmark contexts before sharing peer mappings."""
        pages = -(-self.prefill_chunk_size // self.page_tokens) + 2 * request_capacity(self.max_active)
        if self.benchmark_selection is not None:
            pages = max(pages, self.benchmark_selection.physical_pages(self.page_tokens))
        return pages

    @property
    def tuning_state_rows(self):
        """Null and live decode rows, or one prefill row and its checkpoint."""
        return max(3, 1 + request_capacity(self.max_active))

    @property
    def speculative_round_shapes(self):
        from ignition.components.capacity import parallel_speculative_shapes

        if self.benchmark_selection is not None:
            return self.benchmark_selection.speculative_round_shapes
        if not self.drafting:
            return ()
        if self.hosts is None:
            return parallel_speculative_shapes(self.max_active, self.speculative_widths,
                self.speculative_token_capacity, self.speculative_graph_budget,
                fixed_width=self.speculative_fixed_width)
        # Multi-host exports capture glm53's speculative width ladder: one packed round
        # per configured width at every batch whose verification envelope fits the token
        # capacity, so the serving core always finds a captured width for the requested
        # draft depth (helios "no captured speculative width fits the requested draft
        # depth"). Rows follow glm53's unpacked envelope — the round's packed token
        # capacity is batch * width within the token capacity, and width-one rounds stay
        # plain decode rounds. Single-host preparation keeps the packed single-width
        # catalog byte-identical.
        from ignition.components.capacity import batch_capacities, speculative_token_capacity

        capacity = speculative_token_capacity(self.max_active, max(self.speculative_widths, default=1),
                                              self.speculative_token_capacity)
        batches = batch_capacities(self.max_active)
        shapes = [(batch, width, batch * width) for width in self.speculative_widths
                  for batch in batches if width == 1 or batch * width <= capacity]
        if len(shapes) > self.speculative_graph_budget:
            raise ValueError("multi-host speculative width ladder requires a graph budget of at least "
                             f"{len(shapes)} (captured {len(self.speculative_widths)} widths across "
                             f"{len(batches)} batches)")
        return tuple(sorted(shapes, key=lambda shape: (shape[1], shape[0])))

    @property
    def decode_tokens(self):
        return max((packed for _, _, packed in self.speculative_round_shapes), default=request_capacity(self.max_active))

    @property
    def decode_samples(self):
        return max((batch * width for batch, width, _ in self.speculative_round_shapes), default=request_capacity(self.max_active))

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.drafting else 0

    @property
    def completion_token_capacity(self):
        return max((width for _, width, _ in self.speculative_round_shapes), default=1)


def _benchmark_state(buffers, pool, requests, speculation):
    state = {name: tensor for name, tensor in buffers.items() if name != "state_pool"}
    state["state_pool"] = pool.backing[:int(pool.memory_capacity[0].item())]
    pages = int(pool.memory_capacity[1].item())
    state.update((f"cache.{name}", tensor[:pages]) for name, tensor in pool.pages.items())
    state.update((f"request.{field.name}", getattr(requests, field.name)) for field in dataclass_fields(requests))
    if speculation is not None:
        state.update((f"speculation.rng.{field.name}", getattr(speculation.rng, field.name))
                     for field in dataclass_fields(speculation.rng))
        state.update({"speculation.generation": speculation.generation, "speculation.offset": speculation.offset})
    return state


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage):
    world = len(config.devices)
    dcp_world = (config.dcp_size or world) if config.context_parallel else 1
    split_context = dcp_world != world and config.context_parallel
    dcp_group, dcp_rank, dcp_members = prepare_context_group(
        group, rank=rank, world=world, size=dcp_world if config.context_parallel else world)
    snapshot = Path(snapshot).resolve()
    model_config = ModelConfig.load(snapshot)
    if config.dcp_size is not None and model_config.dsa is None:
        raise ValueError("explicit DCP size requires Kimi K3.1 DSA")
    record_exchange = prefill_record_exchange(config.prefill_record_exchange)
    if record_exchange.mode != "auto" and model_config.dsa is None:
        raise ValueError("explicit prefill record exchange requires the Kimi K3.1 DSA model")
    if model_config.dsa is not None:
        if not config.context_parallel:
            raise ValueError("Kimi K3.1 preparation requires context parallelism")
        if config.sampling_defaults is None or config.frontend_template is None:
            raise ValueError("Kimi K3.1 preparation requires recipe sampling defaults and a frontend template")
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [shard.header_sha256 for shard in manifest.shards]
    with core_storage.transient():
        peers = _shared_column_peers(group, core_storage, config.hosts)
        weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, online_mxfp8=config.online_mxfp8, online_mxfp8_scope=config.online_mxfp8_scope,
                                    draft_checkpoint=config.draft_checkpoint, peers=peers,
                                    vision=config.vision, weight_storage=core_storage.packed_weights(),
                                    draft_head_quantization=config.draft_head_quantization,
                                    draft_body_quantization=config.draft_body_quantization)
        del manifest
        release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    layout = continuation_layout(weights.config, shard.world_size, speculative_tokens=config.draft_tokens)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Kimi K3 continuation")
    slots, blocks = request_capacity(config.max_active), config.context // config.page_tokens
    drafting = weights.draft is not None
    max_tokens = max(config.prefill_chunk_size, config.decode_tokens)
    state_rows = slots + snapshots
    capture_state_rows = min(state_rows + 1, 1 + 2 * slots)
    pages = config.preparation_pages
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, shard.world_size, state_rows + 1, pages, device,
            speculative_tokens=config.draft_tokens, storage=core_storage,
            maximum_pages=total_memory // page_bytes(weights.config),
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(capture_state_rows, config.tuning_state_rows))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max_tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity), device=device)
        coordinator = CollectivePreparation(group)
        dcp_coordinator = CollectivePreparation(dcp_group) if split_context else coordinator
        if record_exchange.mode != "auto" and not coordinator.spans_hosts:
            raise ValueError("explicit prefill record exchange requires a multi-host RoCE backbone")
        _require_ported_exchanges(coordinator)
        # Prefill activations reduce in chunks of the rows decode and verification already need a channel for: a
        # channel sized for whole prefill chunks would stage 448 MiB per rank for 4096 rows.
        channel_rows = max(min(max_tokens, REDUCE_ROWS), config.decode_tokens)
        # glm53 plans the speculative gather capacity into the runtime instead of
        # paying added-on gathers. The drafted round's draft block runs every
        # configured position per request (block = draft tokens + 1), so the round's
        # physical row envelope is its largest drafted batch times the block — wider
        # than the packed verification rows the single-width catalog sized before.
        drafted = max((batch for batch, width, _ in config.speculative_round_shapes if width > 1), default=0)
        sample_rows = max(config.decode_samples, drafted * (config.draft_tokens + 1))
        if coordinator.spans_hosts:
            # One RoCEnante runtime carries the whole tensor-parallel backbone: hidden
            # reductions, vocabulary and embedding gathers, router columns, DSA prefill
            # staging (stage_runtime below), and the drafting context gathers. Exchange
            # classes beyond the backbone stay gated per class above until their port
            # stages land.
            collective = vocabulary_collective = embedding_collective = router_collective = (
                _remote_backbone_runtime(config, group, device, core_storage=core_storage,
                    model_config=weights.config, shard=shard, channel_rows=channel_rows,
                    sample_rows=sample_rows, slots=slots, record_exchange=record_exchange,
                    communicator_id="tp16" if split_context else None,
                    members=tuple(range(world)) if split_context else None))
        else:
            collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning, max_tokens=channel_rows,
                hidden_size=weights.config.hidden, ipc=core_storage.peer_storage)
            vocabulary_collective = create_all_gather_runtime(group, device, max_rows=sample_rows,
                columns=weights.config.vocabulary, ipc=core_storage.peer_storage)
            # Prefill rows beyond a channel's capacity gather in chunks, as they reduce.
            embedding_collective = create_all_gather_runtime(group, device, max_rows=channel_rows,
                columns=weights.config.hidden, ipc=core_storage.peer_storage)
            router_collective = create_all_gather_runtime(group, device, max_rows=channel_rows,
                columns=2 * shard.world_size * ((weights.config.text["num_experts"] // shard.world_size + 31) // 32 * 32),
                ipc=core_storage.peer_storage)
        context_collective = markov_collective = draft_context_collective = convolution_collective = None
        if drafting:
            # Embedding lookup and accepted-feature projection run serially,
            # gathering the same hidden-column partitions.
            context_collective = embedding_collective
            if isinstance(weights.config.draft, DFlash2Config):
                if coordinator.spans_hosts:
                    # Stage 2: the draft-context and convolution head exchanges are BF16
                    # gathers with config-declared widths; the shared RoCE backbone
                    # carries them without any PCIe channel attributes.
                    draft_context_collective = convolution_collective = collective
                else:
                    draft_context_collective = create_head_exchange(group, device, max_rows=channel_rows,
                        heads=shard.world_size, head_dim=(weights.config.draft.layers*576//shard.world_size+7)//8*8,
                        ipc=core_storage.peer_storage)
                    convolution_collective = create_head_exchange(group, device,
                        max_rows=slots*(config.draft_tokens+1), heads=shard.world_size,
                        head_dim=4*weights.config.hidden//16//shard.world_size, ipc=core_storage.peer_storage)
            elif isinstance(weights.config.draft, (DSparkConfig, MLADSparkConfig)):
                if coordinator.spans_hosts:
                    # The markov draft's FP32 transition bias gathers as BF16 column
                    # pairs over the shared RoCE backbone (_gather_transport_view keeps
                    # every FP32 bit), replacing the single-host channel's 2 *
                    # vocabulary BF16 columns; its per-round row count stays within
                    # `slots`, which the backbone's gather capacity covers.
                    markov_collective = collective
                else:
                    markov_collective = create_all_gather_runtime(group, device, max_rows=slots,
                        columns=2 * weights.config.vocabulary, ipc=core_storage.peer_storage)
            elif coordinator.spans_hosts:
                raise NotImplementedError("multi-host Kimi K3 preparation requires the DFlash2 or DSpark draft;"
                                          " other draft types' exchanges remain single-host")
            else:
                markov_collective = create_all_gather_runtime(group, device, max_rows=slots,
                    columns=2 * weights.config.vocabulary, ipc=core_storage.peer_storage)
        route_collective = route_remote = None
        if shard.world_size > 1:
            if coordinator.spans_hosts:
                # Stage 3: the decode route exchange rides the shared RoCE backbone; the PCIe
                # route channel is a single-host CUDA-IPC object and is never constructed.
                route_collective, route_remote = collective, True
            else:
                route_collective = create_route_runtime(group, device, ipc=core_storage.peer_storage)
        prefill_route_collective = None
        if shard.world_size > 1:
            prefill_route_collective = (collective if coordinator.spans_hosts else
                create_head_exchange(group, device, max_rows=channel_rows,
                    heads=shard.world_size,
                    head_dim=(weights.config.text["routed_expert_hidden_size"] + 2 * weights.config.text["num_experts"]) // shard.world_size,
                    ipc=core_storage.peer_storage))
        dcp_collective = collective
        if split_context:
            dcp_collective = _remote_backbone_runtime(config, dcp_group, device,
                core_storage=core_storage, model_config=weights.config,
                shard=TensorParallelShard(dcp_rank, dcp_world), channel_rows=channel_rows,
                sample_rows=sample_rows, slots=slots, record_exchange=record_exchange,
                communicator_id=f"dcp{dcp_world}-{dcp_members[0]}", members=dcp_members)
        context = None
        if config.context_parallel and shard.world_size > 1:
            from .dcp import RECORD_BYTES, ContextParallel, ContextShards

            extra = {}
            if weights.config.dsa is not None:
                from ignition.recipes.kimi31.dcp import AttentionConstants, ContextParallel

                extra["attention_constants"] = AttentionConstants.allocate(device, query_capacity=config.decode_tokens)
                # Stage 4: the replicated candidate and selection gathers ride the shared
                # backbone; the owner exchange (and its tuning composition below) stays a
                # single-host PCIe optimization and is never constructed under multi-host.
                extra["candidates"] = (dcp_collective if coordinator.spans_hosts else create_head_exchange(group, device,
                    max_rows=max(min(config.prefill_chunk_size, 128), config.decode_tokens),
                    heads=shard.world_size, head_dim=8192, ipc=core_storage.peer_storage))
                if (not coordinator.spans_hosts and drafting and config.compile_decoder
                        and config.autotune_effort != "off" and config.collective_tuning == "static"):
                    from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange

                    extra["candidate_owners"] = PCIeDCPTopKOwnerExchange.from_exchange_group(
                        exchange_group=group, device=device, max_rows=config.draft_tokens + 1,
                        topk=2048, stream_affine=False, ipc=core_storage.peer_storage)
            # Decode and verification gather every rank's absorbed E4M3 query heads and combine latent outputs.
            heads = weights.config.text["num_attention_heads"]
            head_rows = max(config.decode_tokens, slots * config.draft_tokens)
            heads_runtime = (dcp_collective if coordinator.spans_hosts
                             else create_head_exchange(group, device, max_rows=head_rows, heads=heads,
                                                       head_dim=512, query_head_dim=RECORD_BYTES,
                                                       ipc=core_storage.peer_storage))
            context = ContextParallel(ContextShards.on(dcp_rank, dcp_world, device),
                heads=heads_runtime,
                coordinator=dcp_coordinator,
                peers=({} if coordinator.spans_hosts else pool.share_pages(core_storage, group)), slab=None,
                remote=coordinator.spans_hosts,
                head_rows=(head_rows if coordinator.spans_hosts else None), **extra)
            if weights.config.dsa is not None and config.prefill_chunk_size >= 4096:
                if coordinator.spans_hosts:
                    selections = dcp_collective
                    index_records = torch.zeros((config.context // 64, 64 * 132), dtype=torch.uint8, device=device)
                    index_columns = torch.arange(config.context // 64, dtype=torch.int32, device=device)
                    context = replace(context, selections=selections, index_records=index_records,
                        index_columns=index_columns)
                    context = remote_record_context(context, config, record_exchange, weights.config,
                        pool, rank=dcp_rank, shard=TensorParallelShard(dcp_rank, dcp_world),
                        device=device, runtime=dcp_collective)
                else:
                    from ignition.components.shared_pages import SharedPages

                    staging = SharedPages(group, device, config.context // shard.world_size * RECORD_BYTES,
                                          ipc=core_storage.peer_storage)
                    index_staging = SharedPages(group, device, config.context // shard.world_size * 132,
                                                ipc=core_storage.peer_storage)
                    context = replace(context, staging=staging, stage_runtime=collective,
                        stage_signals=tuple(torch.zeros((1, 8), dtype=torch.bfloat16, device=device) for _ in range(4)),
                        index_staging=index_staging,
                        index_signals=tuple(torch.zeros((1, 8), dtype=torch.bfloat16, device=device) for _ in range(4)),
                        selections=create_head_exchange(group, device,
                            max_rows=-(-config.prefill_chunk_size // shard.world_size),
                            heads=shard.world_size, head_dim=4096, ipc=core_storage.peer_storage))
            if (getattr(context, "candidate_owners", None) is not None
                    and not coordinator.spans_hosts):
                # The owner alternative exists only for the single-host tuning composition;
                # under multi-host candidate_owners is never constructed above, so the
                # selection tuning is skipped with it.
                from ignition.recipes.kimi31.selection_tuning import CandidateSelectionTuning

                context = replace(context, selection_tuning=CandidateSelectionTuning(
                    context, collective, cache_root=config.kernel_cache))
        # Every MLA step gathers each rank's query and KV latent rows, padded to 16 bytes: one exchange for up to
        # eight rows, as vLLM's b12x latent transport carries, and one that prefill steps use in chunks.
        text = weights.config.text
        latent = (text["q_lora_rank"] + text["kv_lora_rank"] + text["qk_rope_head_dim"]) // shard.world_size
        latent_runtimes, latent_rows, latent_columns = _latent_exchange(collective,
            coordinator.spans_hosts, group, device, world_size=shard.world_size, latent=latent,
            channel_rows=channel_rows, max_tokens=max_tokens, ipc=core_storage.peer_storage)
        identity = target_tuning_identity(weights.config, headers, shard.world_size,
            online_mxfp8_scope=config.online_mxfp8_scope,
            prefill_record_exchange=record_exchange.mode, roce_pipe_chunk_bytes=config.roce_pipe_chunk_bytes,
            dcp_size=config.dcp_size)
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device, checkpoint=identity,
            workload_schema="kimi3", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage)
        session.__enter__()
        dcp_session = None
        if split_context:
            dcp_session = RecordedPreparationSession(
                compile_workers=config.compile_workers, collective_tuning=config.collective_tuning,
                autotune_effort=config.autotune_effort,
                native_dense_activations=config.native_dense_activations,
                lock_gpu_clocks=config.lock_gpu_clocks, device=device,
                checkpoint=identity, workload_schema="kimi31-context",
                numerical_policy="default", coordinator=dcp_coordinator,
                headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage)
            context = replace(context, session=dcp_session)
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        deferred_pool = core_storage.scratch_pool()
        with core_storage.scratch_pool() as scratch_pool:
            patches = PATCH_CAPACITIES if config.vision and config.benchmark_selection is None else ()
            cursor, residual_blocks, shared_scratch, text_shapes = allocate_serial_workspace(
                graph_buffers, text_rows=max_tokens, hidden_size=weights.config.hidden,
                residual_count=-(-len(weights.layers) // weights.config.text["attn_res_block_size"]),
                auxiliary_count=len(weights.config.draft.auxiliary_layers) if drafting else 0,
                vision_patches=max(patches, default=0))
            activation, total, kda_output, moe_output, *auxiliary = scratch_views(shared_scratch, text_shapes)
            mla_output = kda_output
            prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, weights.config, shard.world_size, device,
                                               activations=activation[:config.prefill_chunk_size], buffers=cursor)
            decode_storage = Scratch.allocate(config.decode_tokens, config.decode_samples, weights.config, shard.world_size,
                                              device, buffers=cursor)
            vision_storage = {}
            if context is not None and weights.config.dsa is None:
                from .dcp import CONTEXT_WINDOW, RECORD_BYTES

                # Serial prefill layers gather their context one window at a time into one slab beside the chunk's
                # own records; it is released while forwards compile.
                window = min(CONTEXT_WINDOW, config.context)
                context = replace(context, window=window, slab=torch.zeros(
                    (window + config.prefill_chunk_size, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device))
            if context is not None and getattr(context, "staging", None) is not None:
                records, indices = allocate_prefill_histories(config.context, device)
                context = replace(context, staged_records=records, index_records=indices)
        if context is not None and getattr(context, "staging", None) is not None:
            context = replace(context, staged_address=torch.tensor(
                [context.staged_records.data_ptr()], dtype=torch.int64, device=device),
                index_columns=torch.arange(config.context // 64, dtype=torch.int32, device=device))
        if (context is not None and getattr(context, "window_records", 0) > 0
                and getattr(context, "staged_address", None) is None
                and getattr(context, "staged_records", None) is not None):
            # Option W: the remote staged slab needs its self-address for the
            # selections' local peer-record gather (owners all zero).
            context = replace(context, staged_address=torch.tensor(
                [context.staged_records.data_ptr()], dtype=torch.int64, device=device))
        vision_inputs = vision_prefill = vision_state = None
        if patches:
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            # Uploads and published image features are unused by private tuning
            # fixtures. Retain addresses while releasing their physical backing.
            with deferred_pool:
                vision_inputs = VisionInputs.allocate(device)
                vision_state = VisionState(inputs, vision_prefill, requests, image_slots=config.image_concurrency,
                                           image_token=163605, features=FEATURES, hidden=HIDDEN // shard.world_size)
        vision_buffers = {} if not patches else {**vision_inputs.buffers(), **vision_prefill.buffers()}
        buffers = dict(inputs.buffers(), **vision_buffers, **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=core_storage.peer_storage, runtime=collective, coordinator=coordinator)
    forwards, signatures, metadata = {}, {}, {}
    # The first residual mixture consumes the gathered embedding before its
    # storage is reused for the running prefix. Committed blocks stay separate.
    residual = dict(blocks=residual_blocks, prefix=activation)
    decoder_buffers = DecoderStorage.allocate(weights.config, shard, device, allocator=cursor, scratch={
        "residual": residual, "total": total, "kda": {"output": kda_output},
        "mla": {"output": mla_output}, "dsa": {"output": mla_output}, "moe": {"output": moe_output},
        "local_embedding": cursor.zeros((max_tokens, weights.embedding.shape[1]), torch.bfloat16),
        **({"auxiliary": auxiliary[0]} if drafting else {})})
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    if config.benchmark_selection is not None:
        prefill_sizes = config.benchmark_selection.prefill_capacities(config.prefill_chunk_size)
        batch_sizes = config.benchmark_selection.batches
    collectives = dict(runtime=collective, embedding_runtime=embedding_collective, router_runtime=router_collective,
        route_runtime=route_collective, route_remote=route_remote, vocabulary_runtime=vocabulary_collective,
        coordinator=coordinator, vision_state=vision_state, context=context, latent_runtimes=latent_runtimes,
        latent_rows=latent_rows, latent_columns=latent_columns, prefill_route_runtime=prefill_route_collective)
    speculation = None
    if drafting:
        from .dspark import SpeculativePreparation

        extra = dict(markov_runtime=markov_collective)
        if isinstance(weights.config.draft, DFlash2Config):
            from ignition.recipes.kimi31.dflash import SpeculativePreparation
            extra = dict(latent_runtime=latent_runtimes[0], context_latent_runtime=draft_context_collective,
                         convolution_runtime=convolution_collective,
                         context_buffers=decoder_buffers.stage_allocator("draft_context"),
                         block_proposal_broadcast=True)
        elif coordinator.spans_hosts:
            # deepseek41's block broadcast does NOT map to the K3.1 DSpark draft:
            # "block proposal broadcast requires replicated Markov work without a
            # selector" (parallel_draft.py) — the K3.1 DSpark draft uses a selector
            # shortlist, so proposals cannot ride one packed broadcast. These rounds
            # keep per-position proposal distribution over the backbone and the
            # filter-capable chain sampler (recipe top_p range (.95, 1] serves
            # filtered requests). The fused vocabulary-parallel path stays
            # deepseek41-only (planning.py rejects it for every other recipe).
            extra = dict(markov_runtime=markov_collective)
        speculation = SpeculativePreparation(weights, pool, requests, inputs, decode_storage, completion=completion,
            frame=frame, sequence=sequence, session=session, workspace=arena, collectives=collectives,
            decoder_buffers=decoder_buffers, draft_tokens=config.draft_tokens, context_runtime=context_collective,
            page_tokens=config.page_tokens, **extra)
    modes = (drafting,) if config.benchmark_selection is not None else serving_modes(drafting, config.mtp_only)
    count = (len(modes) * (prefill_graph_count(prefill_sizes, scoring) + 4 * len(batch_sizes))
             + (len(batch_sizes) if False in modes else 0)
             + len(config.speculative_round_shapes) + len(patches))
    if config.benchmark_selection is not None:
        count = prefill_graph_count(prefill_sizes, scoring) + len(batch_sizes) + len(config.speculative_round_shapes)
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                packed_token_capacity=packed_tokens)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for kind, mode, sizes in prefill_modes(prefill_sizes,
                    () if enabled or config.benchmark_selection is not None else reversed(batch_sizes), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        session=session, name=name, workspace=arena, buffers=decoder_buffers, **collectives)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" and patches else None,
                        page_tokens=config.page_tokens)
                    # Plain graphs do not update draft context.
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=(None if speculation is None or not enabled
                                         else speculation.forwards.context))
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1,
                        (*forward_inputs, *(vision_prefill.buffers() if mode == "prefill" and patches else ())),
                        enabled=enabled)
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound

            speculation.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
            for batch, width, packed in sorted(config.speculative_round_shapes, key=lambda shape: (shape[2], shape[0]), reverse=True):
                name = f"speculative{batch}x{width}-tokens{packed}"
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                    buffers=speculation.sampling_buffers.cursor(), target_token_capacity=packed)
                forward.prepare()
                add(name, forward, "speculative", batch, width, (*inputs.buffers(), "sampling_grammar_mask"), enabled=True, packed_tokens=packed)

        for capacity in reversed(patches):
            name = f"vision{capacity}"
            with scratch_pool:
                cursor = graph_buffers.cursor()
                vision_storage[capacity] = VisionScratch.allocate(capacity, device, buffers=cursor)
            encoder = PreparedVision(weights.vision, vision_storage[capacity], vision_inputs,
                session=session, workspace=arena, name=name, buffers=cursor)
            forward = PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence,
                                                feature_range=shard.partition(HIDDEN), buffers=cursor)
            add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for capacity in reversed(batch_sizes):
                if config.benchmark_selection is None:
                    name = f"resume{capacity}" + suffix
                    forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                        rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator, session=session, name=name, activation_mode="auto")
                    add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in (("init",) if config.benchmark_selection is not None else ("init", "restore", "snapshot")):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled, page_tokens=config.page_tokens)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()),
                        enabled=enabled)
    decoder_buffers.allocator.join(*decoder_buffers.stage_allocators.values())
    deferred_pool.release_backing()
    try:
        with phase(progress, "Prepare and tune kernels") as stage, transient:
            tuning = session.tune(show_progress=progress is not None, console=stage.console)
            if dcp_session is not None:
                coordinator.barrier("TP tuning drained before DCP tuning", coordinator.ranks)
                tuning["context"] = dcp_session.tune(show_progress=progress is not None, console=stage.console)
                coordinator.barrier("DCP tuning drained before model capture", coordinator.ranks)
    finally:
        deferred_pool.materialize()
    stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()
        deferred_pool.release_backing()

    if context is not None and getattr(context, "selection_tuning", None) is not None:
        with phase(progress, "Tune sparse candidate compositions"), core_storage.transient():
            tuning["candidate_compositions"] = context.selection_tuning.tune(session)

    def materialize():
        scratch_pool.materialize()
        deferred_pool.materialize()
        # Maintenance graphs need snapshot destinations beside live rows;
        # operator tuning does not. Grow before complete-forward warmup.
        core_storage.grow("STATE_ROWS", capture_state_rows)
        pool.memory_capacity[0] = capture_state_rows
        if vision_inputs is not None:
            vision_inputs.vision_slot.fill_(-1)
        pool.clear_for_preparation()

    # Large vision allocations must precede smaller text allocations in the shared graph pool.
    vision_names = tuple(f"vision{capacity}" for capacity in reversed(patches))
    capture_order = (*vision_names, *(name for name in forwards if name not in vision_names))
    capture_catalog = {name: forwards[name] for name in capture_order}
    try:
        graphs, compiled = capture_forwards(capture_catalog, signatures, session=session, device=device,
            stream=stream, pool=graph_pool, collective=collective, coordinator=coordinator,
            compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
            retain_graphs=config.retain_graphs, materialize=materialize,
            warmup_storage=core_storage,
            additional_sessions=() if dcp_session is None else (dcp_session,),
            additional_collectives=() if dcp_session is None else (dcp_collective,),
            compile_kinds=("prefill", "scored_prefill", "decode", "resume", "vision", "speculative"))
    finally:
        scratch_pool.materialize()
        deferred_pool.materialize()
    session.freeze()
    if dcp_session is not None:
        dcp_session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    from .tokenizer import convert, runtime_assets
    assets = runtime_assets(convert(snapshot), template=config.frontend_template)
    sampling_defaults = config.sampling_defaults
    if sampling_defaults is None:
        from .recipe import RECIPE

        sampling_defaults = RECIPE.sampling_defaults.as_dict()
    eos = weights.config.text["eos_token_id"]
    eos_ids = list(dict.fromkeys([*(eos if isinstance(eos, list) else [eos]), 163586]))
    frontend = prepare_frontend(snapshot, eos_token_ids=eos_ids, image_profile="moonvit" if patches else None)
    frontend["template_encoding"] = 1
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=config.page_tokens, frontend=frontend,
        sampling_defaults=sampling_defaults, assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency,
                                        vocabulary=vocabulary, page_tokens=config.page_tokens,
                                        draft_lag=0 if drafting else None, vision=bool(patches)))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=config.devices[rank], weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config), page_tokens=config.page_tokens,
        kv_bytes=pages * page_bytes(weights.config), completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    if scoring is not None:
        memory.update(scoring.memory_report())
    if drafting:
        server_config.update(speculative_widths=tuple(sorted({width for _, width, _ in config.speculative_round_shapes})),
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths),
                                                                  config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_round_shapes=config.speculative_round_shapes,
                      speculative_graph_budget=config.speculative_graph_budget)
    draft_disposable = [] if speculation is None else [speculation.draft_scratch.scratch_tensors(), speculation.context_buffers,
        speculation.kda_records.corrections, speculation.kda_records.keys_and_gates]
    aliases = (("weights.draft.embedding", "weights.embedding"),) if (
        drafting and not isinstance(weights.config.draft, DFlash2Config)
        and weights.draft.embedding.data_ptr() == weights.embedding.data_ptr()
        and weights.draft.embedding.shape == weights.embedding.shape
        and weights.draft.embedding.stride() == weights.embedding.stride()) else ()
    state = None if config.benchmark_selection is None else _benchmark_state(buffers, pool, requests, speculation)
    return PreparedModel(rebuild_weights=declare_weights(weights,
        components=('embedding', 'norm', 'head', 'output_res_score', 'vision', 'draft'), aliases=aliases),
        device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        benchmark_state=state, benchmark_pool=pool if state is not None else None,
        benchmark_requests=requests if state is not None else None,
        benchmark_prefills=None if state is None else {name: value for name, value in metadata.items()
            if signatures[name].kind == "prefill"},
        benchmark_rounds=None if state is None else {name: forward for name, forward in forwards.items()
            if signatures[name].kind == "speculative"},
        close_resources=(() if dcp_session is None else (dcp_session,)) +
                         (() if scoring is None else (scoring,)) +
                        (() if context is None or getattr(context, "candidate_owners", None) is None
                         else (context.candidate_owners,)) +
                        (() if context is None or getattr(context, "staging", None) is None
                         else (context.staging, context.index_staging)),
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges([prefill_storage, decode_storage, graph_buffers.chunks,
            None if speculation is None else speculation.sampling_buffers.chunks, arena.tensor, metadata,
                                           *(() if context is None else (context.slab,)),
                                           *(() if context is None or getattr(context, "staging", None) is None
                                             else (context.staged_records, context.staging.tensor,
                                                   context.index_records, context.index_staging.tensor)),
                                           *(() if not patches else (vision_storage, vision_state.arena)),
                                           decoder_buffers.scratch_tensors(), *draft_disposable]), *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage, context,
                    decode_storage, metadata, compiled, stream, graph_pool, session, coordinator, collective,
                    dcp_session, dcp_coordinator, dcp_collective, dcp_group,
                   vocabulary_collective, embedding_collective, router_collective, route_collective,
                   prefill_route_collective,
                   vision_inputs, vision_prefill, vision_state, vision_storage, deferred_pool, scratch_pool, shared_scratch, graph_buffers,
                   speculation, context_collective, markov_collective, latent_runtimes,
                   draft_context_collective, convolution_collective))
