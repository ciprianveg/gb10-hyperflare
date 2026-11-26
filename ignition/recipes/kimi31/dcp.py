# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Selected-record addressing for interleaved Kimi context shards."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.recipes.kimi3.config import PAGE_TOKENS
from ignition.recipes.kimi3.dcp import (RECORD_BYTES as RECORD_BYTES, ContextShards, _launch,
                                        write_records as write_records)

TOPK = 2048
PREFILL_ROWS = 128


@dataclass(frozen=True)
class AttentionConstants:
    """Immutable one-record-page attention metadata shared by serial layers and slices."""

    pages: torch.Tensor
    starts: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def allocate(cls, device, *, query_capacity=PREFILL_ROWS):
        return cls(torch.arange(PREFILL_ROWS * TOPK, dtype=torch.int32, device=device).view(PREFILL_ROWS, TOPK),
                   torch.arange(max(PREFILL_ROWS, query_capacity) + 1, dtype=torch.int32, device=device),
                   torch.ones(1, dtype=torch.float32, device=device))


@dataclass(frozen=True)
class ContextParallel:
    """Latent-head and exact candidate exchanges shared by sparse attention layers.

    Under multi-host preparation (stage 2) ``heads`` is the ranks' shared RoCEnante
    backbone and ``remote`` selects the remote exchange branches: sliced LSE
    combines with caller-owned staging, and no auxiliary-stream query overlap."""

    shards: ContextShards
    heads: object
    coordinator: object
    peers: dict
    attention_constants: AttentionConstants
    slab: torch.Tensor | None = None
    candidates: object = None
    window: int = 0
    staging: object = None
    staged_records: torch.Tensor | None = None
    staged_address: torch.Tensor | None = None
    stage_signals: tuple = ()
    stage_runtime: object = None
    index_staging: object = None
    index_records: torch.Tensor | None = None
    index_columns: torch.Tensor | None = None
    index_signals: tuple = ()
    selections: object = None
    candidate_owners: object = None
    selection_tuning: object = None
    # A11: the per-layer record replica (PreparedRecordReplica) when the chunk's
    # records publish once per chunk instead of the per-part exchange.
    replica: object = None
    # Option W: records per rank per windowed latent-exchange window; zero disables
    # the windowed exchange (the per-part selected-record exchange applies instead).
    window_records: int = 0
    # Ranks on separate hosts exchange heads through one RoCEnante runtime.
    remote: bool = False
    # Row capacity of the decode/verify head exchanges; DSA sizes its parts from
    # indexer bounds, so this is carried for parity with kimi3's ContextParallel.
    head_rows: int | None = None
    session: object = None


def index_pages(model_pages, output):
    """Map each 128-record model page to its two planar 64-record index pages."""
    output[:, 0::2].copy_(torch.where(model_pages >= 0, model_pages * 2, -1))
    output[:, 1::2].copy_(torch.where(model_pages >= 0, model_pages * 2 + 1, -1))


def pack_candidates(indices, scores, output, shards):
    """Pack FP32 scores and bit-preserved global int32 token IDs for BF16 transport."""
    output[:, :, 0].copy_(torch.where(indices >= 0, scores, -torch.inf))
    output.view(torch.int32)[:, :, 1].copy_(torch.where(indices >= 0,
        indices * shards.world + shards.owner, -1).to(torch.int32))


@triton.jit
def _selection_slots(Selected, Positions, Tables, Owners, Slots, Counts, Rank,
                     columns, table_stride, cache_pages, WORLD: tl.constexpr,
                     LOCAL: tl.constexpr, TOP_K: tl.constexpr, PAGE: tl.constexpr,
                     GLOBAL_RECORDS: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, TOP_K)
    token = tl.load(Selected + row * TOP_K + column).to(tl.int64)
    end = tl.load(Positions + row).to(tl.int64)
    local = tl.maximum(token, 0) // WORLD
    owner = token % WORLD
    block = local // PAGE
    live = (token >= 0) & (token <= end) & (block < columns)
    if LOCAL:
        live &= owner == tl.load(Rank)
    page = tl.load(Tables + row * table_stride + block, mask=live, other=-1).to(tl.int64)
    live &= (page >= 0) & (page < cache_pages)
    target = tl.cumsum(live.to(tl.int32), 0) - 1
    tl.store(Owners + row * TOP_K + column, -1)
    tl.store(Slots + row * TOP_K + column, -1)
    tl.debug_barrier()
    slot = page * PAGE + local % PAGE
    if GLOBAL_RECORDS:
        owner = tl.full((TOP_K,), 0, tl.int64)
        slot = token
    tl.store(Owners + row * TOP_K + target, owner.to(tl.int32), mask=live)
    tl.store(Slots + row * TOP_K + target, slot.to(tl.int32), mask=live)
    tl.store(Counts + row, tl.sum(live.to(tl.int32), 0))


def selection_slots(selected, positions, tables, owners, slots, counts, shards, cache_pages, *, local,
                    global_records=False):
    """Compact valid selections without changing their order.

    ``global_records`` addresses a local global-position slab instead of peer
    page pools; the original page-validity and causal masks still apply.
    """
    torch.library.wrap_triton(_selection_slots)[(selected.shape[0],)](
        selected, positions, tables, owners, slots, counts, shards.index,
        tables.shape[1], tables.stride(0), cache_pages, WORLD=shards.world, LOCAL=local,
        TOP_K=TOPK, PAGE=PAGE_TOKENS, GLOBAL_RECORDS=global_records, num_warps=8)


# Stage 5 exchanges whole 576-byte records as 144 int32 words and index records as 33.
RECORD_WORDS = RECORD_BYTES // 4
BLOCK_WORDS = 256  # next_power_of_2(RECORD_WORDS); the index kernels slice 64 of it
# Selections packed per program of the remote record pack and scatter kernels.
RECORD_SELECTIONS = 8


@triton.jit
def _selection_remote(Selected, Positions, Tables, Owners, Slots, RowCounts, Rank,
                      columns, table_stride, cache_pages, WORLD: tl.constexpr, TOP_K: tl.constexpr,
                      PAGE: tl.constexpr):
    # Address every valid selection by its owning rank's pool record and tally each
    # rank's share of the row. Selections keep their original order: the scatter
    # writes record k back to selection k. Every input is replicated, so owners,
    # slots and tallies come out identical on every rank (the varlen contract).
    row = tl.program_id(0)
    column = tl.arange(0, TOP_K)
    token = tl.load(Selected + row * TOP_K + column).to(tl.int64)
    end = tl.load(Positions + row).to(tl.int64)
    local = tl.maximum(token, 0) // WORLD
    owner = token % WORLD
    block = local // PAGE
    live = (token >= 0) & (token <= end) & (block < columns)
    page = tl.load(Tables + row * table_stride + block, mask=live, other=-1).to(tl.int64)
    live &= (page >= 0) & (page < cache_pages)
    slot = (page * PAGE + local % PAGE).to(tl.int32)
    for rank in tl.static_range(WORLD):
        tl.store(RowCounts + row * WORLD + rank, tl.sum((live & (owner == rank)).to(tl.int32), 0))
    tl.store(Owners + row * TOP_K + column, tl.where(live, owner, -1).to(tl.int32))
    tl.store(Slots + row * TOP_K + column, tl.where(live, slot, -1).to(tl.int32))


def selection_remote(selected, positions, tables, owners, slots, row_counts, shards, cache_pages):
    """Address every valid selection by its owner's pool record, keeping selection order.

    ``owners`` and ``slots`` hold one entry per selection (-1 when invalid);
    ``row_counts`` [rows, world] tallies each rank's live selections of every row.
    """
    if (selected.shape != owners.shape or owners.shape != slots.shape or owners.shape[1] != TOPK
            or row_counts.shape != (selected.shape[0], shards.world) or row_counts.dtype != torch.int32
            or owners.dtype != torch.int32 or slots.dtype != torch.int32):
        raise ValueError("remote selection addressing requires [rows, 2048] owners and slots")
    _launch(_selection_remote, (selected.shape[0],),
        (selected, positions, tables, owners, slots, row_counts, shards.index,
         tables.shape[1], tables.stride(0), cache_pages),
        WORLD=shards.world, TOP_K=TOPK, PAGE=PAGE_TOKENS, num_warps=8)


@triton.jit
def _selection_offsets(Owners, Base, Offsets, WORLD: tl.constexpr, TOP_K: tl.constexpr):
    # Place each selection inside its owner's packed row: the owner's records of the
    # rows before this one, then this row's earlier selections of the same owner.
    row = tl.program_id(0)
    column = tl.arange(0, TOP_K)
    owner = tl.load(Owners + row * TOP_K + column)
    offsets = tl.full((TOP_K,), -1, tl.int32)
    for rank in tl.static_range(WORLD):
        same = owner == rank
        prefix = tl.cumsum(same.to(tl.int32), 0) - same.to(tl.int32)
        base = tl.load(Base + row * WORLD + rank)
        offsets = tl.where(same, base + prefix, offsets)
    tl.store(Offsets + row * TOP_K + column, offsets)


def selection_offsets(owners, base, offsets):
    """Compute each selection's record index within its owner's packed varlen row.

    ``base`` [rows, world] int32 holds each row's exclusive prefix of per-owner
    record counts, so owner r's records of this row start at ``base[row, r]``.
    """
    if (owners.shape != offsets.shape or owners.dtype != torch.int32 or offsets.dtype != torch.int32
            or base.dtype != torch.int32 or base.shape[0] != owners.shape[0]
            or not base.is_contiguous() or base.shape[1] < 1):
        raise ValueError("remote selection offsets require [rows, world] int32 bases")
    _launch(_selection_offsets, (owners.shape[0],), (owners, base, offsets),
            WORLD=base.shape[1], TOP_K=TOPK, num_warps=8)


@triton.jit
def _pack_selected(Pool, Owners, Slots, Offsets, Stage, Rank, row_words, TOP_K: tl.constexpr,
                   WORDS: tl.constexpr, BLOCK: tl.constexpr, SELECTIONS: tl.constexpr):
    # Copy this rank's selected records from its pool into its varlen row at the
    # offsets every rank computed. Programs of other ranks' selections exit early.
    row = tl.program_id(0)
    block = tl.program_id(1) * SELECTIONS + tl.arange(0, SELECTIONS)
    owner = tl.load(Owners + row * TOP_K + block)
    rank = tl.load(Rank).to(tl.int32)
    take = owner == rank
    slot = tl.load(Slots + row * TOP_K + block, mask=take, other=0).to(tl.int64)
    offset = tl.load(Offsets + row * TOP_K + block, mask=take, other=0).to(tl.int64)
    word = tl.arange(0, BLOCK)
    values = tl.load(Pool + slot[:, None] * WORDS + word[None, :],
                     mask=take[:, None] & (word[None, :] < WORDS), other=0)
    tl.store(Stage + rank.to(tl.int64) * row_words + offset[:, None] * WORDS + word[None, :],
             values, mask=take[:, None] & (word[None, :] < WORDS))


@triton.jit
def _scatter_selected(Stage, Owners, Offsets, Output, row_words, TOP_K: tl.constexpr,
                      WORDS: tl.constexpr, BLOCK: tl.constexpr, SELECTIONS: tl.constexpr):
    # Place every owner's packed records at their selections of this row window.
    row = tl.program_id(0)
    block = tl.program_id(1) * SELECTIONS + tl.arange(0, SELECTIONS)
    owner = tl.load(Owners + row * TOP_K + block).to(tl.int64)
    take = owner >= 0
    offset = tl.load(Offsets + row * TOP_K + block, mask=take, other=0).to(tl.int64)
    word = tl.arange(0, BLOCK)
    values = tl.load(Stage + owner[:, None] * row_words + offset[:, None] * WORDS + word[None, :],
                     mask=take[:, None] & (word[None, :] < WORDS), other=0)
    tl.store(Output + (row * TOP_K + block)[:, None] * WORDS + word[None, :], values,
             mask=take[:, None] & (word[None, :] < WORDS))


def pack_selected(pool, owners, slots, offsets, stage, shards):
    """Pack this rank's selected pool records into its row of ``stage``.

    ``pool`` is this rank's latent cache as [records, 144] int32 words; ``stage`` is
    the [world, row_words] int32 varlen staging whose row ``shards.index`` owns.
    """
    if (pool.shape[1] != RECORD_WORDS or pool.stride(1) != 1 or pool.storage_offset() % 4
            or stage.shape[0] != shards.world or stage.shape[1] % RECORD_BYTES
            or owners.shape != slots.shape or slots.shape != offsets.shape):
        raise ValueError("selected-record packing requires word-aligned pool and staging rows")
    rows, topk = owners.shape
    block = triton.next_power_of_2(RECORD_WORDS)
    _launch(_pack_selected, (rows, triton.cdiv(topk, RECORD_SELECTIONS)),
        (pool, owners, slots, offsets, stage.view(torch.int32), shards.index, stage.shape[1] // 4),
        TOP_K=topk, WORDS=RECORD_WORDS, BLOCK=block, SELECTIONS=RECORD_SELECTIONS, num_warps=4)


def scatter_selected(stage, owners, offsets, output, shards):
    """Place every rank's packed records into the selected [rows, selections, 576] FP8 slab."""
    if (output.shape[0] != owners.shape[0] or output.shape[1] != owners.shape[1]
            or output.shape[2] != RECORD_BYTES or output.storage_offset() % 4
            or stage.shape[0] != shards.world or stage.shape[1] % RECORD_WORDS):
        raise ValueError("selected-record scattering requires a [rows, selections, 576] slab")
    rows, topk = owners.shape
    _launch(_scatter_selected, (rows, triton.cdiv(topk, RECORD_SELECTIONS)),
        (stage.view(torch.int32), owners, offsets, output.view(torch.int32), stage.shape[1] // 4),
        TOP_K=topk, WORDS=RECORD_WORDS, BLOCK=triton.next_power_of_2(RECORD_WORDS),
        SELECTIONS=RECORD_SELECTIONS, num_warps=4)


# Index history positions one varlen window carries per rank (glm53's windowed prefill
# gathers): windows past the context's end contribute zero bytes.
INDEX_WINDOW_RECORDS = 65536
INDEX_WORDS = 33  # 128 index-key bytes and one FP32 scale


@triton.jit
def _index_pack(Pool, Table, End, Rank, Stage, columns, first, row_words,
                WORLD: tl.constexpr, PAGE: tl.constexpr, ROWS: tl.constexpr, WORDS: tl.constexpr):
    # Pack this rank's committed index records of local positions [first, first + rows)
    # into its varlen row: 32 key words and the FP32 scale word of each position.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    word = tl.arange(0, 64)
    local_end = tl.maximum((end - rank + WORLD - 1) // WORLD, 0)
    current = first + local
    live = current < local_end
    block = current // PAGE
    page = tl.load(Table + tl.minimum(block, columns - 1), mask=live, other=-1).to(tl.int64)
    live &= (block < columns) & (page >= 0)
    subpage = page * 2 + current % PAGE // 64
    keys = tl.load(Pool + subpage[:, None] * 2112 + (current % 64)[:, None] * 32 + word[None, :],
                   mask=live[:, None] & (word[None, :] < 32), other=0)
    scale = tl.load(Pool + subpage * 2112 + 2048 + current % 64, mask=live, other=0)
    values = tl.where(word[None, :] < 32, keys, scale[:, None])
    tl.store(Stage + rank * row_words + local[:, None] * WORDS + word[None, :], values,
             mask=live[:, None] & (word[None, :] < WORDS))


@triton.jit
def _index_scatter(Stage, End, History, first, row_words,
                   WORLD: tl.constexpr, ROWS: tl.constexpr, WORDS: tl.constexpr):
    # Place every owner's packed index records at their global positions: keys into the
    # position's 64-record planar page, the scale into its trailing word.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    owner = tl.program_id(1).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    word = tl.arange(0, 64)
    local_end = tl.maximum((end - owner + WORLD - 1) // WORLD, 0)
    current = first + local
    live = current < local_end
    token = current * WORLD + owner
    values = tl.load(Stage + owner * row_words + local[:, None] * WORDS + word[None, :],
                     mask=live[:, None] & (word[None, :] < WORDS), other=0)
    destination = token // 64 * 2112
    tl.store(History + destination[:, None] + (token % 64)[:, None] * 32 + word[None, :], values,
             mask=live[:, None] & (word[None, :] < 32))
    tl.store(History + destination + 2048 + token % 64, tl.sum(tl.where(word[None, :] == 32, values, 0), 1),
             mask=live)


def index_window_counts(end, world, first, window):
    """Bytes of index history each rank owns among local positions [first, first + window).

    Counts round up to whole 16-byte packs, as the variable-length gather requires;
    the trailing bytes of a rank's row are never read back.
    """
    ranks = torch.arange(world, device=end.device, dtype=torch.int64)
    local_end = ((end - ranks + world - 1) // world).clamp(min=0)
    owned = (local_end - first).clamp(0, window)
    return (owned * (INDEX_WORDS * 4) + 15) // 16 * 16


def pack_index_records(pool, table, end, stage, shards, first):
    """Pack this rank's index records of local positions from ``first`` into its row.

    ``pool`` is this rank's planar index cache as a flat int32 word view; rows hold
    whole 132-byte records in local order, whatever pages back them.
    """
    if (stage.shape[0] != shards.world or stage.shape[1] % INDEX_WORDS
            or pool.storage_offset() % 4 or table.dtype != torch.int32 or end.shape != (1,)):
        raise ValueError("index packing requires a planar int32 pool and one int64 end")
    window = stage.shape[1] // (4 * INDEX_WORDS)
    words = pool.view(torch.int32).view(-1) if pool.dtype == torch.uint8 else pool.view(-1)
    _launch(_index_pack, (triton.cdiv(window, 16),),
        (words, table, end, shards.index, stage.view(torch.int32), table.numel(), first,
         stage.shape[1] // 4),
        WORLD=shards.world, PAGE=PAGE_TOKENS, ROWS=16, WORDS=INDEX_WORDS, num_warps=4)


def scatter_index_records(stage, end, history, shards, first):
    """Place every rank's gathered index records at their global positions of ``history``."""
    if (history.storage_offset() % 4 or history.shape[1] * history.element_size() != 2112 * 4
            or stage.shape[0] != shards.world or stage.shape[1] % INDEX_WORDS):
        raise ValueError("index scattering requires planar [pages, 2112] int32 history")
    window = stage.shape[1] // (4 * INDEX_WORDS)
    words = history.view(torch.int32) if history.dtype == torch.uint8 else history
    _launch(_index_scatter, (triton.cdiv(window, 16), shards.world),
        (stage.view(torch.int32), end, words, first, stage.shape[1] // 4),
        WORLD=shards.world, ROWS=16, WORDS=INDEX_WORDS, num_warps=4)


# Option W: one window of the windowed latent exchange carries this many records per
# rank. The bootstrap derives the serve window from the backbone's varlen row budget
# (window x 576 bytes must fit it); this default matches the index-history window.
LATENT_WINDOW_RECORDS = 65536


@triton.jit
def _latent_pack(Pool, Table, End, Rank, Stage, columns, first, row_words,
                 WORLD: tl.constexpr, PAGE: tl.constexpr, ROWS: tl.constexpr,
                 WORDS: tl.constexpr, BLOCK: tl.constexpr):
    # Pack this rank's committed latent records of local positions [first, first + rows)
    # into its varlen row: WORDS int32 words (576 bytes) per position, local order.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    word = tl.arange(0, BLOCK)
    local_end = tl.maximum((end - rank + WORLD - 1) // WORLD, 0)
    current = first + local
    live = current < local_end
    block = current // PAGE
    page = tl.load(Table + tl.minimum(block, columns - 1), mask=live, other=-1).to(tl.int64)
    live &= (block < columns) & (page >= 0)
    slot = page * PAGE + current % PAGE
    values = tl.load(Pool + slot[:, None] * WORDS + word[None, :],
                     mask=live[:, None] & (word[None, :] < WORDS), other=0)
    tl.store(Stage + rank * row_words + local[:, None] * WORDS + word[None, :], values,
             mask=live[:, None] & (word[None, :] < WORDS))


@triton.jit
def _latent_scatter(Stage, End, Slab, first, row_words,
                    WORLD: tl.constexpr, ROWS: tl.constexpr, WORDS: tl.constexpr,
                    BLOCK: tl.constexpr):
    # Place every owner's gathered latent records at their global positions of the
    # shared staged slab: one flat [positions, 576] row per token.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    owner = tl.program_id(1).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    word = tl.arange(0, BLOCK)
    local_end = tl.maximum((end - owner + WORLD - 1) // WORLD, 0)
    current = first + local
    live = current < local_end
    token = current * WORLD + owner
    values = tl.load(Stage + owner * row_words + local[:, None] * WORDS + word[None, :],
                     mask=live[:, None] & (word[None, :] < WORDS), other=0)
    tl.store(Slab + token[:, None] * WORDS + word[None, :], values,
             mask=live[:, None] & (word[None, :] < WORDS))


def latent_window_counts(end, world, first, window):
    """Bytes of latent records each rank owns among local positions [first, first + window).

    Derived from the replicated end and world only, so the counts are bit-identical on
    every rank (the PreparedVarlenGather contract) and change between replays without
    host involvement — the in-graph counts discipline. 576-byte records are whole
    16-byte packs, so no rounding slack exists.
    """
    ranks = torch.arange(world, device=end.device, dtype=torch.int64)
    local_end = ((end - ranks + world - 1) // world).clamp(min=0)
    owned = (local_end - first).clamp(0, window)
    return owned * (RECORD_WORDS * 4)


def pack_latent_records(pool, table, end, stage, shards, first):
    """Pack this rank's latent records of local positions from ``first`` into its row.

    ``pool`` is this rank's paged latent cache as a flat int32 word view; rows hold
    whole 576-byte records in local order, whatever pages back them.
    """
    if (stage.shape[0] != shards.world or stage.shape[1] % (4 * RECORD_WORDS)
            or pool.storage_offset() % 4 or table.dtype != torch.int32 or end.shape != (1,)):
        raise ValueError("latent packing requires a paged int32 pool and one int64 end")
    window = stage.shape[1] // (4 * RECORD_WORDS)
    words = pool.view(torch.int32).view(-1) if pool.dtype == torch.uint8 else pool.view(-1)
    _launch(_latent_pack, (triton.cdiv(window, 16),),
        (words, table, end, shards.index, stage.view(torch.int32), table.numel(), first,
         stage.shape[1] // 4),
        WORLD=shards.world, PAGE=PAGE_TOKENS, ROWS=16, WORDS=RECORD_WORDS,
        BLOCK=BLOCK_WORDS, num_warps=4)


def scatter_latent_records(stage, end, slab, shards, first):
    """Place every rank's gathered latent records at their global positions of ``slab``."""
    if (slab.shape[1] != RECORD_BYTES or slab.storage_offset() % 4
            or stage.shape[0] != shards.world or stage.shape[1] % (4 * RECORD_WORDS)):
        raise ValueError("latent scattering requires a planar [positions, 576] slab")
    window = stage.shape[1] // (4 * RECORD_WORDS)
    words = slab.view(torch.int32) if slab.dtype == torch.uint8 else slab.view(torch.int32)
    _launch(_latent_scatter, (triton.cdiv(window, 16), shards.world),
        (stage.view(torch.int32), end, words, first, stage.shape[1] // 4),
        WORLD=shards.world, ROWS=16, WORDS=RECORD_WORDS, BLOCK=BLOCK_WORDS, num_warps=4)


# Query rows of one remote selected-record exchange. One exchange runs per 128-row
# indexer part; its worst case moves every selection's 576-byte record from the one
# rank whose pool holds it (REMOTE_RECORD_ROWS * 2048 * 576 bytes of staging row).
REMOTE_RECORD_ROWS = 128


class PreparedSelectedRecords:
    """Stage-5 exchange of a prefill part's selected latent records over the backbone.

    After the replicated candidate merge every rank holds the bit-identical global
    top-2048 selection of each row, so each rank's byte count is computable identically
    everywhere (the PreparedVarlenGather contract): rank r packs the selected records
    its own pool holds into row r of the shared staging, one variable-length gather
    replicates every rank's pack, and the scatter places them at their selections.
    Selections keep their original order, no peer page is ever read, and the gather is
    self-synchronizing — none of the staged-history publish/consume signals exist.
    """

    def __init__(self, cache, output, storage, shards, *, runtime, session, coordinator, name):
        from ignition.components.collectives import PreparedVarlenGather

        rows, world = REMOTE_RECORD_ROWS, shards.world
        if (cache.ndim != 3 or cache.shape[1] != PAGE_TOKENS or cache.shape[2] != RECORD_BYTES
                or cache.storage_offset() % 4 or output.shape[1:] != (TOPK, RECORD_BYTES)
                or output.shape[0] > rows or output.storage_offset() % 4):
            raise ValueError("remote selected records require paged caches and a [rows, 2048, 576] slab")
        self.shards, self.cache_pages = shards, cache.shape[0]
        self.pool = cache.view(torch.int32).view(-1, RECORD_WORDS)
        self.output = output
        self.owners = storage.buffer("remote_owners", (rows, TOPK), torch.int32)
        self.slots = storage.buffer("remote_slots", (rows, TOPK), torch.int32)
        self.offsets = storage.buffer("remote_offsets", (rows, TOPK), torch.int32)
        self.row_counts = storage.buffer("remote_row_counts", (rows, world), torch.int32)
        self.base = storage.buffer("remote_owner_base", (rows, world), torch.int32)
        self.counts = storage.buffer("remote_counts", (world,), torch.int64)
        # A rank packs up to rows * TOPK records (every selection owned by one rank),
        # so its staging row must hold rows * TOPK * RECORD_BYTES — the runtime's
        # varlen_bytes budget and the gather's per-rank limit both assume it.
        self.stage = storage.buffer("remote_stage", (world, rows * TOPK * RECORD_BYTES), torch.uint8)
        self.gather = PreparedVarlenGather(self.stage, self.counts, runtime=runtime, session=session,
            coordinator=coordinator, name=name + ".records")

    def __call__(self, selected, positions, tables):
        rows = selected.shape[0]
        if rows > REMOTE_RECORD_ROWS or selected.shape[1] != TOPK:
            raise ValueError("remote selected records exchange one indexer part at a time")
        owners, slots = self.owners[:rows], self.slots[:rows]
        selection_remote(selected, positions, tables, owners, slots, self.row_counts[:rows],
                         self.shards, self.cache_pages)
        # Owner bases and byte counts derive from replicated inputs only, so they are
        # bit-identical on every rank and free to change between replays.
        self.base[:rows].copy_(torch.cumsum(self.row_counts[:rows], 0, dtype=torch.int32)
                               .sub_(self.row_counts[:rows]))
        self.counts.copy_(self.row_counts[:rows].sum(0, dtype=torch.int64))
        self.counts.mul_(RECORD_BYTES)
        selection_offsets(owners, self.base[:rows], self.offsets[:rows])
        pack_selected(self.pool, owners, slots, self.offsets[:rows], self.stage, self.shards)
        self.gather()
        scatter_selected(self.stage, owners, self.offsets[:rows], self.output[:rows], self.shards)
        return self.output[:rows]


class PreparedRemoteIndexHistory:
    """Stage-5 windowed varlen exchange that replicates a DSA layer's planar index history.

    Query-sharded scoring needs every position's 132 bytes of keys and scale on every
    rank. Each window packs this rank's committed records of its local positions into
    its row of the shared staging, gathers every rank's pack over the backbone, and
    scatters them at their global positions of the replicated history slab. The
    self-synchronizing gather replaces the staged history's four BF16 signal
    all-reduces; windows past the context's end carry zero bytes.
    """

    def __init__(self, cache, table, end, shards, history, storage, *, runtime, session,
                 coordinator, name):
        from ignition.components.collectives import PreparedVarlenGather

        world = shards.world
        if (cache.dtype != torch.uint8 or cache.shape[1] != 64 * 132 or cache.storage_offset() % 4
                or table.dtype != torch.int32 or end.shape != (1,) or end.dtype != torch.int64
                or history.dtype != torch.uint8 or history.shape[1] != 64 * 132
                or history.storage_offset() % 4):
            raise ValueError("remote index history requires planar uint8 pools and one int64 end")
        self.shards, self.table, self.end = shards, table, end
        self.pool, self.history = cache, history.view(torch.int32)
        self.windows = -(-(history.shape[0] * 64) // (INDEX_WINDOW_RECORDS * world))
        self.stage = storage.buffer("remote_index_stage", (world, INDEX_WINDOW_RECORDS * 132), torch.uint8)
        self.counts = storage.buffer("remote_index_counts", (world,), torch.int64)
        self.gather = PreparedVarlenGather(self.stage, self.counts, runtime=runtime, session=session,
            coordinator=coordinator, name=name + ".history")

    def __call__(self):
        for window in range(self.windows):
            first = window * INDEX_WINDOW_RECORDS
            # Identical on every rank: derived from the replicated end and world only.
            self.counts.copy_(index_window_counts(self.end, self.shards.world, first,
                                                  INDEX_WINDOW_RECORDS))
            pack_index_records(self.pool, self.table, self.end, self.stage, self.shards, first)
            self.gather()
            scatter_index_records(self.stage, self.end, self.history, self.shards, first)
        return self.history


class PreparedRemoteLatentHistory:
    """Option W: windowed varlen exchange that refills the shared staged slab with one
    DSA layer's latent records.

    The staged slab is per-layer scratch (neither history survives the layer): each
    window packs this rank's committed records of its local positions into its row of
    the shared staging, gathers every rank's pack over the backbone, and scatters them
    at their global positions of the slab. Selections then read the slab locally by
    global position — the single-host staged-records read path — through the peer-record
    gather with owners all zero. Zero replica memory: the slab is reused by every layer,
    so the publish covers [0, end), not just the chunk's own range. The counts derive
    in-graph from the replicated end on the same stream as the pack (no host copies —
    the counts-write race class does not exist here by construction), and the
    self-synchronizing gather keeps the windows stream-serialized with no lockstep
    barrier between them.
    """

    def __init__(self, cache, table, end, shards, slab, storage, *, window, runtime, session,
                 coordinator, name):
        from ignition.components.collectives import PreparedVarlenGather

        world = shards.world
        if (cache.ndim != 3 or cache.shape[1] != PAGE_TOKENS or cache.shape[2] != RECORD_BYTES
                or cache.storage_offset() % 4 or table.dtype != torch.int32
                or end.shape != (1,) or end.dtype != torch.int64
                or slab.ndim != 2 or slab.shape[1] != RECORD_BYTES or slab.storage_offset() % 4
                or window < 1):
            raise ValueError("remote latent history requires paged records, a planar slab, "
                             "and one int64 end")
        self.shards, self.table, self.end, self.slab = shards, table, end, slab
        self.pool = cache.view(torch.int32).view(-1, RECORD_WORDS)
        self.window, self.windows = window, -(-slab.shape[0] // (window * world))
        self.stage = storage.buffer("remote_latent_stage", (world, window * RECORD_BYTES), torch.uint8)
        self.counts = storage.buffer("remote_latent_counts", (world,), torch.int64)
        self.gather = PreparedVarlenGather(self.stage, self.counts, runtime=runtime, session=session,
            coordinator=coordinator, name=name + ".latent")

    def __call__(self):
        for window in range(self.windows):
            first = window * self.window
            # Identical on every rank: derived from the replicated end and world only.
            self.counts.copy_(latent_window_counts(self.end, self.shards.world, first,
                                                   self.window))
            pack_latent_records(self.pool, self.table, self.end, self.stage, self.shards, first)
            self.gather()
            scatter_latent_records(self.stage, self.end, self.slab, self.shards, first)
        return self.slab


# A11 record replica: per-chunk bulk publish of each rank's newly committed records into
# every rank's per-layer replica slabs. Replaces the per-part selected-record exchange and
# the per-layer windowed index-history exchange: one varlen gather per record kind per
# chunk (the chunk's only selection barriers), with selections then reading the local
# replica by global position (the single-host staged-records read path). The layers'
# pools and replicas live in one backing each, so the kernels address them by layer
# offset instead of per-layer pointers.


@triton.jit
def _replica_latent_pack(Pool, PoolOffsets, Table, Start, End, Rank, Stage, columns,
                         table_stride, layer_words, row_words,
                         WORLD: tl.constexpr, PAGE: tl.constexpr, ROWS: tl.constexpr,
                         WORDS: tl.constexpr, BLOCK: tl.constexpr, LAYERS: tl.constexpr):
    # Pack this rank's latent records of its own tokens inside [Start, End) of every
    # layer into its varlen row: layer L's records land at offset L * layer_words.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    layer = tl.program_id(1).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    start = tl.load(Start).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    first = (start - rank + WORLD - 1) // WORLD
    current = first + local
    token = current * WORLD + rank
    live = token < end
    block = current // PAGE
    page = tl.load(Table + tl.minimum(block, columns - 1), mask=live, other=-1).to(tl.int64)
    live &= (block < columns) & (page >= 0)
    slot = page * PAGE + current % PAGE
    pool = tl.load(PoolOffsets + layer).to(tl.int64) // 4
    word = tl.arange(0, BLOCK)
    values = tl.load(Pool + pool + slot[:, None] * WORDS + word[None, :],
                     mask=live[:, None] & (word[None, :] < WORDS), other=0)
    # The layers' packs are compacted: layer L's region starts after L records of this
    # rank's owned count, so the gather's counts bytes cover every layer.
    layer_stride = ((end - rank + WORLD - 1) // WORLD - first) * WORDS
    destination = rank * row_words + layer * layer_stride + local[:, None] * WORDS
    tl.store(Stage + destination + word[None, :], values,
             mask=live[:, None] & (word[None, :] < WORDS))


@triton.jit
def _replica_latent_scatter(Stage, Start, End, Replica, layer_words, row_words, replica_stride,
                            WORLD: tl.constexpr, ROWS: tl.constexpr, WORDS: tl.constexpr,
                            BLOCK: tl.constexpr):
    # Place every owner's gathered latent records at their global positions of the
    # layer's replica slab.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    owner = tl.program_id(1).to(tl.int64)
    layer = tl.program_id(2).to(tl.int64)
    start = tl.load(Start).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    first = (start - owner + WORLD - 1) // WORLD
    current = first + local
    token = current * WORLD + owner
    live = token < end
    word = tl.arange(0, BLOCK)
    layer_stride = ((end - owner + WORLD - 1) // WORLD - first) * WORDS
    values = tl.load(Stage + owner * row_words + layer * layer_stride + local[:, None] * WORDS + word[None, :],
                     mask=live[:, None] & (word[None, :] < WORDS), other=0)
    tl.store(Replica + layer * replica_stride + token[:, None] * WORDS + word[None, :], values,
             mask=live[:, None] & (word[None, :] < WORDS))


@triton.jit
def _replica_index_pack(Pool, PoolOffsets, Table, Start, End, Rank, Stage, columns,
                        layer_words, row_words,
                        WORLD: tl.constexpr, ROWS: tl.constexpr, WORDS: tl.constexpr):
    # Pack this rank's committed index records (32 key words + the FP32 scale) of its own
    # tokens inside [Start, End) of every layer into its varlen row.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    layer = tl.program_id(1).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    start = tl.load(Start).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    first = (start - rank + WORLD - 1) // WORLD
    current = first + local
    token = current * WORLD + rank
    live = token < end
    block = current // 128
    page = tl.load(Table + tl.minimum(block, columns - 1), mask=live, other=-1).to(tl.int64)
    live &= (block < columns) & (page >= 0)
    subpage = page * 2 + current % 128 // 64
    pool = tl.load(PoolOffsets + layer).to(tl.int64) // 4
    word = tl.arange(0, 64)
    keys = tl.load(Pool + pool + subpage[:, None] * 2112 + (current % 64)[:, None] * 32 + word[None, :],
                   mask=live[:, None] & (word[None, :] < 32), other=0)
    scale = tl.load(Pool + pool + subpage * 2112 + 2048 + current % 64, mask=live, other=0)
    values = tl.where(word[None, :] < 32, keys, scale[:, None])
    layer_stride = ((end - rank + WORLD - 1) // WORLD - first) * WORDS
    destination = rank * row_words + layer * layer_stride + local[:, None] * WORDS
    tl.store(Stage + destination + word[None, :], values,
             mask=live[:, None] & (word[None, :] < WORDS))


@triton.jit
def _replica_index_scatter(Stage, Start, End, History, layer_words, row_words, history_stride,
                           WORLD: tl.constexpr, ROWS: tl.constexpr, WORDS: tl.constexpr):
    # Place every owner's gathered index records at their global positions of the layer's
    # replicated planar history slab.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    owner = tl.program_id(1).to(tl.int64)
    layer = tl.program_id(2).to(tl.int64)
    start = tl.load(Start).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    first = (start - owner + WORLD - 1) // WORLD
    current = first + local
    token = current * WORLD + owner
    live = token < end
    word = tl.arange(0, 64)
    layer_stride = ((end - owner + WORLD - 1) // WORLD - first) * WORDS
    values = tl.load(Stage + owner * row_words + layer * layer_stride + local[:, None] * WORDS + word[None, :],
                     mask=live[:, None] & (word[None, :] < WORDS), other=0)
    destination = layer * history_stride + token // 64 * 2112
    tl.store(History + destination[:, None] + (token % 64)[:, None] * 32 + word[None, :], values,
             mask=live[:, None] & (word[None, :] < 32))
    tl.store(History + destination + 2048 + token % 64,
             tl.sum(tl.where(word[None, :] == 32, values, 0), 1), mask=live)


def replica_counts(start, end, world, record_bytes, layers, counts):
    """Bytes each rank owns inside [start, end) across every layer, rounded to packs.

    Derived from the replicated start, end and world only, so the counts are
    bit-identical on every rank (the PreparedVarlenGather contract).
    """
    ranks = torch.arange(world, device=end.device, dtype=torch.int64)
    first = ((start - ranks) + world - 1) // world
    last = ((end - ranks) + world - 1) // world
    owned = (last - first).clamp(min=0)
    counts.copy_((owned * record_bytes * layers + 15) // 16 * 16)


# Fused per-layer publish: one 708-byte record per token carries the layer's 576-byte
# E4M3 latent followed by its 132-byte planar index, so one varlen gather per layer
# replaces the two per-kind gathers (the per-op rendezvous and skew dominate their
# small payloads). Region offsets derive from the replicated range only.
PAIR_WORDS = RECORD_WORDS + INDEX_WORDS  # 144 + 33 = 177 words = 708 bytes


def pair_counts(start, end, world, records, counts):
    """Bytes each rank owns inside [start, end), one 708-byte record per token.

    Derived from the replicated start, end and world only, so the counts are
    bit-identical on every rank (the PreparedVarlenGather contract); the trailing
    bytes of a rank's 16-byte-rounded row are never read back.
    """
    ranks = torch.arange(world, device=end.device, dtype=torch.int64)
    first = ((start - ranks) + world - 1) // world
    last = ((end - ranks) + world - 1) // world
    owned = (last - first).clamp(min=0).clamp(max=records)
    counts.copy_((owned * (PAIR_WORDS * 4) + 15) // 16 * 16)


@triton.jit
def _replica_pair_pack(LatentPool, LatentOffset, IndexPool, IndexOffset, Table,
                       Start, End, Rank, Stage, columns, table_stride, row_words,
                       WORLD: tl.constexpr, PAGE: tl.constexpr, ROWS: tl.constexpr,
                       LATENT_WORDS: tl.constexpr, INDEX_WORDS: tl.constexpr,
                       PAIR_WORDS: tl.constexpr, BLOCK: tl.constexpr):
    # Pack this rank's owned records inside [Start, End): 576 latent bytes at the
    # record's start, 132 index bytes (32 key words + the FP32 scale) right after.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    start = tl.load(Start).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    first = (start - rank + WORLD - 1) // WORLD
    current = first + local
    token = current * WORLD + rank
    live = token < end
    block = current // PAGE
    page = tl.load(Table + tl.minimum(block, columns - 1), mask=live, other=-1).to(tl.int64)
    live &= (block < columns) & (page >= 0)
    slot = page * PAGE + current % PAGE
    latent_pool = tl.load(LatentOffset).to(tl.int64) // 4
    index_pool = tl.load(IndexOffset).to(tl.int64) // 4
    word = tl.arange(0, BLOCK)
    latents = tl.load(LatentPool + latent_pool + slot[:, None] * LATENT_WORDS + word[None, :],
                      mask=live[:, None] & (word[None, :] < LATENT_WORDS), other=0)
    tl.store(Stage + rank * row_words + local[:, None] * PAIR_WORDS + word[None, :], latents,
             mask=live[:, None] & (word[None, :] < LATENT_WORDS))
    tail = tl.arange(0, 64)
    subpage = page * 2 + current % PAGE // 64
    keys = tl.load(IndexPool + index_pool + subpage[:, None] * 2112 + (current % 64)[:, None] * 32 + tail[None, :],
                   mask=live[:, None] & (tail[None, :] < 32), other=0)
    scale = tl.load(IndexPool + index_pool + subpage * 2112 + 2048 + current % 64, mask=live, other=0)
    values = tl.where(tail[None, :] < 32, keys, scale[:, None])
    tl.store(Stage + rank * row_words + local[:, None] * PAIR_WORDS + LATENT_WORDS + tail[None, :], values,
             mask=live[:, None] & (tail[None, :] < INDEX_WORDS))


@triton.jit
def _replica_pair_latent_scatter(Stage, Start, End, Replica, row_words, replica_stride,
                                 WORLD: tl.constexpr, ROWS: tl.constexpr,
                                 LATENT_WORDS: tl.constexpr, PAIR_WORDS: tl.constexpr,
                                 BLOCK: tl.constexpr):
    # Place every owner's gathered latent records at their global positions of the
    # layer's replica slab: each record's first 576 bytes.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    owner = tl.program_id(1).to(tl.int64)
    start = tl.load(Start).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    first = (start - owner + WORLD - 1) // WORLD
    current = first + local
    token = current * WORLD + owner
    live = token < end
    word = tl.arange(0, BLOCK)
    values = tl.load(Stage + owner * row_words + local[:, None] * PAIR_WORDS + word[None, :],
                     mask=live[:, None] & (word[None, :] < LATENT_WORDS), other=0)
    tl.store(Replica + token[:, None] * LATENT_WORDS + word[None, :], values,
             mask=live[:, None] & (word[None, :] < LATENT_WORDS))


@triton.jit
def _replica_pair_index_scatter(Stage, Start, End, History, row_words, history_stride,
                                WORLD: tl.constexpr, ROWS: tl.constexpr,
                                LATENT_WORDS: tl.constexpr, INDEX_WORDS: tl.constexpr,
                                PAIR_WORDS: tl.constexpr):
    # Place every owner's gathered index records at their global positions of the
    # layer's replicated planar history slab: words 144..176 of each fused record.
    local = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    owner = tl.program_id(1).to(tl.int64)
    start = tl.load(Start).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    first = (start - owner + WORLD - 1) // WORLD
    current = first + local
    token = current * WORLD + owner
    live = token < end
    word = tl.arange(0, 64)
    values = tl.load(Stage + owner * row_words + local[:, None] * PAIR_WORDS + LATENT_WORDS + word[None, :],
                     mask=live[:, None] & (word[None, :] < INDEX_WORDS), other=0)
    destination = token // 64 * 2112
    tl.store(History + destination[:, None] + (token % 64)[:, None] * 32 + word[None, :], values,
             mask=live[:, None] & (word[None, :] < 32))
    tl.store(History + destination + 2048 + token % 64,
             tl.sum(tl.where(word[None, :] == 32, values, 0), 1), mask=live)


def replica_capacity(free_bytes, layers, context_tokens, chunk_records, *, headroom_bytes):
    """Replica token capacity that leaves the tuning session's scratch budget free.

    The session's scratch limit is computed from free device memory at tuning time
    (candidate_scratch_limit: free - headroom - tuning growth, divided by pools), so
    the replica's allocation must keep that reserve free: the session headroom (the
    larger of the 8 GiB default and the caller's), the 1 GiB tuning growth, and a
    2 GiB serving margin.
    """
    per_token = layers * (RECORD_BYTES + 132)
    reserve = max(8 << 30, headroom_bytes) + (1 << 30) + (2 << 30)
    tokens = max(0, free_bytes - reserve) // per_token
    capacity = min(context_tokens, tokens)
    return capacity if capacity >= chunk_records else 0


class PreparedRecordReplica:
    """Per-chunk bulk publish of newly committed records into per-layer replica slabs.

    Each rank owns one token in ``world``; the chunk's own range [start, end) holds
    chunk/world records per rank per layer, which one variable-length gather per record
    kind replicates to every rank (the chunk's only selection barriers — two ops
    instead of the per-part exchange's 1,760). Selections then read the local replica
    by global position; rows past ``end`` keep their previous contents and are never
    read before the publish that covers them.
    """

    def __init__(self, layers, capacity, chunk_records, backing, shards, storage, *, fused_publish=False):
        world = shards.world
        if chunk_records > capacity:
            raise ValueError("the replica publish range must fit the per-rank record capacity")
        self.layers, self.capacity, self.shards, self.backing = layers, capacity, shards, backing
        self.fused_publish = fused_publish
        self.gather_latent = self.gather_index = self.gather_pair = None
        self.debug_no_gather = False
        self.stage_latent = storage.buffer("replica_stage_latent",
            (world, layers * chunk_records * RECORD_BYTES), torch.uint8)
        self.stage_index = storage.buffer("replica_stage_index",
            (world, layers * chunk_records * 132), torch.uint8)
        self.stage_pair = storage.buffer("replica_stage_pair",
            (world, (chunk_records * PAIR_WORDS * 4 + 15) // 16 * 16), torch.uint8)
        self.counts_latent = storage.buffer("replica_counts_latent", (world,), torch.int64)
        self.counts_index = storage.buffer("replica_counts_index", (world,), torch.int64)
        self.counts_pair = storage.buffer("replica_counts_pair", (world,), torch.int64)
        self.start = storage.buffer("replica_start", (1,), torch.int64)
        self.end = storage.buffer("replica_end", (1,), torch.int64)
        self.latents = storage.buffer("replica_latent",
            (layers, capacity, RECORD_BYTES), torch.float8_e4m3fn)
        self.indices = storage.buffer("replica_index",
            (layers, -(-capacity // 64), 64 * 132), torch.uint8)
        self.latent_pools = None
        self.index_pools = None
        self.latent_table = None

    def bind(self, runtime, session, coordinator, name):
        """Create the publish gathers once the preparation session exists."""
        from ignition.components.collectives import PreparedVarlenGather

        if self.gather_latent is None:
            self.gather_latent = PreparedVarlenGather(self.stage_latent, self.counts_latent,
                runtime=runtime, session=session, coordinator=coordinator, name=name + ".latent")
            self.gather_index = PreparedVarlenGather(self.stage_index, self.counts_index,
                runtime=runtime, session=session, coordinator=coordinator, name=name + ".index")
            self.gather_pair = PreparedVarlenGather(self.stage_pair, self.counts_pair,
                runtime=runtime, session=session, coordinator=coordinator, name=name + ".pair")

    def register(self, layer, latent_pool, index_pool, table):
        """Carry one layer's pools (views of the shared backing) and the request page table."""
        offset = lambda tensor: tensor.data_ptr() - self.backing.data_ptr()
        if self.latent_pools is None:
            self.latent_pools = torch.zeros(self.layers, dtype=torch.int64, device=table.device)
            self.index_pools = torch.zeros(self.layers, dtype=torch.int64, device=table.device)
        self.latent_pools[layer] = offset(latent_pool)
        self.index_pools[layer] = offset(index_pool)
        self.latent_table = table

    def layer(self, index):
        """This layer's replica slabs: (latent [capacity, 576] E4M3, index [pages, 2112 words])."""
        return self.latents[index], self.indices[index]

    def publish_latents(self, metadata, layer):
        """Publish this layer's latent records of [starts, starts + lengths) into the
        layer's replica slab; called by the layer right after its records are written
        (records are per-layer and written progressively, so the publish is per-layer
        too — one varlen gather per layer per chunk)."""
        start = metadata.starts[0].reshape(1)
        end = (metadata.starts[0] + metadata.lengths[0]).reshape(1)
        # No capacity check here: the comparison would force a device sync inside the
        # captured graph.  The range is bounded by construction — the publish stages
        # ceil(prefill_chunk / world) records per rank, which the bootstrap sizes.
        self.start.copy_(start)
        self.end.copy_(end)
        world, row_words = self.stage_latent.shape[0], self.stage_latent.shape[1] // 4
        records = row_words // RECORD_WORDS
        replica_counts(self.start, self.end, world, RECORD_BYTES, 1, self.counts_latent)
        _launch(_replica_latent_pack, (triton.cdiv(records, 16), 1),
            (self.backing.view(torch.int32).view(-1), self.latent_pools[layer:layer + 1],
             self.latent_table, self.start, self.end, self.shards.index,
             self.stage_latent.view(torch.int32).view(-1), self.latent_table.numel(),
             self.latent_table.stride(0), 0, row_words),
            WORLD=world, PAGE=PAGE_TOKENS, ROWS=16, WORDS=RECORD_WORDS,
            BLOCK=BLOCK_WORDS, LAYERS=1, num_warps=4)
        if not self.debug_no_gather:
            self.gather_latent()
        _launch(_replica_latent_scatter, (triton.cdiv(records, 16), world, 1),
            (self.stage_latent.view(torch.int32).view(-1), self.start, self.end,
             self.latents[layer:].view(torch.int32).view(-1), 0, row_words,
             self.latents.stride(0) // 4),
            WORLD=world, ROWS=16, WORDS=RECORD_WORDS, BLOCK=BLOCK_WORDS, num_warps=4)

    def publish_indexes(self, metadata, layer):
        """Publish this layer's index records; called by the layer's query-sharded
        scorer at the point where the windowed history exchange used to run."""
        start = metadata.starts[0].reshape(1)
        end = (metadata.starts[0] + metadata.lengths[0]).reshape(1)
        self.start.copy_(start)
        self.end.copy_(end)
        world, row_words = self.stage_index.shape[0], self.stage_index.shape[1] // 4
        records = row_words // INDEX_WORDS
        replica_counts(self.start, self.end, world, 132, 1, self.counts_index)
        _launch(_replica_index_pack, (triton.cdiv(records, 16), 1),
            (self.backing.view(torch.int32).view(-1), self.index_pools[layer:layer + 1],
             self.latent_table, self.start, self.end, self.shards.index,
             self.stage_index.view(torch.int32).view(-1), self.latent_table.numel(),
             0, row_words),
            WORLD=world, ROWS=16, WORDS=INDEX_WORDS, num_warps=4)
        if not self.debug_no_gather:
            self.gather_index()
        _launch(_replica_index_scatter, (triton.cdiv(records, 16), world, 1),
            (self.stage_index.view(torch.int32).view(-1), self.start, self.end,
             self.indices[layer:].view(torch.int32).view(-1), 0, row_words,
             self.indices.stride(0) // 4),
            WORLD=world, ROWS=16, WORDS=INDEX_WORDS, num_warps=4)

    def publish(self, metadata, layer):
        """Publish this layer's latent AND index records of [starts, starts + lengths)
        in one varlen gather.

        The layer's latent records (written by the attention's write_records) and index
        keys (written by the indexer's projection) are both committed before the
        attention calls this, so one 708-byte-per-token fused record — 576 latent bytes
        followed by 132 index bytes — replaces the two per-kind gathers. The per-op
        rendezvous and its skew dominate these small payloads, so halving the publishes
        halves their share of the frame's pipe ops. Region offsets inside each rank's
        staged row derive from the replicated range only, so every rank agrees on the
        layout without communication (the PreparedVarlenGather contract)."""
        start = metadata.starts[0].reshape(1)
        end = (metadata.starts[0] + metadata.lengths[0]).reshape(1)
        self.start.copy_(start)
        self.end.copy_(end)
        world, row_words = self.stage_pair.shape[0], self.stage_pair.shape[1] // 4
        records = row_words // PAIR_WORDS
        pair_counts(self.start, self.end, world, records, self.counts_pair)
        _launch(_replica_pair_pack, (triton.cdiv(records, 16),),
            (self.backing.view(torch.int32).view(-1), self.latent_pools[layer:layer + 1],
             self.index_pools[layer:layer + 1], self.latent_table, self.start, self.end,
             self.shards.index, self.stage_pair.view(torch.int32).view(-1),
             self.latent_table.numel(), self.latent_table.stride(0), row_words),
            WORLD=world, PAGE=PAGE_TOKENS, ROWS=16, LATENT_WORDS=RECORD_WORDS,
            INDEX_WORDS=INDEX_WORDS, PAIR_WORDS=PAIR_WORDS, BLOCK=BLOCK_WORDS, num_warps=4)
        if not self.debug_no_gather:
            self.gather_pair()
        _launch(_replica_pair_latent_scatter, (triton.cdiv(records, 16), world),
            (self.stage_pair.view(torch.int32).view(-1), self.start, self.end,
             self.latents[layer:].view(torch.int32).view(-1), row_words,
             self.latents.stride(0) // 4),
            WORLD=world, ROWS=16, LATENT_WORDS=RECORD_WORDS, PAIR_WORDS=PAIR_WORDS,
            BLOCK=BLOCK_WORDS, num_warps=4)
        _launch(_replica_pair_index_scatter, (triton.cdiv(records, 16), world),
            (self.stage_pair.view(torch.int32).view(-1), self.start, self.end,
             self.indices[layer:].view(torch.int32).view(-1), row_words,
             self.indices.stride(0) // 4),
            WORLD=world, ROWS=16, LATENT_WORDS=RECORD_WORDS, INDEX_WORDS=INDEX_WORDS,
            PAIR_WORDS=PAIR_WORDS, num_warps=4)
