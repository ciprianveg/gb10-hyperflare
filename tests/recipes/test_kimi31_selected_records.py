# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-5 remote exchange kernels against single-rank torch references.

The selected-record kernels (addressing, offsets, pack, scatter) and the windowed
index-history kernels are checked bitwise against direct torch copies over the same
inputs, including adversarial selections: empty rows, every selection owned by one
rank, duplicate tokens, and unmapped pages. One GPU emulates every rank by driving
the rank-addressed pack kernel once per owner; the two-host RoCE test separately
qualifies the real exchange over the backbone.
"""

import os

import pytest
import torch

from ignition.recipes.kimi3.config import PAGE_TOKENS
from ignition.recipes.kimi3.dcp import ContextShards
from ignition.recipes.kimi31.dcp import (INDEX_WORDS, RECORD_BYTES, RECORD_WORDS, TOPK,
    index_window_counts, pack_index_records, pack_selected, scatter_index_records,
    scatter_selected, selection_offsets, selection_remote)

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned GPU")


def shard(rank, world, device):
    return ContextShards.on(rank, world, device)


def make_selections(rows, world, end_max, device, *, seed, empty=False, one_owner=False,
                    duplicates=False, columns=8, cache_pages=6):
    """Random valid selections with the requested adversarial shape."""
    generator = torch.Generator(device=device).manual_seed(seed)
    end = torch.randint(1, end_max + 1, (rows,), device=device, generator=generator)
    if empty:
        selected = torch.full((rows, TOPK), -1, device=device, dtype=torch.int32)
    elif duplicates:
        stride = max(1, end_max // TOPK)
        selected = (torch.randint(0, max(1, end_max // 4), (rows, 1), device=device,
                                  generator=generator)
                    + torch.arange(TOPK, device=device) % stride).int() % end[:, None].clamp(min=1)
    else:
        selected = torch.randint(0, end_max, (rows, TOPK), device=device,
                                 generator=generator).int()
    if one_owner:
        selected = selected // world * world  # every position owned by rank zero
    tables = torch.randint(-1, cache_pages, (rows, columns), device=device,
                           generator=generator).int()
    return selected, end.int(), tables


def reference_selection(selected, positions, tables, world, cache_pages):
    """Torch mirror of the kernels' addressing, tallies, bases and offsets."""
    rows = selected.shape[0]
    token = selected.long()
    end = positions.long()
    local = token.clamp(min=0) // world
    owner = token % world
    block = local // PAGE_TOKENS
    live = (token >= 0) & (token <= end[:, None]) & (block < tables.shape[1])
    page = torch.where(live, tables.gather(1, block.clamp(0, tables.shape[1] - 1)).long(), -1)
    live = live & (page >= 0) & (page < cache_pages)
    slot = torch.where(live, page * PAGE_TOKENS + local % PAGE_TOKENS, -1)
    owners = torch.where(live, owner, -1).int()
    row_counts = torch.stack([((live) & (owner == rank)).sum(1) for rank in range(world)], 1).int()
    base = row_counts.cumsum(0, dtype=torch.int32) - row_counts
    offsets = torch.full((rows, TOPK), -1, device=selected.device, dtype=torch.int32)
    for rank in range(world):
        same = owners == rank
        prefix = same.int().cumsum(1) - same.int()
        offsets = torch.where(same, (base[:, rank][:, None] + prefix).int(), offsets)
    return owners, slot.int(), row_counts, base, offsets.int()


def emulate_exchange(selected, positions, tables, world, cache_pages, pools, device, rows):
    """Drive the kernels once per owner on one GPU and return the gathered slab."""
    owners, slots, row_counts, base, offsets = reference_selection(
        selected, positions, tables, world, cache_pages)
    got_owners = torch.full_like(owners, -7)
    got_slots = torch.full_like(slots, -7)
    got_counts = torch.full_like(row_counts, -7)
    selection_remote(selected, positions, tables, got_owners, got_slots, got_counts,
                     shard(0, world, device), cache_pages)
    assert torch.equal(got_owners, owners)
    assert torch.equal(got_slots, slots)
    assert torch.equal(got_counts, row_counts)
    got_offsets = torch.full_like(offsets, -7)
    selection_offsets(got_owners, base.contiguous(), got_offsets)
    assert torch.equal(got_offsets, offsets)
    stage = torch.zeros((world, rows * TOPK * RECORD_BYTES), device=device, dtype=torch.uint8)
    for rank in range(world):
        pack_selected(pools[rank].view(torch.int32).view(-1, RECORD_WORDS), got_owners,
                      got_slots, got_offsets, stage, shard(rank, world, device))
    output = torch.full((rows, TOPK, RECORD_BYTES), 0x55, device=device, dtype=torch.uint8)
    scatter_selected(stage, got_owners, got_offsets, output,
                     shard(0, world, device))
    counts = row_counts.sum(0, dtype=torch.int64) * RECORD_BYTES
    expected = torch.full_like(output, 0x55)
    live = owners >= 0
    expected[live] = torch.stack(pools, 0).to(torch.uint8)[
        owners.long().clamp(min=0)[live], slots.long()[live]]
    return output, expected, counts, row_counts, stage


@torch.no_grad()
def test_selected_record_exchange_matches_direct_copy():
    device = torch.device("cuda:0")
    for world, rows, seed in ((2, 128, 21), (8, 128, 22), (8, 37, 23), (16, 128, 24)):
        selected, positions, tables = make_selections(rows, world, 900, device, seed=seed)
        pools = [torch.randint(0, 256, (6 * PAGE_TOKENS, RECORD_BYTES), device=device,
                               dtype=torch.uint8) for _ in range(world)]
        output, expected, counts, row_counts, stage = emulate_exchange(
            selected, positions, tables, world, 6, pools, device, rows)
        assert torch.equal(output, expected)
        # The varlen contract: counts are exact byte prefixes of each owner's row.
        for rank in range(world):
            assert counts[rank] == row_counts[:, rank].sum() * RECORD_BYTES
            assert counts[rank] <= stage.shape[1]


@torch.no_grad()
def test_selected_exchange_handles_empty_and_single_owner_rows():
    device = torch.device("cuda:0")
    world, rows = 8, 128
    for empty, one_owner in ((True, False), (False, True)):
        selected, positions, tables = make_selections(rows, world, 5000, device, seed=31,
            empty=empty, one_owner=one_owner)
        pools = [torch.randint(0, 256, (6 * PAGE_TOKENS, RECORD_BYTES), device=device,
                               dtype=torch.uint8) for _ in range(world)]
        output, expected, counts, row_counts, _ = emulate_exchange(
            selected, positions, tables, world, 6, pools, device, rows)
        assert torch.equal(output, expected)
        if empty:
            assert counts.sum() == 0
        if one_owner:
            assert counts[0] == row_counts[:, 0].sum() * RECORD_BYTES and counts[1:].sum() == 0


@torch.no_grad()
def test_selected_exchange_preserves_duplicate_token_order():
    device = torch.device("cuda:0")
    world, rows = 8, 128
    selected, positions, tables = make_selections(rows, world, 4000, device, seed=77,
                                                  duplicates=True)
    pools = [torch.randint(0, 256, (6 * PAGE_TOKENS, RECORD_BYTES), device=device,
                           dtype=torch.uint8) for _ in range(world)]
    output, expected, _, _, _ = emulate_exchange(selected, positions, tables, world, 6,
                                                 pools, device, rows)
    assert torch.equal(output, expected)


def reference_index_stage(pools, table, end, world, first, window):
    """Torch mirror of one window's per-owner index packing."""
    stage = torch.zeros((world, window * INDEX_WORDS), device=pools[0].device, dtype=torch.int32)
    for rank in range(world):
        local_end = max((end.item() - rank + world - 1) // world, 0)
        words_all = pools[rank].view(torch.int32).view(-1)
        for local in range(first, min(first + window, local_end)):
            page = table[local // PAGE_TOKENS].item()
            if page < 0:
                continue
            subpage = page * 2 + (local % PAGE_TOKENS) // 64
            words = words_all[subpage * 2112:subpage * 2112 + 2112]
            stage[rank, (local - first) * INDEX_WORDS:(local - first + 1) * INDEX_WORDS] = torch.cat(
                (words[(local % 64) * 32:(local % 64) * 32 + 32],
                 words[2048 + local % 64:2048 + local % 64 + 1]))
    return stage


@torch.no_grad()
def test_index_history_windows_match_reference():
    device = torch.device("cuda:0")
    for world, pages, seed in ((2, 4, 41), (8, 6, 42)):
        del seed
        window = 512
        maximum = pages * PAGE_TOKENS * world
        pools = [torch.randint(0, 256, (2 * pages, 64 * 132), device=device, dtype=torch.uint8)
                 for _ in range(world)]
        table = torch.randint(-1, pages, (pages,), device=device).int()
        stage = torch.zeros((world, window * 132), device=device, dtype=torch.uint8)
        history = torch.zeros((maximum // 64, 64 * 132), device=device, dtype=torch.uint8)
        for end_value in (0, 1, 63, 64, maximum // 3, maximum):
            end = torch.tensor([end_value], device=device, dtype=torch.int64)
            ranks = torch.arange(world, device=device, dtype=torch.int64)
            local_end = ((end - ranks + world - 1) // world).clamp(min=0)
            history.zero_()
            expected_full = torch.zeros_like(history.view(torch.int32))
            for first in range(0, -(-maximum // world), window):
                owned = (local_end - first).clamp(0, window)
                assert torch.equal(index_window_counts(end, world, first, window),
                                   (owned * 132 + 15) // 16 * 16)
                stage.zero_()
                for rank in range(world):
                    pack_index_records(pools[rank], table, end, stage, shard(rank, world, device),
                                       first)
                assert torch.equal(stage.view(torch.int32),
                                   reference_index_stage(pools, table, end, world, first, window))
                scatter_index_records(stage, end, history, shard(0, world, device), first)
            for owner in range(world):
                words_all = pools[owner].view(torch.int32).view(-1)
                for local in range(0, local_end[owner].item()):
                    token = local * world + owner
                    page = table[local // PAGE_TOKENS].item()
                    if page < 0:
                        continue
                    subpage = page * 2 + (local % PAGE_TOKENS) // 64
                    words = words_all[subpage * 2112:subpage * 2112 + 2112]
                    expected_full[token // 64, (token % 64) * 32:(token % 64) * 32 + 32] = \
                        words[(local % 64) * 32:(local % 64) * 32 + 32]
                    expected_full[token // 64, 2048 + token % 64] = words[2048 + local % 64]
            assert torch.equal(history.view(torch.int32), expected_full)


@torch.no_grad()
def test_exchange_wrapper_validation_rejects_bad_geometry():
    device = torch.device("cpu")
    shards = ContextShards.on(0, 8, device)
    owners = torch.zeros((4, TOPK), device=device, dtype=torch.int32)
    slots = torch.zeros((4, TOPK), device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="remote selection addressing"):
        selection_remote(torch.zeros((4, TOPK), device=device, dtype=torch.int32),
                         torch.zeros(4, device=device),
                         torch.zeros((4, 8), device=device, dtype=torch.int32),
                         owners, slots, torch.zeros((4, 7), device=device, dtype=torch.int32),
                         shards, 6)
    with pytest.raises(ValueError, match="remote selection offsets"):
        selection_offsets(owners, torch.zeros((4, 8), device=device, dtype=torch.int64),
                          torch.zeros((4, TOPK), device=device, dtype=torch.int32))
    with pytest.raises(ValueError, match="selected-record packing"):
        pack_selected(torch.zeros((4, 144), device=device, dtype=torch.int32), owners, slots,
                      owners, torch.zeros((7, 256 * RECORD_BYTES), device=device, dtype=torch.uint8),
                      shards)
    with pytest.raises(ValueError, match="selected-record scattering"):
        scatter_selected(torch.zeros((8, 256 * RECORD_BYTES), device=device, dtype=torch.uint8),
                         owners, owners, torch.zeros((4, TOPK, 100), device=device, dtype=torch.uint8),
                         shards)
    with pytest.raises(ValueError, match="index packing"):
        pack_index_records(torch.zeros((4, 64 * 132), device=device, dtype=torch.uint8),
                           torch.zeros((4,), device=device, dtype=torch.int64),
                           torch.zeros(1, device=device, dtype=torch.int64),
                           torch.zeros((8, 512 * 132), device=device, dtype=torch.uint8), shards, 0)
    with pytest.raises(ValueError, match="index scattering"):
        scatter_index_records(torch.zeros((8, 512 * 132), device=device, dtype=torch.uint8),
                              torch.zeros(1, device=device, dtype=torch.int64),
                              torch.zeros((4, 64 * 100), device=device, dtype=torch.uint8), shards, 0)
