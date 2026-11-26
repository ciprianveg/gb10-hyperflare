# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-4 remote DSA candidate exchange: replicated composition wiring and pack parity.

Every rewired site is checked with fake runtimes: under multi-host preparation the
candidate and selection gathers are the shared Stage-1 RoCE backbone, the PCIe owner
exchange and its tuning composition are never constructed, and the replicated merge
packs FP32 (score, global-id) pairs through a byte-exact BF16 view. The real transport
and the exact local fold are exercised by the two-host RoCE test; here the exchange is
simulated with the protocol's rank-major row layout.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import candidate_selection
from ignition.components.ordered_row_topk import PreparedOrderedRowTopK, fused_cutoff_enabled
from ignition.recipes.kimi31 import indexer as indexer_module
from ignition.recipes.kimi31.indexer import PreparedIndexSelection, SharedBuffers


class FakeExchange:
    """Simulates the backbone all-gather: rank-major packed rows into the receive buffer."""

    packed_rows = ()
    sources = []

    def __init__(self, source, output, **kwargs):
        self.source, self.output = source, output
        FakeExchange.sources.append(self)

    def __call__(self, source):
        self.output.copy_(torch.cat(list(self.packed_rows), dim=1)[:, :self.output.shape[1]])


def _candidate_pattern(rows, world, rank, seed):
    """One rank's local candidates and the global reference they pack into.

    Valid token IDs are rank-local (0..2047); the merge maps them to global
    ``id * world + rank`` pairs. Padding is the canonical (-1, -inf) suffix.
    """
    generator = torch.Generator().manual_seed(seed * 61 + rank)
    valid = torch.randint(0, 2048 + 1, (rows,), generator=generator)
    ids = torch.where(torch.arange(2048)[None] < valid[:, None],
                      torch.randint(0, 2048, (rows, 2048), generator=generator), -1).to(torch.int32)
    scores = torch.where(ids >= 0, torch.randn((rows, 2048), generator=generator),
                         torch.full((), float("-inf")))
    return ids, scores


def _reference_packing(world, patterns):
    """The global gathered buffer every rank must observe after the all-gather."""
    rows = patterns[0][0].shape[0]
    packed = torch.empty((rows, 2048 * world, 2), dtype=torch.float32)
    values, mapped = [], []
    for rank, (ids, scores) in enumerate(patterns):
        values.append(scores)
        mapped.append(torch.where(ids >= 0, ids * world + rank, -1).to(torch.int32))
    packed[:, :, 0] = torch.cat(values, dim=1)
    packed.view(torch.int32)[:, :, 1] = torch.cat(mapped, dim=1)
    return packed


@pytest.mark.parametrize("world,rows,seed", [(2, 3, 1), (8, 2, 2), (16, 4, 3)])
def test_replicated_candidate_pack_gathers_rank_major_at_every_world(monkeypatch, world, rows, seed):
    from ignition.recipes.kimi3.dcp import ContextShards

    patterns = [_candidate_pattern(rows, world, rank, seed) for rank in range(world)]
    reference = _reference_packing(world, patterns)
    FakeExchange.sources = []
    FakeExchange.packed_rows = ()
    monkeypatch.setattr(candidate_selection, "PreparedAllGather", FakeExchange)
    # The compact and fold run real triton kernels (exercised by the two-host RoCE
    # test against the integer oracle); the pack and transport protocol is CPU-checked.
    monkeypatch.setattr(candidate_selection, "PreparedCandidateRows",
                        lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(candidate_selection, "PreparedOrderedRowTopK",
                        lambda *args, **kwargs: SimpleNamespace())
    packed_rows = []
    for rank, (ids, scores) in enumerate(patterns):
        shards = ContextShards.on(rank, world, torch.device("cpu"))
        selection = candidate_selection.PreparedCandidateSelection(
            ids, scores, torch.arange(rows, dtype=torch.int64), shards.index,
            torch.zeros((rows, 2048), dtype=torch.int32), world=world,
            gather_runtime=SimpleNamespace(world_size=world), owner_runtime=None,
            reduction_runtime=None, session=None, coordinator=None, name=f"merge{world}.{rows}",
            buffers=SharedBuffers({}, "cpu"), alternatives=("replicated",))
        assert selection.owner is None
        selection.bind("replicated")
        packed = selection.pack("replicated")
        packed_rows.append(packed.view(torch.bfloat16).view(rows, -1))
        # The packed pairs carry the exact FP32 score bits and the mapped global ids.
        assert torch.equal(packed.view(torch.int32)[:, :, 1],
                           reference.view(torch.int32)[:, rank * 2048:(rank + 1) * 2048, 1])
        assert torch.equal(packed[:, :, 0], reference[:, rank * 2048:(rank + 1) * 2048, 0])
    # Simulate the rank-major gather through the BF16 transport view: every rank observes
    # the same global buffer, byte-exact.
    FakeExchange.packed_rows = tuple(packed_rows)
    gathered = torch.empty((rows, 2048 * world, 2), dtype=torch.float32).view(torch.bfloat16).view(rows, -1)
    exchange = FakeExchange(packed_rows[0], gathered)
    exchange(packed_rows[0])
    assert torch.equal(gathered.view(torch.float32).view(rows, 2048 * world, 2).view(torch.int32),
                       reference.view(torch.int32))


def _index_selection(monkeypatch, remote, tune_available):
    """Construct the indexer's merge with stubbed components, recording its wiring."""
    recorded, registered = {}, []

    class StubMerge:
        implementation = None

        def bind(self, implementation):
            recorded.setdefault("bound", implementation)
            self.implementation = implementation

        def pack(self, *args):
            return None

        def exchange(self, *args, **kwargs):
            return None

    def selection(indices, scores, positions, rank, output, **kwargs):
        recorded.update(kwargs, indices=indices, scores=scores, output=output, rank=rank)
        return StubMerge()

    monkeypatch.setattr(indexer_module, "PreparedCandidateSelection", selection)
    monkeypatch.setattr(indexer_module, "PreparedFP8Indexer",
                        lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(indexer_module, "UnquantizedProjection",
                        lambda *args, **kwargs: SimpleNamespace())
    tuning = SimpleNamespace(register=registered.append, reduction=object())
    context = SimpleNamespace(candidates=object(), candidate_owners=object(),
                              selection_tuning=tuning if tune_available else None,
                              coordinator=None, remote=remote)
    metadata = SimpleNamespace(token_page_table=torch.zeros((1, 8), dtype=torch.int32),
                               positions=torch.arange(4, dtype=torch.int64),
                               starts=torch.zeros(1, dtype=torch.int64))
    shards = SimpleNamespace(index=torch.tensor([0], dtype=torch.int64), world=8,
                             local_tokens=lambda values: values)
    buffers = {}
    merge = PreparedIndexSelection(torch.zeros((4, 4096), dtype=torch.bfloat16),
        torch.zeros((4, 32), dtype=torch.bfloat16), torch.zeros(64 * 132, dtype=torch.uint8),
        metadata, shards, context, buffers, start=0, end=4, mode="decode", session=None,
        name="index", workspace=None, allocator=None, latent_cache=object())
    return merge, recorded, registered, context


def test_remote_index_selection_binds_the_replicated_composition_only(monkeypatch):
    merge, recorded, registered, context = _index_selection(monkeypatch, remote=True, tune_available=True)
    # The backbone runtime carries the gather; the owner exchange and its tuning are
    # never constructed, even when a tuning object happens to be present.
    assert recorded["gather_runtime"] is context.candidates
    assert recorded["owner_runtime"] is None and recorded["reduction_runtime"] is None
    assert recorded["alternatives"] == ("replicated",)
    assert recorded["bound"] == "replicated" and registered == []
    assert merge.merge.implementation == "replicated"


def test_single_host_index_selection_keeps_the_owner_tuning_path(monkeypatch):
    merge, recorded, registered, context = _index_selection(monkeypatch, remote=False, tune_available=True)
    assert recorded["gather_runtime"] is context.candidates
    assert recorded["owner_runtime"] is context.candidate_owners
    assert recorded["reduction_runtime"] is context.selection_tuning.reduction
    assert recorded["alternatives"] == ("replicated", "owner")
    assert registered == [merge.merge]


def test_fused_cutoff_flag_reads_the_environment(monkeypatch):
    monkeypatch.delenv("FLARE_CANDIDATE_MERGE_FUSED", raising=False)
    assert fused_cutoff_enabled() is False
    monkeypatch.setenv("FLARE_CANDIDATE_MERGE_FUSED", "1")
    assert fused_cutoff_enabled() is True
    monkeypatch.setenv("FLARE_CANDIDATE_MERGE_FUSED", "0")
    assert fused_cutoff_enabled() is False
    monkeypatch.setenv("FLARE_CANDIDATE_MERGE_FUSED", "")
    assert fused_cutoff_enabled() is False
