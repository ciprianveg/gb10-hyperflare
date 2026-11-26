# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The Kimi DSA candidate exchange merges exactly across two Spark hosts.

Run the test on two assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1,
FLARE_TEST_RANK=0 or 1, and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT.
The replicated candidate composition (stage 4) packs FP32 (score, global-id) pairs,
ships them through a byte-exact BF16 view over the Stage-1 backbone runtime, compacts
and folds locally; the integer oracle (a global single-rank ordered sort over the same
compact output) is the reference, and both ranks must agree bit-for-bit. The gathered
pairs are checked bitwise against the globally known packing. The voipmonitor #10
fused cutoff fold (FLARE_CANDIDATE_MERGE_FUSED=1) must stay integer-identical to the
default four-launch fold. Rank zero writes measured merge latencies to
FLARE_TEST_CANDIDATE_TIMINGS when set.
"""

from datetime import timedelta
import json
import os
import socket
import statistics
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist

from ignition.preparation.collectives import CollectivePreparation
from ignition.preparation.compilation import compile_forward
from ignition.preparation.host_ranks import _use_interface
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3 import bootstrap
from ignition.recipes.kimi3.dcp import ContextShards
from ignition.recipes.kimi31.dcp import TOPK
from ignition.recipes.kimi31.indexer import SharedBuffers

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                                reason="requires two assigned Spark hosts")

HIDDEN = 7168
VOCABULARY = 163840
WIDTH = TOPK * 2  # one rank's candidate pairs, packed


def build_merge(local_ids, local_scores, positions, selected, shards, runtime, session,
                coordinator, name, buffers):
    from ignition.components.candidate_selection import PreparedCandidateSelection

    merge = PreparedCandidateSelection(local_ids, local_scores, positions, shards.index,
        selected, world=shards.world, gather_runtime=runtime, owner_runtime=None,
        reduction_runtime=None, session=session, coordinator=coordinator, name=name,
        buffers=SharedBuffers(buffers, local_ids.device), alternatives=("replicated",))
    merge.bind("replicated")
    return merge


@torch.no_grad()
def test_remote_candidate_merge_over_roce():
    from tests.components.test_ordered_row_topk import adversarial_rows, ordered_reference

    rank = int(os.environ["FLARE_TEST_RANK"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(minutes=5))
    coordinator = CollectivePreparation(dist.group.WORLD)
    assert coordinator.spans_hosts
    config = SimpleNamespace(prefill_chunk_size=4096, decode_tokens=1024, roce_ring=True,
                             roce_chain=False)
    model_config = SimpleNamespace(hidden=HIDDEN, vocabulary=VOCABULARY,
                                   text={"num_attention_heads": 96})
    runtime = bootstrap._remote_backbone_runtime(config, dist.group.WORLD, device,
        core_storage=None, model_config=model_config, shard=SimpleNamespace(world_size=2),
        channel_rows=1024, sample_rows=256)
    graphs = []
    try:
        torch.manual_seed(97 + rank)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="kimi-candidates-roce",
                workload_schema="kimi3", numerical_policy="ordered-uint32-input-ties",
                coordinator=coordinator, compile_workers=2) as session:
            session.register_scratch(arena)
            cases, functions = {}, {}

            def merge_forward(merge, local_ids, local_scores, positions):
                merge.pack()
                return merge.exchange(positions=positions)

            for rows in (1, 16, 256):
                shards = ContextShards.on(rank, 2, device)
                inputs = SharedBuffers({}, device)
                local_ids = inputs.buffer("local_ids", (rows, TOPK), torch.int32)
                local_scores = inputs.buffer("local_scores", (rows, TOPK), torch.float32)
                positions = inputs.buffer("positions", (rows,), torch.int64)
                selected = inputs.buffer("selected", (rows, TOPK), torch.int32)
                merges = {False: build_merge(local_ids, local_scores, positions, selected,
                    shards, runtime, session, coordinator, f"merge{rows}", {})}
                os.environ["FLARE_CANDIDATE_MERGE_FUSED"] = "1"
                try:
                    merges[True] = build_merge(local_ids, local_scores, positions, selected,
                        shards, runtime, session, coordinator, f"merge{rows}.fused", {})
                finally:
                    del os.environ["FLARE_CANDIDATE_MERGE_FUSED"]
                cases[rows] = (merges, local_ids, local_scores, positions, selected)
                for fused, merge in merges.items():
                    functions[rows, fused] = compile_forward(
                        lambda merge=merge: merge_forward(merge, local_ids, local_scores, positions))
                    functions[rows, fused]()
            session.tune(show_progress=False)
            pool = torch.cuda.graph_pool_handle()
            graph_of = {}
            for key, function in functions.items():
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    function()
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                        graph, pool=pool, stream=stream):
                    function()
                graphs.append(graph)
                graph_of[key] = graph

            def fixture(rows, seed):
                """Global candidate rows and their per-rank local halves."""
                if seed < 3:
                    scores, lengths, ids, _ = adversarial_rows(width=WIDTH, topk=TOPK, seed=seed)
                    chosen = torch.arange(rows).add(seed).remainder(scores.shape[0])
                    scores, lengths, ids = (scores[chosen].clone(), lengths[chosen].clone(),
                                            ids[chosen].clone())
                else:
                    generator = torch.Generator().manual_seed(seed)
                    scores = torch.rand((rows, WIDTH), generator=generator)
                    ids = torch.arange(WIDTH, dtype=torch.int32).repeat(rows, 1)
                    lengths = torch.full((rows,), WIDTH, dtype=torch.int32)
                ids[torch.arange(WIDTH)[None] >= lengths[:, None]] = -1
                scores[ids < 0] = -torch.inf
                return scores, ids, lengths > 0

            # Parity: the exchanged and merged candidates equal the global ordered-sort
            # oracle on every rank, eagerly and replayed, with the fused fold identical.
            for iteration in range(4):
                for rows, (merges, local_ids, local_scores, positions, selected) in cases.items():
                    scores, ids, active = fixture(rows, iteration)
                    local_scores.copy_(scores[:, rank * TOPK:(rank + 1) * TOPK])
                    local_ids.copy_(ids[:, rank * TOPK:(rank + 1) * TOPK])
                    positions.copy_(torch.where(active, torch.arange(rows, dtype=torch.int64),
                                                torch.full((rows,), -1, dtype=torch.int64)))
                    for merge in merges.values():
                        merge.output.fill_(-91)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    for fused in (False, True):
                        graph_of[rows, fused].replay()
                    torch.cuda.synchronize(device)
                    runtime.check_health()
                    # The gather carried every rank's packed pairs, rank-major, byte-exact.
                    packed = torch.empty((rows, WIDTH, 2), dtype=torch.float32, device=device)
                    packed[:, :, 0].copy_(torch.where(ids >= 0, scores, -torch.inf))
                    owner = torch.arange(WIDTH, dtype=torch.int32).div(TOPK, rounding_mode="floor")
                    packed.view(torch.int32)[:, :, 1].copy_(
                        torch.where(ids >= 0, ids * 2 + owner, -1))
                    for merge in merges.values():
                        assert torch.equal(merge.gathered.view(torch.int32).cpu(),
                                           packed.view(torch.int32).cpu())
                        # The oracle is a global single-rank ordered sort over the same
                        # compact output this rank folded.
                        expected_values, expected_ids, _ = ordered_reference(
                            merge.values.cpu(), merge.lengths.cpu(), merge.global_ids.cpu(), TOPK)
                        assert torch.equal(merge.output.cpu(), expected_ids)
                        assert torch.equal(merge.selected_scores.view(torch.int32).cpu(),
                                           expected_values)
                    assert torch.equal(merges[False].output.cpu(), merges[True].output.cpu())
                    peers = [None] * 2
                    dist.all_gather_object(peers, merges[False].output.cpu())
                    assert all(torch.equal(value, peers[0]) for value in peers)

            # Two-rank merge latency at the packed-verification geometry, captured graphs,
            # slowest rank per round trip: the default fold, then the fused fold.
            timings = {}
            for rows in (1, 256):
                for fused in (False, True):
                    function = functions[rows, fused]
                    pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                              torch.cuda.Event(enable_timing=True, external=True)) for _ in range(4)]
                    stream = torch.cuda.Stream(device=device)
                    timing_graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                            timing_graph, pool=pool, stream=stream):
                        for begin, end in pairs:
                            begin.record()
                            function()
                            end.record()
                    graphs.append(timing_graph)
                    samples = []
                    for _ in range(10):
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        timing_graph.replay()
                        torch.cuda.synchronize(device)
                        peers = [None] * 2
                        dist.all_gather_object(peers, [begin.elapsed_time(end) for begin, end in pairs])
                        samples.extend(max(values) for values in zip(*peers, strict=True))
                    timings[rows, fused] = statistics.median(samples)
    finally:
        for graph in graphs:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
    if rank == 0:
        report = {f"rows_{rows}.{'fused' if fused else 'default'}":
                  {"median_ms": value, "median_us": value * 1000}
                  for (rows, fused), value in timings.items()}
        print("candidate merge latency:", json.dumps(report))
        if os.environ.get("FLARE_TEST_CANDIDATE_TIMINGS"):
            with open(os.environ["FLARE_TEST_CANDIDATE_TIMINGS"], "w") as handle:
                json.dump(report, handle, indent=2)
