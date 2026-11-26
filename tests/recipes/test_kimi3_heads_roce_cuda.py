# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The Kimi head exchanges ride one RoCEnante backbone across two Spark hosts (stage 2).

Run the test on two assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1,
FLARE_TEST_RANK=0 or 1, and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT.
The runtime uses the Stage-1 kimi sizing through ``_remote_backbone_runtime``; the
exercises are (a) the FP8-as-BF16 transport view's byte-exact head gather, (b) the
sliced LSE pack/gather/merge against a single-rank reference, and (c) a captured
CUDA-graph replay of both exchanges composed on the shared channel.
"""

from datetime import timedelta
import os
import socket
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist

from ignition.components.collectives import PreparedAllGather
from ignition.components.head_exchange import pack_width
from ignition.preparation.collectives import CollectivePreparation
from ignition.preparation.compilation import compile_forward
from ignition.preparation.host_ranks import _use_interface
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3 import bootstrap
from ignition.recipes.kimi3.attention import SlicedLseCombine
from ignition.recipes.kimi3.dcp import LSE_SLICE_ROWS, RECORD_BYTES

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                                reason="requires two assigned Spark hosts")

HIDDEN = 7168
VOCABULARY = 163840
HEADS = 96


def _expected_gathered(local, world):
    """Bit-exact rank-order column gather of E4M3 bytes, through the control group."""
    raw = local.view(torch.uint8).cpu()
    parts = [torch.empty_like(raw) for _ in range(world)]
    dist.all_gather(parts, raw)
    return torch.cat(parts, dim=1).to(local.device)


def _expected_combined(partials, lses, world):
    """Single-rank LSE combination of every rank's head shard, in FP32 like the merge."""
    rows, heads, dim = partials[0].shape
    local = heads // world
    stacked = torch.stack([partial.float() for partial in partials])   # [world, rows, heads, dim]
    scores = torch.stack([lse for lse in lses])                        # [world, rows, heads]
    peak = scores.max(0).values
    base = torch.where(peak > float("-inf"), peak, torch.zeros_like(peak))
    weights = torch.exp(scores - base)
    total = weights.sum(0).clamp(min=1e-30)
    combined = (weights.unsqueeze(-1) * stacked).sum(0) / total.unsqueeze(-1)
    return torch.where(total.unsqueeze(-1) > 0, combined, 0.).bfloat16()  # [rows, heads, dim]


@torch.no_grad()
def test_head_exchanges_over_roce():
    from types import SimpleNamespace

    rank = int(os.environ["FLARE_TEST_RANK"])
    world = 2
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(minutes=5))
    coordinator = CollectivePreparation(dist.group.WORLD)
    assert coordinator.spans_hosts
    config = SimpleNamespace(prefill_chunk_size=4096, decode_tokens=1024, roce_ring=True,
                             roce_chain=False)
    model_config = SimpleNamespace(hidden=HIDDEN, vocabulary=VOCABULARY,
                                   text={"num_attention_heads": HEADS})
    runtime = bootstrap._remote_backbone_runtime(config, dist.group.WORLD, device,
        core_storage=None, model_config=model_config, shard=SimpleNamespace(world_size=world),
        channel_rows=1024, sample_rows=256)
    graphs = []
    try:
        torch.manual_seed(171 + rank)
        heads_per_rank = HEADS // world
        lse_width = pack_width(world, HEADS, 512)
        arena = ScratchArena(device)
        # One shared staging pair serves every LSE slice, exactly as the prepared layers reuse it.
        lse_send = torch.zeros((LSE_SLICE_ROWS, lse_width), dtype=torch.bfloat16, device=device)
        lse_received = torch.zeros((LSE_SLICE_ROWS, world * lse_width), dtype=torch.bfloat16, device=device)
        rank_tensor = torch.tensor([rank], dtype=torch.int64, device=device)
        cases = []
        with RecordedPreparationSession(device=device, checkpoint="kimi-heads-roce",
                workload_schema="kimi3", numerical_policy="default",
                coordinator=coordinator, compile_workers=4) as session:
            session.register_scratch(arena)
            for rows in (1, 96, 200):
                # (a) E4M3 absorbed-query head gather through the byte-exact BF16 transport
                # view: random bytes, including NaN and infinity lanes, must survive intact.
                quantized = torch.randint(0, 256, (rows, heads_per_rank * RECORD_BYTES),
                    dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
                gathered = torch.empty((rows, HEADS * RECORD_BYTES), dtype=torch.float8_e4m3fn,
                                       device=device)
                gather = PreparedAllGather(quantized, gathered, runtime=runtime,
                    session=session, coordinator=coordinator, name=f"gather_heads{rows}")
                # (b) the sliced LSE combine: rows beyond the slice bound exercise staging
                # reuse across slices; one row per case leaves every rank's LSE at -inf.
                partial = torch.randn((rows, HEADS, 512), dtype=torch.bfloat16, device=device) * .25
                lse = torch.randn((rows, HEADS), dtype=torch.float32, device=device)
                if rows > 1:
                    lse[-1] = float("-inf")
                combined = torch.empty((rows, HEADS // world, 512), dtype=torch.bfloat16, device=device)
                combine = SlicedLseCombine(partial, lse, combined, runtime=runtime,
                    session=session, coordinator=coordinator, name=f"combine{rows}",
                    rank=rank_tensor, send=lse_send, received=lse_received)

                def forward(gather=gather, combine=combine, quantized=quantized,
                            partial=partial, lse=lse):
                    # A serial head-exchange composition on the one ordered channel.
                    gather(quantized)
                    combine(partial, lse)

                cases.append((rows, quantized, gathered, partial, lse, combined, forward))
            session.tune(show_progress=False)
            pool = torch.cuda.graph_pool_handle()
            for case in cases:
                rows, *_, forward = case
                for compiled in (False, True):
                    function = compile_forward(forward) if compiled else forward
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        function()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                            graph, pool=pool, stream=stream):
                        function()
                    graphs.append((case, compiled, graph))

            # Alternate row counts and compiled/uncompiled graphs, overwriting every output.
            for iteration in range(3):
                for case, compiled, graph in (graphs if iteration % 2 == 0 else reversed(graphs)):
                    rows, quantized, gathered, partial, lse, combined, _ = case
                    quantized.copy_(torch.randint(0, 256, quantized.shape, dtype=torch.uint8,
                        device=device).view(torch.float8_e4m3fn))
                    partial.normal_(std=.25)
                    lse.normal_()
                    if rows > 1:
                        lse[-1] = float("-inf")
                    expected_gathered = _expected_gathered(quantized, world)
                    raw_partial = partial.view(torch.uint8).cpu()
                    gathered_partial = [torch.empty_like(raw_partial) for _ in range(world)]
                    dist.all_gather(gathered_partial, raw_partial)
                    peer_partial = [part.to(device).view(torch.float8_e4m3fn).view(torch.bfloat16)
                                    for part in gathered_partial]
                    gathered_lse = [torch.empty_like(lse, device="cpu") for _ in range(world)]
                    dist.all_gather(gathered_lse, lse.cpu())
                    expected_combined = _expected_combined(
                        peer_partial, [s.to(device) for s in gathered_lse], world)
                    gathered.view(torch.uint8).fill_(0xFF)
                    combined.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    graph.replay()
                    torch.cuda.synchronize(device)
                    runtime.check_health()
                    # FP8 transport is byte-exact; the merge rounds once from FP32 weights.
                    assert torch.equal(gathered.view(torch.uint8), expected_gathered)
                    shard = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
                    torch.testing.assert_close(combined, expected_combined[:, shard],
                                               rtol=1e-2, atol=1e-2)
                    if rows > 1:
                        # The row no rank attended anything for combines to zero.
                        assert not combined[-1].count_nonzero()
    finally:
        for _, _, graph in graphs:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
