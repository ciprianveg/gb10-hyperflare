# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The Kimi decode route exchange gathers and selects across two Spark hosts.

Run the test on two assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1,
FLARE_TEST_RANK=0 or 1, and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT.
The remote route exchange (stage 3) packs one row's latent and router shards into a
BF16 send row over the Stage-1 backbone runtime and selects experts with the fused
kernel's semantics afterwards; the gloo all-gather of the same shards is the reference.
The gathered latent and the selected expert ids must be bit-exact against the reference;
the route weights agree to a few ULP because the tanh-form sigmoid rounds through the
device library on GPU and libm on the reference's CPU. Rank zero writes the measured
exchange latencies to FLARE_TEST_ROUTE_TIMINGS when set.
"""

from datetime import timedelta
import json
import os
import socket
import statistics
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
from ignition.recipes.kimi3.routing import PreparedRouteGather
from tests.recipes.test_kimi3_remote_route import reference_route_select

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                                reason="requires two assigned Spark hosts")

HIDDEN = 7168
VOCABULARY = 163840
LATENT = 3584
EXPERTS = 896


@torch.no_grad()
def test_remote_route_exchange_over_roce():
    from types import SimpleNamespace

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
    graphs, timings = [], {}
    try:
        torch.manual_seed(97 + rank)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="kimi-route-roce",
                workload_schema="kimi3", numerical_policy="default",
                coordinator=coordinator, compile_workers=2) as session:
            session.register_scratch(arena)
            cases = []
            for rows in (1, 16):
                # Each rank's latent-down projection columns and its experts' router logits.
                local_down = torch.randn((rows, LATENT // 2), dtype=torch.bfloat16, device=device) * .1
                local_router = torch.randn((rows, EXPERTS // 2), device=device) * .1
                correction = torch.randn((EXPERTS,), device=device) * .3
                latent = torch.zeros((rows, LATENT), dtype=torch.bfloat16, device=device)
                weights = torch.zeros((rows, 16), device=device)
                ids = torch.zeros((rows, 16), dtype=torch.int32, device=device)
                gather = PreparedRouteGather(local_down, local_router, correction, latent,
                    weights, ids, runtime=runtime, session=session, coordinator=coordinator,
                    name=f"route{rows}", remote=True, transport={})

                def forward(gather=gather):
                    gather()

                cases.append((rows, local_down, local_router, correction, latent, weights, ids, forward))
            session.tune(show_progress=False)
            pool = torch.cuda.graph_pool_handle()
            functions = []
            for rows, local_down, local_router, correction, latent, weights, ids, forward in cases:
                for compiled in (False, True):
                    function = compile_forward(forward) if compiled else forward
                    functions.append((rows, compiled, function))
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        function()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                            graph, pool=pool, stream=stream):
                        function()
                    graphs.append((rows, compiled, local_down, local_router, correction,
                                   latent, weights, ids, graph))

            # Parity: the exchanged route equals the gloo reference, eagerly and replayed.
            for iteration in range(3):
                entries = graphs if iteration % 2 == 0 else reversed(graphs)
                for rows, compiled, local_down, local_router, correction, latent, weights, ids, graph in entries:
                    local_down.normal_(std=.1)
                    local_router.normal_(std=.1)
                    down_parts = [torch.empty_like(local_down) for _ in range(2)]
                    router_parts = [torch.empty_like(local_router) for _ in range(2)]
                    dist.all_gather(down_parts, local_down)
                    dist.all_gather(router_parts, local_router)
                    full_logits = torch.cat(router_parts, dim=-1)
                    expected_ids, expected_weights = reference_route_select(full_logits.cpu(),
                                                                            correction.cpu())
                    expected_latent = torch.cat(down_parts, dim=-1)
                    latent.fill_(float("nan"))
                    weights.fill_(float("nan"))
                    ids.fill_(-91)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    graph.replay()
                    torch.cuda.synchronize(device)
                    runtime.check_health()
                    torch.testing.assert_close(latent, expected_latent, rtol=0, atol=0)
                    assert torch.equal(ids.cpu(), expected_ids)
                    # Weights carry the same formula and reduction order, but the tanh-form
                    # sigmoid rounds through the device library on GPU and libm on CPU, so
                    # the reference is held to a few ULP instead of bit equality (the CPU
                    # parity test pins bit equality on one device).
                    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=4e-7, atol=0)

            # Two-rank exchange latency, captured graphs, slowest rank per round trip.
            for rows in (1, 16):
                forward = next(function for case_rows, _, function in functions if case_rows == rows)
                pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                          torch.cuda.Event(enable_timing=True, external=True)) for _ in range(4)]
                stream = torch.cuda.Stream(device=device)
                timing_graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                        timing_graph, pool=pool, stream=stream):
                    for begin, end in pairs:
                        begin.record()
                        forward()
                        end.record()
                graphs.append((rows, "timing", None, None, None, timing_graph))
                samples = []
                for _ in range(10):
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    timing_graph.replay()
                    torch.cuda.synchronize(device)
                    peers = [None] * 2
                    dist.all_gather_object(peers, [begin.elapsed_time(end) for begin, end in pairs])
                    samples.extend(max(values) for values in zip(*peers, strict=True))
                timings[rows] = statistics.median(samples)
    finally:
        for *_, graph in graphs:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
    if rank == 0:
        report = {str(rows): {"median_ms": value, "median_us": value * 1000}
                  for rows, value in timings.items()}
        print("route exchange latency:", json.dumps(report))
        if os.environ.get("FLARE_TEST_ROUTE_TIMINGS"):
            with open(os.environ["FLARE_TEST_ROUTE_TIMINGS"], "w") as handle:
                json.dump(report, handle, indent=2)
