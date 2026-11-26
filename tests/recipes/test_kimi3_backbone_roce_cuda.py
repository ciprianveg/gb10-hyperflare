# SPDX-License-Identifier: Apache-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The Kimi TP backbone's RoCE runtime reduces and gathers across two Spark hosts.

Run the test on two assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1,
FLARE_TEST_RANK=0 or 1, and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT.
The runtime uses the Stage-1 kimi sizing (one channel for hidden reductions,
vocabulary gathers, and DSA staging) through ``_remote_backbone_runtime``.
"""

from datetime import timedelta
import os
import socket
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce
from ignition.preparation.collectives import CollectivePreparation
from ignition.preparation.compilation import compile_forward
from ignition.preparation.host_ranks import _use_interface
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3 import bootstrap

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                                reason="requires two assigned Spark hosts")

HIDDEN = 7168
VOCABULARY = 163840


@torch.no_grad()
def test_backbone_reduce_and_gather_over_roce():
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
    graphs = []
    try:
        torch.manual_seed(171 + rank)
        arena = ScratchArena(device)
        cases = []
        with RecordedPreparationSession(device=device, checkpoint="kimi-backbone-roce",
                workload_schema="kimi3", numerical_policy="default",
                coordinator=coordinator, compile_workers=4) as session:
            session.register_scratch(arena)
            for rows, gather_rows in ((1, 1), (96, 96), (1024, 256)):
                # One decode/verification reduce row block and one vocabulary gather
                # slice, both inside the Stage-1 channel's capacities: reductions run
                # up to the channel rows, gathers up to the verification sample rows.
                source = torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=device) * .1
                reduced = torch.empty_like(source)
                local = torch.randn((gather_rows, VOCABULARY // 2), dtype=torch.bfloat16, device=device) * .1
                gathered = torch.empty((gather_rows, VOCABULARY), dtype=torch.bfloat16, device=device)
                reduce = PreparedAllReduce(source, reduced, runtime=runtime,
                    session=session, coordinator=coordinator, name=f"reduce{rows}")
                gather = PreparedAllGather(local, gathered, runtime=runtime,
                    session=session, coordinator=coordinator, name=f"gather{gather_rows}")

                def forward(reduce=reduce, gather=gather, source=source, local=local):
                    reduce(source)
                    gather(local)

                cases.append((rows, gather_rows, source, local, reduced, gathered, forward))
            session.tune(show_progress=False)
            pool = torch.cuda.graph_pool_handle()
            for case in cases:
                rows, gather_rows, *_, forward = case
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

            # Alternate row blocks and compiled/uncompiled graphs, overwriting outputs.
            for iteration in range(3):
                for case, compiled, graph in (graphs if iteration % 2 == 0 else reversed(graphs)):
                    rows, gather_rows, source, local, reduced, gathered, _ = case
                    source.normal_(std=.1)
                    local.normal_(std=.1)
                    expected_reduced = source.cpu()
                    dist.all_reduce(expected_reduced)
                    expected_local = local.cpu().reshape(gather_rows, -1)
                    parts = [torch.empty_like(expected_local) for _ in range(2)]
                    dist.all_gather(parts, expected_local)
                    expected_gathered = torch.cat(parts, dim=-1).to(device)
                    reduced.fill_(float("nan"))
                    gathered.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    graph.replay()
                    torch.cuda.synchronize(device)
                    runtime.check_health()
                    torch.testing.assert_close(reduced.cpu(), expected_reduced, rtol=0, atol=0)
                    torch.testing.assert_close(gathered, expected_gathered, rtol=0, atol=0)
    finally:
        for _, _, graph in graphs:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
