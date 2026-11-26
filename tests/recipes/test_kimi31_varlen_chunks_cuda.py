# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Drive the stage-5 varlen gather at exact chunk counts across two Spark hosts.

Run on two assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1, FLARE_TEST_RANK=0 or 1,
and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT.  The test writes controlled
byte counts into the shared counts tensor and replays one captured gather, so every
chunk count of the real records exchange runs through the identical captured graph:
the production records op gathers 150,994,944 bytes = 9,437,184 packs = 512 chunks of
18,432 packs — exactly the proxy's ROCE_VARLEN_MAX_CHUNKS boundary.  Counts are
exercised equal across ranks and skewed (one rank empty), eager and replayed; rank r's
row is filled with byte r, so parity is a one-value check per row prefix.
"""

from contextlib import suppress
from datetime import timedelta
import gc
import os
import socket
from types import SimpleNamespace
from urllib.parse import urlsplit

import faulthandler
import pytest
import torch
import torch.distributed as dist

faulthandler.dump_traceback_later(150, exit=False)

from ignition.preparation.collectives import CollectivePreparation
from ignition.preparation.compilation import compile_forward
from ignition.preparation.host_ranks import _use_interface
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3 import bootstrap
from ignition.recipes.kimi31.dcp import RECORD_BYTES, REMOTE_RECORD_ROWS, TOPK
from ignition.recipes.kimi31.indexer import SharedBuffers

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                                reason="requires two assigned Spark hosts")

CHUNK_PACKS = 18432  # the captured arg[6]: max(varlen_chunk_bytes, varlen_bytes/16/512) packs
CHUNK_BYTES = CHUNK_PACKS * 16
CAPACITY = REMOTE_RECORD_ROWS * TOPK * RECORD_BYTES  # 150,994,944 = 512 chunks exactly


def _release_device_state(objects):
    """Close runtimes and drop test state so the next test's tuning fits in shared memory.

    Both tests run in one pytest process and each builds a full remote backbone
    runtime (RoCE proxy, pinned region, staged rows, and the capacity-gated A11
    replica slab).  tuning_limits refuses to tune when fewer than its 4 GiB
    shared-memory floor is free, and a runtime left alive by the previous test
    (its proxy keeps beaconing, its allocator blocks stay reserved) is enough
    to push the layer-sequence test's tune under that floor.  Close the runtime
    explicitly - its proxy and transport are released immediately, its __del__
    is idempotent - then let the fixture's gc/empty_cache below return the
    caching allocator's blocks once the test frame is gone.
    """
    for obj in objects:
        close = getattr(obj, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


@pytest.fixture(autouse=True)
def _return_device_memory():
    """After each test frame dies, collect cycles and return device blocks."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def counts_for(chunks, empty_rank=None):
    """Per-rank byte counts carrying `chunks` full chunks on every non-empty rank."""
    count = chunks * CHUNK_BYTES
    assert count <= CAPACITY and count % 16 == 0
    world = int(os.environ.get("FLARE_TEST_WORLD", "2"))
    return [0 if r == empty_rank else count for r in range(world)]


@torch.no_grad()
def test_varlen_chunk_sweep_over_roce():
    rank = int(os.environ["FLARE_TEST_RANK"])
    world = int(os.environ.get("FLARE_TEST_WORLD", "2"))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(minutes=10))
    print(f"[rank {rank}] process group up", flush=True)
    coordinator = CollectivePreparation(dist.group.WORLD)
    assert coordinator.spans_hosts
    print(f"[rank {rank}] coordinator ready", flush=True)
    config = SimpleNamespace(prefill_chunk_size=4096, decode_tokens=1024, roce_ring=True,
                             roce_chain=False)
    model_config = SimpleNamespace(hidden=7168, vocabulary=163840,
                                   text={"num_attention_heads": 96})
    runtime = bootstrap._remote_backbone_runtime(config, dist.group.WORLD, device,
        core_storage=None, model_config=model_config, shard=SimpleNamespace(world_size=world),
        channel_rows=1024, sample_rows=256)
    print(f"[rank {rank}] runtime up (varlen_chunk_packs="
          f"{getattr(runtime, '_varlen_chunk_packs', 'n/a')})", flush=True)
    from ignition.components.collectives import PreparedVarlenGather

    graphs = []
    try:
        torch.manual_seed(97 + rank)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="kimi-varlen-chunks",
                workload_schema="kimi3", numerical_policy="exact-bytes-varlen-chunks",
                coordinator=coordinator,
                compile_workers=int(os.environ.get("FLARE_TEST_COMPILE_WORKERS", "2"))) as session:
            session.register_scratch(arena)
            storage = SharedBuffers({}, device)
            stage = storage.buffer("stage", (world, CAPACITY), torch.uint8)
            counts = storage.buffer("counts", (world,), torch.int64)
            gather = PreparedVarlenGather(stage, counts, runtime=runtime, session=session,
                coordinator=coordinator, name="chunk.sweep")
            print(f"[rank {rank}] gather declared", flush=True)
            # rank r's row holds byte r everywhere; the gather must replicate the
            # first counts[r] bytes of every row to every rank.
            for r in range(world):
                stage[r].fill_(r)
            counts.zero_()
            torch.cuda.synchronize(device)

            function = compile_forward(lambda: gather())
            print(f"[rank {rank}] compiled", flush=True)
            function()
            print(f"[rank {rank}] eager warmup done", flush=True)
            session.tune(show_progress=False)
            print(f"[rank {rank}] tuned", flush=True)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                function()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                    graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
                function()
            graphs.append(graph)
            print(f"[rank {rank}] captured", flush=True)

            def verify(tag, values):
                torch.cuda.synchronize(device)
                runtime.check_health()
                for r in range(world):
                    want = values[r]
                    if want == 0:
                        continue
                    got = stage[r][:want]
                    if not bool((got == r).all().item()):
                        bad = (got != r).nonzero().flatten()
                        first = int(bad[0].item()) if bad.numel() else -1
                        raise AssertionError(
                            f"{tag}: row {r} prefix mismatch at byte {first}"
                            f" (got {int(got[first].item())}, want {r});"
                            f" count={want} chunks={want // CHUNK_BYTES}")
                print(f"[rank {rank}] {tag}: OK", flush=True)

            def set_counts(values):
                for r, value in enumerate(values):
                    counts[r] = value
                torch.cuda.synchronize(device)

            sweep = [int(v) for v in os.environ.get(
                "FLARE_TEST_CHUNKS", "512,511,256,33,31,30,2,1").split(",")]
            # eager first: every count through the prepared (uncaptured) path
            for chunks in sweep:
                values = counts_for(chunks)
                set_counts(values)
                dist.barrier()
                function()
                verify(f"eager equal chunks={chunks}", values)
                values = counts_for(chunks, empty_rank=1)
                set_counts(values)
                dist.barrier()
                function()
                verify(f"eager skew chunks={chunks}", values)

            # replayed: one captured graph, counts rewritten between replays
            for chunks in sweep:
                values = counts_for(chunks)
                set_counts(values)
                dist.barrier()
                graph.replay()
                verify(f"replay equal chunks={chunks}", values)
            values = counts_for(sweep[0], empty_rank=1)
            set_counts(values)
            dist.barrier()
            graph.replay()
            verify(f"replay skew chunks={sweep[0]}", values)
            if rank == 0:
                print(f"varlen chunk sweep passed: {sweep} (capacity {CAPACITY},"
                      f" chunk {CHUNK_BYTES} B)")
    finally:
        for graph in graphs:
            del graph
        graphs.clear()
        held = locals().get("runtime")
        _release_device_state([held] if held is not None else [])
        del held
        dist.destroy_process_group()


@torch.no_grad()
def test_varlen_layer_sequence_over_roce():
    """One captured graph carrying a DSA layer's op mix: 32 full records parts and
    two history windows, counts rewritten on device between ops, replayed with
    varying per-op counts like a real prefill chunk."""
    rank = int(os.environ["FLARE_TEST_RANK"])
    world = int(os.environ.get("FLARE_TEST_WORLD", "2"))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(minutes=10))
    coordinator = CollectivePreparation(dist.group.WORLD)
    config = SimpleNamespace(prefill_chunk_size=4096, decode_tokens=1024, roce_ring=True,
                             roce_chain=False)
    model_config = SimpleNamespace(hidden=7168, vocabulary=163840,
                                   text={"num_attention_heads": 96})
    runtime = bootstrap._remote_backbone_runtime(config, dist.group.WORLD, device,
        core_storage=None, model_config=model_config, shard=SimpleNamespace(world_size=world),
        channel_rows=1024, sample_rows=256)
    from ignition.components.collectives import PreparedVarlenGather

    graphs = []
    try:
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="kimi-varlen-layer",
                workload_schema="kimi3", numerical_policy="exact-bytes-varlen-layer",
                coordinator=coordinator,
                compile_workers=int(os.environ.get("FLARE_TEST_COMPILE_WORKERS", "2"))) as session:
            session.register_scratch(arena)
            storage = SharedBuffers({}, device)
            stage = storage.buffer("stage", (world, CAPACITY), torch.uint8)
            counts = storage.buffer("counts", (world,), torch.int64)
            gather = PreparedVarlenGather(stage, counts, runtime=runtime, session=session,
                coordinator=coordinator, name="chunk.layer")
            # per-op count vectors, cycled through the layer: the 512-chunk boundary,
            # empty ranks, the 30-chunk history size, single chunks, a skew, a mid size.
            plan = [512, 0, 30, 1, 511, 0, 33, 2, 256, 0, 31, 512]

            def count_vector(chunks):
                # rank 1 contributes nothing on the skew op (identical on all ranks)
                empty = 1 if chunks == 511 else None
                return [0 if r == empty else chunks * CHUNK_BYTES for r in range(world)]

            cvals = [torch.tensor(count_vector(c), dtype=torch.int64, device=device)
                     for c in plan]
            # marker for row r of op i, identical on every rank
            mark_table = torch.tensor(
                [[(r + 11 * i) % 255 + 1 for r in range(world)] for i in range(len(plan))],
                dtype=torch.uint8, device=device)
            flags = torch.zeros(len(plan), dtype=torch.int32, device=device)
            counts.zero_()
            torch.cuda.synchronize(device)

            def forward():
                for i in range(len(plan)):
                    for r in range(world):
                        stage[r].copy_(mark_table[i][r])
                    counts.copy_(cvals[i])
                    gather()
                    ok = (stage[0] == mark_table[i][0]).all()
                    for r in range(1, world):
                        ok = ok & (stage[r] == mark_table[i][r]).all()
                    flags[i] = ok.to(torch.int32)
                return flags

            function = compile_forward(lambda: forward())
            function()
            session.tune(show_progress=False)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                function()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                    graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
                function()
            graphs.append(graph)
            for replay in range(4):
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)
                runtime.check_health()
                bad = (flags != 1).nonzero().flatten().tolist()
                if bad:
                    raise AssertionError(f"replay {replay}: ops with wrong parity: {bad}")
                print(f"[rank {rank}] layer replay {replay}: {len(plan)} ops OK",
                      flush=True)
            if rank == 0:
                print("varlen layer sequence passed")
    finally:
        for graph in graphs:
            del graph
        graphs.clear()
        held = locals().get("runtime")
        _release_device_state([held] if held is not None else [])
        del held
        dist.destroy_process_group()
