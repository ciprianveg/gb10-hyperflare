# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-5 prefill exchanges merge exactly across two Spark hosts.

Run the test on two assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1,
FLARE_TEST_RANK=0 or 1, and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT.
The selected-record exchange packs each rank's share of the bit-identical global
top-2048 selection into its row of the shared staging, gathers it through the
Stage-1 backbone's variable-length channel, and scatters it into the selected
slab in original selection order; the reference is a direct copy from the
globally known concatenated pool. The windowed index history replicates each
rank's planar keys and scales byte-exactly. Both run eagerly and replayed from
captured graphs. Rank zero writes measured latencies to
FLARE_TEST_PREFILL_TIMINGS when set.
"""

from datetime import timedelta
import json
import os
import socket
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
from ignition.recipes.kimi3.config import PAGE_TOKENS
from ignition.recipes.kimi3.dcp import ContextShards
from ignition.recipes.kimi31.dcp import (INDEX_WINDOW_RECORDS, RECORD_BYTES, REMOTE_RECORD_ROWS,
    TOPK, index_window_counts, pack_index_records, pack_selected, scatter_index_records,
    scatter_selected, selection_offsets, selection_remote)
from ignition.recipes.kimi31.indexer import SharedBuffers

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                                reason="requires two assigned Spark hosts")

PAGES = 8  # latent pages per rank; index subpages are two per latent page
TABLE_COLUMNS = 8


def exchange_forward(exchange, selected, positions, tables):
    """One part's full selected-record exchange through the shared buffers."""
    rows = selected.shape[0]
    selection_remote(selected, positions, tables, exchange.owners[:rows], exchange.slots[:rows],
                     exchange.row_counts[:rows], exchange.shards, exchange.cache_pages)
    exchange.base[:rows].copy_(torch.cumsum(exchange.row_counts[:rows], 0, dtype=torch.int32)
                               .sub_(exchange.row_counts[:rows]))
    exchange.counts.copy_(exchange.row_counts[:rows].sum(0, dtype=torch.int64))
    exchange.counts.mul_(RECORD_BYTES)
    selection_offsets(exchange.owners[:rows], exchange.base[:rows], exchange.offsets[:rows])
    pack_selected(exchange.pool, exchange.owners[:rows], exchange.slots[:rows],
                  exchange.offsets[:rows], exchange.stage, exchange.shards)
    exchange.gather()
    scatter_selected(exchange.stage, exchange.owners[:rows], exchange.offsets[:rows],
                     exchange.records[:rows], exchange.shards)
    return exchange.records[:rows]


def index_forward(history, stage, gather, shards, pool, table, end):
    """The windowed index-history exchange: every window packs, gathers, scatters."""
    for first in range(0, history.windows * INDEX_WINDOW_RECORDS, INDEX_WINDOW_RECORDS):
        history.counts.copy_(index_window_counts(end, shards.world, first, INDEX_WINDOW_RECORDS))
        pack_index_records(pool, table, end, stage, shards, first)
        gather()
        scatter_index_records(stage, end, history.output, shards, first)
    return history.output


@torch.no_grad()
def test_remote_prefill_exchange_over_roce():
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
    model_config = SimpleNamespace(hidden=7168, vocabulary=163840,
                                   text={"num_attention_heads": 96})
    runtime = bootstrap._remote_backbone_runtime(config, dist.group.WORLD, device,
        core_storage=None, model_config=model_config, shard=SimpleNamespace(world_size=2),
        channel_rows=1024, sample_rows=256)
    from ignition.components.collectives import PreparedVarlenGather

    shards = ContextShards.on(rank, 2, device)
    generator = torch.Generator().manual_seed(411)
    caches = torch.randint(0, 256, (2, PAGES * PAGE_TOKENS, RECORD_BYTES),
                           generator=generator).to(torch.uint8).to(device)
    index_pools = torch.randint(0, 256, (2, 2 * PAGES, 64 * 132), generator=generator)
    index_pools = index_pools.to(torch.uint8).to(device)
    tables = torch.randint(-1, PAGES, (REMOTE_RECORD_ROWS, TABLE_COLUMNS),
                           generator=generator).int().to(device)
    graphs = []
    try:
        torch.manual_seed(97 + rank)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="kimi-prefill-roce",
                workload_schema="kimi3", numerical_policy="exact-bytes-selected-records",
                coordinator=coordinator, compile_workers=2) as session:
            session.register_scratch(arena)
            storage = SharedBuffers({}, device)
            rows = REMOTE_RECORD_ROWS
            exchange = SimpleNamespace(
                shards=shards, cache_pages=PAGES, pool=caches[rank].view(torch.int32).view(-1, RECORD_BYTES // 4),
                records=storage.buffer("records", (rows, TOPK, RECORD_BYTES), torch.uint8),
                owners=storage.buffer("owners", (rows, TOPK), torch.int32),
                slots=storage.buffer("slots", (rows, TOPK), torch.int32),
                offsets=storage.buffer("offsets", (rows, TOPK), torch.int32),
                row_counts=storage.buffer("row_counts", (rows, 2), torch.int32),
                base=storage.buffer("base", (rows, 2), torch.int32),
                counts=storage.buffer("counts", (2,), torch.int64),
                stage=storage.buffer("stage", (2, rows * TOPK * RECORD_BYTES), torch.uint8))
            exchange.gather = PreparedVarlenGather(exchange.stage, exchange.counts, runtime=runtime,
                session=session, coordinator=coordinator, name="selected.records")
            selected = storage.buffer("selected", (rows, TOPK), torch.int32)
            positions = storage.buffer("positions", (rows,), torch.int64)
            # The windowed index history: two windows over a 262,144-position context.
            maximum = 4 * INDEX_WINDOW_RECORDS
            index_pages = maximum // PAGE_TOKENS // 2
            index_storage = SharedBuffers({}, device)
            history = SimpleNamespace(
                windows=-(-maximum // (INDEX_WINDOW_RECORDS * 2)),
                counts=index_storage.buffer("index_counts", (2,), torch.int64),
                output=index_storage.buffer("index_history", (maximum // 64, 64 * 132), torch.uint8))
            i_stage = index_storage.buffer("index_stage", (2, INDEX_WINDOW_RECORDS * 132), torch.uint8)
            i_end = index_storage.buffer("index_end", (1,), torch.int64)
            i_table = index_storage.buffer("index_table", (index_pages,), torch.int32)
            i_gather = PreparedVarlenGather(i_stage, history.counts, runtime=runtime, session=session,
                coordinator=coordinator, name="index.history")
            functions = {
                "records": compile_forward(
                    lambda: exchange_forward(exchange, selected, positions, tables)),
                "index": compile_forward(
                    lambda: index_forward(history, i_stage, i_gather, shards,
                                          index_pools[rank], i_table, i_end))}
            functions["records"]()
            functions["index"]()
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

            def selection_fixture(seed):
                """Identical-on-both-ranks selections with an adversarial shape."""
                local = torch.Generator().manual_seed(seed)
                end = torch.randint(1, PAGES * PAGE_TOKENS * 2, (REMOTE_RECORD_ROWS,),
                                    generator=local)
                if seed % 4 == 2:
                    chosen = torch.full((REMOTE_RECORD_ROWS, TOPK), -1, dtype=torch.int32)
                elif seed % 4 == 3:
                    chosen = (torch.randint(0, PAGES * PAGE_TOKENS * 2, (REMOTE_RECORD_ROWS, TOPK),
                                            generator=local).int() // 2 * 2)  # all rank zero's
                else:
                    chosen = torch.randint(0, PAGES * PAGE_TOKENS * 2, (REMOTE_RECORD_ROWS, TOPK),
                                           generator=local).int()
                page_table = torch.randint(-1, PAGES, (TABLE_COLUMNS,), generator=local).int()
                index_table = torch.randint(-1, PAGES, (index_pages,), generator=local).int()
                return chosen, end, page_table, index_table

            # Parity: the scattered slab equals a direct copy from the globally known
            # pools, in original selection order, eagerly and replayed.
            for iteration in range(4):
                chosen, ends, page_table, index_table = selection_fixture(900 + iteration)
                selected.copy_(chosen.to(device))
                positions.copy_(ends.to(device))
                tables.copy_(page_table.to(device))
                i_table.copy_(index_table.to(device))
                i_end.copy_(torch.tensor([max(1, iteration * maximum // 4)], device=device,
                                         dtype=torch.int64))
                exchange.records.fill_(0x55)
                history.output.fill_(0x55)
                torch.cuda.synchronize(device)
                dist.barrier()
                for key in ("records", "index"):
                    graph_of[key].replay()
                torch.cuda.synchronize(device)
                runtime.check_health()
                # The global pool both ranks derive: gather the peer's bytes.
                peers = [None] * 2
                dist.all_gather_object(peers, caches[rank].cpu())
                global_pool = torch.stack(peers).to(device)
                token = selected.long()
                end = positions.long()
                local = token.clamp(min=0) // 2
                owner = token % 2
                block = local // PAGE_TOKENS
                live = (token >= 0) & (token <= end[:, None]) & (block < TABLE_COLUMNS)
                page = torch.where(live, tables.gather(1, block.clamp(0, TABLE_COLUMNS - 1)).long(), -1)
                live = live & (page >= 0) & (page < PAGES)
                slot = page * PAGE_TOKENS + local % PAGE_TOKENS
                expected = torch.full_like(exchange.records, 0x55)
                expected[live] = global_pool[owner.clamp(min=0)[live], slot[live]]
                assert torch.equal(exchange.records.cpu(), expected.cpu())
                # Index history: replicated planar pages hold every position's keys and
                # scale bits below the end, byte-exact, whatever pages back them.
                index_peers = [None] * 2
                dist.all_gather_object(index_peers, index_pools[rank].cpu())
                live_end = int(i_end.item())
                expected_words = torch.full((history.output.shape[0], 2112), 0x55555555,
                                            dtype=torch.int32, device=device)
                for owner_rank in range(2):
                    words = index_peers[owner_rank].view(torch.int32).view(-1).to(device)
                    table_host = i_table.cpu()
                    for local_index in range(0, -(-live_end // 2)):
                        page = int(table_host[local_index // PAGE_TOKENS])
                        if page < 0:
                            continue
                        token = local_index * 2 + owner_rank
                        if token >= live_end:
                            continue
                        subpage = page * 2 + (local_index % PAGE_TOKENS) // 64
                        source = words[subpage * 2112:(subpage + 1) * 2112]
                        row = expected_words[token // 64]
                        column = (token % 64) * 32
                        row[column:column + 32] = source[(local_index % 64) * 32:(local_index % 64) * 32 + 32]
                        row[2048 + token % 64] = source[2048 + local_index % 64]
                # Page-invalid positions below the end carry staging leftovers (the
                # scorer masks them out through the global page ids), so the parity
                # check covers exactly the positions a valid page backs.
                valid = torch.zeros(maximum, dtype=torch.bool)
                for owner_rank in range(2):
                    for local_index in range(0, -(-live_end // 2)):
                        token = local_index * 2 + owner_rank
                        if token < live_end and int(i_table.cpu()[local_index // PAGE_TOKENS]) >= 0:
                            valid[token] = True
                gone = history.output.cpu().view(torch.int32).clone()
                gw = expected_words.cpu().clone()
                void = torch.nonzero(~valid).flatten()
                rows, columns = void // 64, (void % 64) * 32
                for word in range(32):
                    gone[rows, columns + word] = 0
                    gw[rows, columns + word] = 0
                gone[rows, 2048 + void % 64] = 0
                gw[rows, 2048 + void % 64] = 0
                if not torch.equal(gone, gw):
                    bad = (history.output.cpu().view(torch.int32)
                           != expected_words.cpu()).nonzero()
                    print("INDEX MISMATCH count", bad.shape[0], "end", live_end,
                          "first bad", bad[:6].tolist())
                    row, col = bad[0].tolist()
                    print("  bad cols:", bad[:40].tolist())
                    print("  i_table[0]", int(table_host[0]), "i_end", live_end)
                    words0 = index_peers[0].view(torch.int32).view(-1)
                    print("  pool0@sub", words0[int(table_host[0]) * 2112:int(table_host[0]) * 2112 + 3].tolist())
                    import struct
                    for name, value in (("got", int(history.output.cpu().view(torch.int32)[0, 0].item())),
                                        ("exp", int(expected_words.cpu()[0, 0].item()))):
                        raw = struct.pack("<i", value)
                        for owner_rank in range(2):
                            flat = index_peers[owner_rank].view(-1)
                            hits = [p for p in (flat[:len(flat)-3] == raw[0]).nonzero().flatten().tolist()
                                    if all(int(flat[p + k]) == raw[k] for k in range(4))]
                            print(f"  {name} bytes in pool{owner_rank} at", hits[:4])
                    words0 = index_peers[0].view(torch.int32).view(-1)
                    raise AssertionError("index history parity")
                peers_out = [None] * 2
                dist.all_gather_object(peers_out, exchange.records.cpu())
                assert all(torch.equal(value, peers_out[0]) for value in peers_out)

            # Latency at the 4096-token chunk geometry: 32 parts of 128 rows per DSA
            # layer, captured, slowest rank per round trip; then one layer's two
            # index-history windows.
            timing_graphs = {}
            for key, parts in (("records", 32), ("index", 1)):
                stream = torch.cuda.Stream(device=device)
                timing_graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                        timing_graph, pool=pool, stream=stream):
                    for _ in range(parts):
                        functions[key]()
                timing_graphs[key] = timing_graph
            timings = {}
            for key in ("records", "index"):
                pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                          torch.cuda.Event(enable_timing=True, external=True)) for _ in range(8)]
                dist.barrier()
                torch.cuda.synchronize(device)
                rounds = []
                for begin, finish in pairs:
                    timing_graphs[key].replay()
                    begin.record()
                    timing_graphs[key].replay()
                    finish.record()
                    torch.cuda.synchronize(device)
                    runtime.check_health()
                    rounds.append(begin.elapsed_time(finish))
                timings[key] = max(rounds)
            if rank == 0 and os.environ.get("FLARE_TEST_PREFILL_TIMINGS"):
                with open(os.environ["FLARE_TEST_PREFILL_TIMINGS"], "w") as handle:
                    json.dump({"selected_records_chunk_layer_ms": timings["records"],
                               "index_history_layer_ms": timings["index"]}, handle)
                print(f"selected records, one 4096-token chunk layer: {timings['records']:.3f} ms")
                print(f"index history, one layer ({history.windows} windows): "
                      f"{timings['index']:.3f} ms")
    finally:
        for graph in graphs:
            del graph
        dist.destroy_process_group()


@torch.no_grad()
def test_record_replica_publish_over_roce():
    """A11: the bulk replica publish replicates every layer's records exactly.

    Each rank's pool holds its own tokens' records (identity page table), the
    publish runs over growing ranges — empty, skewed, and full — and the replicas
    must match a direct copy from the globally known pools, with rows past the
    published end left untouched.
    """
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
    config = SimpleNamespace(prefill_chunk_size=1024, decode_tokens=64, roce_ring=True,
                             roce_chain=False)
    model_config = SimpleNamespace(hidden=7168, vocabulary=163840,
                                   text={"num_attention_heads": 96})
    runtime = bootstrap._remote_backbone_runtime(config, dist.group.WORLD, device,
        core_storage=None, model_config=model_config, shard=SimpleNamespace(world_size=2),
        channel_rows=256, sample_rows=64)
    from ignition.recipes.kimi31.dcp import (PreparedRecordReplica, RECORD_BYTES,
                                             BLOCK_WORDS)

    layers, world, pages = 2, 2, 32
    capacity, chunk_records = pages * PAGE_TOKENS, pages * PAGE_TOKENS // world
    shards = ContextShards.on(rank, world, device)
    backing = torch.zeros(1 << 24, dtype=torch.uint8, device=device)
    storage = SimpleNamespace(buffer=lambda name, shape, dtype: torch.zeros(shape, dtype=dtype, device=device))
    replica = PreparedRecordReplica(layers=layers, capacity=capacity, chunk_records=chunk_records,
                                    backing=backing, shards=shards, storage=storage)
    # Per-layer pools carved from the shared backing at distinct offsets; the identity
    # page table maps local position to pool record directly.  Index pools are planar:
    # one 64-record subpage holds 2048 key words then 64 FP32 scale words (8448 bytes).
    pool_span = pages * PAGE_TOKENS * RECORD_BYTES
    subpages = pages * PAGE_TOKENS // 64
    index_span = subpages * 8448
    latent_pools, index_pools = [], []
    for layer in range(layers):
        off = layer * (pool_span + index_span) + 1024
        latent = backing[off:off + pool_span].view(torch.float8_e4m3fn).view(pages * PAGE_TOKENS, RECORD_BYTES)
        index = backing[off + pool_span:off + pool_span + index_span]
        latent_pools.append(latent)
        index_pools.append(index)
    table = torch.arange(pages, dtype=torch.int32, device=device)
    for layer in range(layers):
        replica.register(layer, latent_pools[layer], index_pools[layer], table)
    # The gathers need a preparation session; drive them through one like the other
    # exchanges in this file.
    with RecordedPreparationSession(device=device, checkpoint="kimi-replica-roce",
            workload_schema="kimi3", numerical_policy="exact-bytes-replica-publish",
            coordinator=coordinator, compile_workers=1) as session:
        replica.bind(runtime, session, coordinator, "replica.test")
        replica.latents.view(torch.uint8).fill_(0xEE)
        replica.indices.fill_(0xEE)
        if os.environ.get("FLARE_REPLICA_DEBUG_NOGATHER") == "1":
            replica.debug_no_gather = True

        def latent_bytes(token, layer):
            return torch.tensor([(token * 7 + layer * 61 + w * 3) % 251 for w in range(RECORD_BYTES)],
                                dtype=torch.uint8, device=device)

        def fill_pool_record(token, layer):
            """Write token's record into this rank's per-layer pools (planar index layout).

            Each rank's pools hold only its own tokens' records — exactly the serve
            path's pools; peers' records reach the replica only through the gather.
            """
            if token % world != rank:
                return
            local = token // world
            latent_pools[layer][local] = latent_bytes(token, layer).view(torch.float8_e4m3fn)
            subpage, slot = local // 64, local % 64
            pool = index_pools[layer]
            keys = latent_bytes(token, layer)[:128]
            pool[subpage * 8448 + slot * 128: subpage * 8448 + slot * 128 + 128] = keys
            scale = torch.tensor([(token + layer) % 251] * 4, dtype=torch.uint8, device=device)
            pool[subpage * 8448 + 8192 + slot * 4: subpage * 8448 + 8192 + slot * 4 + 4] = scale

        def want_latent_bytes(token, layer):
            return latent_bytes(token, layer)

        def want_index_bytes(token, layer):
            """The replica's planar bytes for token: keys at page slot, scale trailing."""
            page, slot = token // 64, token % 64
            keys = latent_bytes(token, layer)[:128]
            scale = torch.tensor([(token + layer) % 251] * 4, dtype=torch.uint8, device=device)
            return keys, scale

        # The pools hold the globally known values before any publish.
        for token in range(capacity):
            for layer in range(layers):
                fill_pool_record(token, layer)

        covered_end = 0
        for start, end in ((0, 0), (0, 3), (511, 513), (0, capacity)):
            meta = SimpleNamespace(starts=torch.tensor([start], device=device),
                                   lengths=torch.tensor([end - start], device=device))
            for layer in range(layers):
                replica.publish_latents(meta, layer)
                replica.publish_indexes(meta, layer)
            torch.cuda.synchronize(device)
            runtime.check_health()
            # Rows below the highest published end must match the pools exactly; rows
            # past it must still hold the 0xEE sentinel (never touched).
            for token in range(capacity):
                for layer in range(layers):
                    latent_row = replica.latents[layer][token].view(torch.uint8)
                    page, slot = token // 64, token % 64
                    index_keys = replica.indices[layer][page, slot * 128: slot * 128 + 128]
                    index_scale = replica.indices[layer][page, 8192 + slot * 4: 8192 + slot * 4 + 4]
                    want_latent, (want_keys, want_scale) = (want_latent_bytes(token, layer),
                                                            want_index_bytes(token, layer))
                    # Without the gather only this rank's own tokens' rows are filled;
                    # peers' rows inside the range read the zeroed stage (the scatter's
                    # masked write), past it they keep the sentinel.
                    if token < covered_end or (start <= token < end):
                        own = token % world == rank or not replica.debug_no_gather
                        if own:
                            assert bool((latent_row == want_latent).all()), \
                                f"latent mismatch at token {token} layer {layer} (start {start} end {end})"
                            assert bool((index_keys == want_keys).all()) and bool((index_scale == want_scale).all()), \
                                f"index mismatch at token {token} layer {layer} (start {start} end {end})"
                        else:
                            assert int(latent_row[0]) == 0, \
                                f"peer row {token} unexpected without gather"
                    else:
                        assert int(latent_row[0]) == 0xEE and int(index_keys[0]) == 0xEE, \
                            f"replica row {token} touched outside the published ranges"
            covered_end = max(covered_end, end)
            print(f"[rank {rank}] replica publish [{start}, {end}): OK", flush=True)
        if rank == 0:
            print("record replica publish passed (empty, skewed, incremental, full)")

        # The attention's local-read path end to end: global-records selection
        # addressing (owners all zero, slots = global tokens) into the replica
        # through PreparedPeerRecordGather — exactly attention.py's replica branch.
        from ignition.components.peer_record_gather import PreparedPeerRecordGather
        from ignition.recipes.kimi31.dcp import TOPK

        rows = 4
        peers = torch.tensor([replica.latents[0].data_ptr()] * world,
                             dtype=torch.int64, device=device)
        owners = torch.zeros(rows, TOPK, dtype=torch.int32, device=device)
        slots = torch.full((rows, TOPK), -1, dtype=torch.int32, device=device)
        counts = torch.zeros(rows, dtype=torch.int32, device=device)
        picks = [[0, 511, capacity - 1], [17, 300, 700], [511, 512, 3], [capacity - 1, 0, 999]]
        for r, tokens in enumerate(picks):
            counts[r] = len(tokens)
            slots[r, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=device)
        selected = torch.zeros(rows, TOPK, RECORD_BYTES, dtype=torch.float8_e4m3fn, device=device)
        gather = PreparedPeerRecordGather(peers, owners, slots, counts, selected,
            peer_records=capacity, session=session, name="replica.read")
        gather()
        torch.cuda.synchronize(device)
        runtime.check_health()
        for r, tokens in enumerate(picks):
            for column, token in enumerate(tokens):
                layer = 0
                want = latent_bytes(token, layer)
                got = selected[r, column].view(torch.uint8)
                assert bool((got == want).all()), \
                    f"replica read mismatch row {r} column {column} token {token}: " \
                    f"got {got[:4].tolist()} want {want[:4].tolist()}"
        print(f"[rank {rank}] replica local-read through PeerRecordGather: OK", flush=True)
        if rank == 0:
            print("record replica publish + local read passed")
    dist.destroy_process_group()


LATENT_TEST_WINDOW = 16  # records per rank per window: forces many windows per publish
LATENT_TEST_TOKENS = 512


def latent_bytes(token):
    return torch.tensor([(token * 7 + w * 3) % 251 for w in range(RECORD_BYTES)],
                        dtype=torch.uint8)


@torch.no_grad()
def test_windowed_latent_history_over_roce():
    """Option W: the windowed latent exchange refills the staged slab exactly.

    W is the FLARE_PREFILL_EXCHANGE=windowed alternative to the stage-5 per-part
    exchange (which stays the default); this test drives the publisher directly, so
    it exercises W regardless of the serve-time gate. Small windows force several
    varlen gathers per publish (multi-window); the end moves between replays of one
    captured graph with the counts derived in-graph (empty, skewed, incremental,
    full), and a global-position selection reads the slab through the peer-record
    gather exactly like attention.py's W branch. A captured 1M-context publish
    (eight windows per layer at world two) is timed into FLARE_TEST_PREFILL_TIMINGS
    as the W headline number.
    """
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
    config = SimpleNamespace(prefill_chunk_size=1024, decode_tokens=64, roce_ring=True,
                             roce_chain=False)
    model_config = SimpleNamespace(hidden=7168, vocabulary=163840,
                                   text={"num_attention_heads": 96})
    runtime = bootstrap._remote_backbone_runtime(config, dist.group.WORLD, device,
        core_storage=None, model_config=model_config, shard=SimpleNamespace(world_size=2),
        channel_rows=256, sample_rows=64)
    from ignition.recipes.kimi31.dcp import (PreparedRemoteLatentHistory, RECORD_BYTES,
                                             RECORD_WORDS, latent_window_counts)

    world = 2
    shards = ContextShards.on(rank, world, device)
    capacity = LATENT_TEST_TOKENS
    pages = capacity // PAGE_TOKENS
    pool = torch.zeros((pages, PAGE_TOKENS, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
    pool_words = pool.view(torch.int32).view(-1, RECORD_WORDS)
    table = torch.arange(pages, dtype=torch.int32, device=device)
    slab = torch.zeros((capacity, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
    end = torch.zeros((1,), dtype=torch.int64, device=device)
    storage = SimpleNamespace(buffer=lambda name, shape, dtype: torch.zeros(shape, dtype=dtype, device=device))
    graph = None
    big_graph = None
    try:
        with RecordedPreparationSession(device=device, checkpoint="kimi-windowed-latent-roce",
                workload_schema="kimi3", numerical_policy="exact-bytes-windowed-latent",
                coordinator=coordinator, compile_workers=1) as session:
            publisher = PreparedRemoteLatentHistory(pool, table, end, shards, slab, storage,
                window=LATENT_TEST_WINDOW, runtime=runtime, session=session,
                coordinator=coordinator, name="windowed.latent")
            # This rank's pool holds only its own tokens' records — the serve path's pools.
            for token in range(rank, capacity, world):
                pool.view(-1, RECORD_BYTES).view(torch.uint8)[token] = latent_bytes(token)
            slab.view(torch.uint8).fill_(0xEE)

            def want(token):
                return latent_bytes(token).to(device)

            def verify(end_value, note):
                torch.cuda.synchronize(device)
                runtime.check_health()
                for token in range(capacity):
                    got = slab.view(torch.uint8)[token]
                    if token < min(end_value, capacity):
                        assert bool((got == want(token)).all()), \
                            f"{note}: slab mismatch at token {token}: {got[:4].tolist()}"
                    else:
                        assert int(got[0]) == 0xEE, \
                            f"{note}: slab row {token} touched outside the published range"

            # Eager: empty, skewed, incremental, full.
            for start, stop in ((0, 0), (0, 3), (251, 257), (0, capacity)):
                slab.view(torch.uint8).fill_(0xEE)
                end.fill_(stop)
                publisher()
                verify(stop, f"eager publish [{start}, {stop})")
                print(f"[rank {rank}] windowed latent publish [{start}, {stop}): OK", flush=True)
            if rank == 0:
                print("windowed latent publish passed (empty, skewed, incremental, full)")

            # In-graph counts: capture one publish, then move the end between replays —
            # the counts derive on-device from the replicated end, so no host copy and
            # no re-tuning is involved.
            slab.view(torch.uint8).fill_(0xEE)
            end.fill_(3)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                publisher()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                    graph, stream=stream):
                publisher()
            for end_value in (0, capacity, 5, capacity // 2):
                slab.view(torch.uint8).fill_(0xEE)
                end.fill_(end_value)
                dist.barrier()
                graph.replay()
                verify(end_value, f"replayed publish end={end_value}")
                print(f"[rank {rank}] replayed windowed publish end={end_value}: OK", flush=True)
            if rank == 0:
                print("windowed latent in-graph counts passed (end moves between replays)")

            # The attention's W read path end to end: global-records selection
            # addressing (owners all zero, slots = global tokens) into the slab
            # through PreparedPeerRecordGather — exactly attention.py's W branch.
            from ignition.components.peer_record_gather import PreparedPeerRecordGather
            from ignition.recipes.kimi31.dcp import TOPK

            end.fill_(capacity)
            publisher()
            torch.cuda.synchronize(device)
            rows = 4
            peers = torch.tensor([slab.data_ptr()] * world, dtype=torch.int64, device=device)
            owners = torch.zeros(rows, TOPK, dtype=torch.int32, device=device)
            slots = torch.full((rows, TOPK), -1, dtype=torch.int32, device=device)
            counts = torch.zeros(rows, dtype=torch.int32, device=device)
            picks = [[0, 511, capacity - 1], [17, 300, 700], [511, 512, 3],
                     [capacity - 1, 0, 999]]
            for r, tokens in enumerate(picks):
                counts[r] = len(tokens)
                slots[r, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=device)
            selected = torch.zeros(rows, TOPK, RECORD_BYTES, dtype=torch.float8_e4m3fn, device=device)
            gather = PreparedPeerRecordGather(peers, owners, slots, counts, selected,
                peer_records=capacity, session=session, name="windowed.read")
            gather()
            torch.cuda.synchronize(device)
            runtime.check_health()
            for r, tokens in enumerate(picks):
                for column, token in enumerate(tokens):
                    got = selected[r, column].view(torch.uint8)
                    assert bool((got == want(token)).all()), \
                        f"slab read mismatch row {r} column {column} token {token}: " \
                        f"got {got[:4].tolist()} want {want(token)[:4].tolist()}"
            print(f"[rank {rank}] staged-slab local-read through PeerRecordGather: OK", flush=True)
            if rank == 0:
                print("windowed latent publish + local read passed")

            # The 1M-context headline: one layer's full publish at window 65536
            # (eight windows per layer at world two), captured and timed.
            big_tokens = 1 << 20
            big_pages = big_tokens // PAGE_TOKENS
            big_pool = torch.zeros((big_pages, PAGE_TOKENS, RECORD_BYTES),
                                   dtype=torch.float8_e4m3fn, device=device)
            big_table = torch.arange(big_pages, dtype=torch.int32, device=device)
            big_slab = torch.zeros((big_tokens, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
            big_end = torch.zeros((1,), dtype=torch.int64, device=device)
            big_end.fill_(big_tokens)
            big = PreparedRemoteLatentHistory(big_pool, big_table, big_end, shards, big_slab,
                storage, window=65536, runtime=runtime, session=session, coordinator=coordinator,
                name="windowed.big")
            big.gather = PreparedVarlenGather(big.stage, big.counts, runtime=runtime,
                session=session, coordinator=coordinator, name="windowed.big.gather")
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                big()
            torch.cuda.synchronize(device)
            big_graph = torch.cuda.CUDAGraph()
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                    big_graph, stream=stream):
                big()
            pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                      torch.cuda.Event(enable_timing=True, external=True)) for _ in range(8)]
            dist.barrier()
            torch.cuda.synchronize(device)
            rounds = []
            for begin, finish in pairs:
                big_graph.replay()
                begin.record()
                big_graph.replay()
                finish.record()
                torch.cuda.synchronize(device)
                runtime.check_health()
                rounds.append(begin.elapsed_time(finish))
            if rank == 0:
                latency = max(rounds)
                print(f"windowed latent publish, one layer at 1M context "
                      f"({big.windows} windows): {latency:.3f} ms")
                if os.environ.get("FLARE_TEST_PREFILL_TIMINGS"):
                    with open(os.environ["FLARE_TEST_PREFILL_TIMINGS"], "w") as handle:
                        json.dump({"windowed_latent_1M_layer_ms": latency}, handle)
    finally:
        if graph is not None:
            del graph
        if big_graph is not None:
            del big_graph
        dist.destroy_process_group()
