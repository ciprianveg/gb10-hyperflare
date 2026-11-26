"""Released-fleet-only TP16/DCP8 selected-attention check; --check is CPU-only.

Uses the existing TP16 sparse-attention oracle and its rtol=.03/atol=.002.
This tests target geometry, not model weights, export, generation or throughput.
"""
import argparse
import json
import os
from pathlib import Path


TP = 16
DCP = 8
LOCAL_HEADS = 6
PAGE = 128
TOPK = 2048
MAX_VISIBLE = 65536


def geometry(rank):
    if type(rank) is not int or not 0 <= rank < TP:
        raise ValueError("global rank must be in TP16")
    first = rank // DCP * DCP
    return dict(rank=rank, local_rank=rank % DCP,
                members=list(range(first, first + DCP)),
                global_heads=list(range(rank * LOCAL_HEADS, (rank + 1) * LOCAL_HEADS)),
                group_heads=list(range(first * LOCAL_HEADS, (first + DCP) * LOCAL_HEADS)),
                page_tokens=PAGE * DCP)


def check_geometry():
    rows = [geometry(rank) for rank in range(TP)]
    assert [h for row in rows for h in row["global_heads"]] == list(range(96))
    assert rows[0]["group_heads"] == list(range(48))
    assert rows[8]["group_heads"] == list(range(48, 96))
    for row in rows:
        offset = row["local_rank"] * LOCAL_HEADS
        assert row["group_heads"][offset:offset + LOCAL_HEADS] == row["global_heads"]
        for token in (0, 7, 8, 1023, 1024, 1025, MAX_VISIBLE - 1):
            local = token // DCP
            assert (local // PAGE * PAGE + local % PAGE) * DCP + token % DCP == token
    return rows


def run(args):
    if not args.exclusive_fleet_confirmed or os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        raise RuntimeError("separate parent fleet/GPU release required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("one explicitly assigned host-local GPU 0 is required")
    layout = geometry(args.rank)
    from fixture_owners import FixtureOwners
    from multi_roce_fixture import candidate_import, initialize_gloo
    extension = candidate_import(args.extension)
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch
    import torch.distributed as dist
    from b12x.attention import dense_mla
    from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, PreparedBroadcast
    from ignition.components.dense_mla import PreparedDenseMLA
    from ignition.components.head_exchange import pack_width
    from ignition.components.row_topk import PreparedRowTopK
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.attention import SlicedLseCombine
    from ignition.recipes.kimi3.bootstrap import prepare_context_group
    from ignition.recipes.kimi31.dcp import ContextShards, pack_candidates, selection_slots
    from tests.recipes.test_kimi31_dsa_tp16_cuda import _records, _scores, RECORD_BYTES

    torch.set_num_threads(1)
    initialize_gloo(args, dist)
    group, local_rank, members = prepare_context_group(
        dist.group.WORLD, rank=args.rank, world=TP, size=DCP)
    assert local_rank == layout["local_rank"] and list(members) == layout["members"]
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    context, _ = extension.current_context()
    graphs, sessions, runtimes = {}, {}, {}
    with FixtureOwners(lambda owners: extension.export_synchronize(context, owners)) as ownership:
        ownership.hold((graphs, sessions, runtimes, group, dist.group.WORLD))
        ownership.after_drain(lambda: [g.reset() for g in graphs.values()])
        ownership.after_drain(lambda: [s.close() for s in sessions.values()])
        ownership.after_drain(lambda: [r.close() for r in runtimes.values()])
        ownership.after_contexts(dist.destroy_process_group)
        images = ownership.enter(ModuleImages())
        storage = ownership.enter(CoreStorage(rank=args.rank, world=TP, device=0, local_ranks=1))
        try:
            rows, heads = 2, LOCAL_HEADS * DCP
            coordinators = dict(tp=CollectivePreparation(dist.group.WORLD), dcp=CollectivePreparation(group))
            lse_width = pack_width(DCP, heads, 512)
            for name, process_group, membership in (
                ("tp", dist.group.WORLD, tuple(range(TP))), ("dcp", group, members),
            ):
                runtimes[name] = create_roce_runtime(
                    process_group, device, max_tokens=rows, pipe_tokens=rows, hidden_size=256,
                    gather_bytes=max(rows * TOPK * 8, rows * lse_width * 2),
                    varlen_bytes=0, core_storage=storage,
                    communicator_id="tp16" if name == "tp" else f"dcp8-{members[0]}",
                    members=list(membership))
                sessions[name] = RecordedPreparationSession(
                    device=device, checkpoint="kimi31-tp16-dcp8-selected-attention",
                    workload_schema="global96-local6-group48", numerical_policy="default",
                    autotune_effort="off", compile_workers=1, coordinator=coordinators[name])
            logical_pages = MAX_VISIBLE // (PAGE * DCP)
            physical_pages = logical_pages + 3
            cache = torch.zeros((physical_pages, PAGE, RECORD_BYTES), device=device, dtype=torch.float8_e4m3fn)
            page_map = ((torch.arange(logical_pages, device=device) * 2 + 3) % physical_pages).int()
            tables = page_map[None].repeat(rows, 1)
            positions = torch.full((rows,), -1, device=device, dtype=torch.int64)
            local_ids = torch.full((rows, TOPK), -1, device=device, dtype=torch.int32)
            local_scores = torch.full((rows, TOPK), -torch.inf, device=device)
            packed = torch.zeros((rows, TOPK, 2), device=device)
            exchanged = torch.zeros((rows, DCP * TOPK, 2), device=device)
            scores = torch.empty((rows, DCP * TOPK), device=device)
            ids = torch.empty_like(scores, dtype=torch.int32)
            lengths = torch.zeros(rows, device=device, dtype=torch.int32)
            selected_scores = torch.empty((rows, TOPK), device=device)
            selected = torch.empty_like(selected_scores, dtype=torch.int32)
            owners, slots = torch.empty_like(selected), torch.empty_like(selected)
            counts = torch.zeros(rows, device=device, dtype=torch.int32)
            head_ids = torch.tensor(layout["global_heads"], device=device)
            channels = torch.arange(RECORD_BYTES, device=device)
            query = (((head_ids[:, None] * 11 + channels[None] * 7).remainder(127) - 63).float() / 512)
            query = query.bfloat16().to(cache.dtype)[None].repeat(rows, 1, 1)
            gathered = torch.zeros((rows, heads, RECORD_BYTES), device=device, dtype=query.dtype)
            partial = torch.zeros((rows, heads, 512), device=device, dtype=torch.bfloat16)
            lse = torch.full((rows, heads), -torch.inf, device=device)
            output = torch.zeros((rows, LOCAL_HEADS, 512), device=device, dtype=torch.bfloat16)
            starts = torch.arange(rows + 1, device=device, dtype=torch.int32)
            shards = ContextShards.on(local_rank, DCP, device)
            arena = ScratchArena(device)
            sessions["tp"].register_scratch(arena)
            tp_kwargs = dict(runtime=runtimes["tp"], session=sessions["tp"], coordinator=coordinators["tp"])
            cp_kwargs = dict(runtime=runtimes["dcp"], session=sessions["dcp"], coordinator=coordinators["dcp"])
            marker = torch.full((rows, 256), args.rank, device=device, dtype=torch.bfloat16)
            reduced = torch.zeros_like(marker)
            reduction = PreparedAllReduce(marker, reduced, name="tp_marker", **tp_kwargs)
            exchange = PreparedAllGather(packed.view(torch.bfloat16).view(rows, -1),
                exchanged.view(torch.bfloat16).view(rows, -1), name="candidates", **cp_kwargs)
            fold = PreparedRowTopK(scores, lengths, ids, selected_scores, selected,
                session=sessions["tp"], name="global_top2048")
            consensus = PreparedBroadcast((selected,), name="selected_consensus", **cp_kwargs)
            gather = PreparedAllGather(query.view(rows, -1), gathered.view(rows, -1),
                name="query_heads", **cp_kwargs)
            caps = dense_mla.Caps(device=device, mode="decode", q_dtype=query.dtype, kv_dtype=cache.dtype,
                num_q_heads=heads, page_size=1, max_total_q=rows, max_batch=rows,
                max_cache_tokens=TOPK, max_page_table_width=TOPK,
                num_cache_pages=physical_pages * PAGE, partial_dtype=torch.float32, use_cuda_graph=True)
            attention = PreparedDenseMLA(gathered, cache.view(-1, 1, RECORD_BYTES), partial,
                slots, counts, starts, caps=caps, scale=192 ** -.5, session=sessions["tp"],
                name="local_selected_attention", workspace=arena, lse=lse)
            combine = SlicedLseCombine(partial, lse, output, name="combine_selected_attention",
                rank=shards.index, send=torch.zeros((rows, lse_width), device=device, dtype=torch.bfloat16),
                received=torch.zeros((rows, DCP * lse_width), device=device, dtype=torch.bfloat16), **cp_kwargs)
            sessions["tp"].tune(show_progress=False)
            dist.barrier()
            sessions["dcp"].tune(show_progress=False)
            dist.barrier()
            local_positions = torch.arange(MAX_VISIBLE // DCP, device=device, dtype=torch.int64)
            physical_slots = page_map[local_positions // PAGE].long() * PAGE + local_positions % PAGE
            cache.view(-1, RECORD_BYTES)[physical_slots] = _records(local_positions * DCP + local_rank).to(cache.dtype)
            saved_cache = cache.view(torch.uint8).clone()

            def forward():
                reduction(marker)
                pack_candidates(local_ids, local_scores, packed, shards)
                exchange(packed.view(torch.bfloat16).view(rows, -1))
                scores.copy_(exchanged[:, :, 0])
                ids.copy_(exchanged.view(torch.int32)[:, :, 1])
                lengths.copy_(torch.where(positions >= 0, DCP * TOPK, 0).int())
                fold()
                consensus()
                selection_slots(selected, positions, tables, owners, slots, counts, shards, physical_pages, local=True)
                gather(query.view(rows, -1))
                attention(gathered, partial)
                combine(partial, lse)

            stream = torch.cuda.Stream(device=device)
            captured, _ = capture_forwards({"decode": forward}, {"decode": GraphSignature("decode", rows, 1)},
                session=sessions["tp"], device=device, stream=stream, pool=storage.graph_pool.id,
                collective=runtimes["tp"], coordinator=coordinators["tp"], compile_decoder=False,
                additional_sessions=(sessions["dcp"],), additional_collectives=(runtimes["dcp"],))
            graphs.update(captured)
            for session in sessions.values():
                session.freeze()
            for visible, concentrated in ((17, False), (1025, False), (MAX_VISIBLE, True)):
                global_ids = torch.arange(local_rank, visible, DCP)
                candidate_scores = _scores(global_ids, concentrated)
                count = min(TOPK, len(global_ids))
                values, indices = candidate_scores.topk(count)
                local_ids.fill_(-1)
                local_scores.fill_(-torch.inf)
                local_ids[1, :count].copy_(indices.to(device, dtype=torch.int32))
                local_scores[1, :count].copy_(values.to(device))
                positions[1] = visible - 1
                for repetition in range(3):
                    arena.tensor.fill_(255)
                    output.fill_(17)
                    partial.fill_(torch.nan)
                    lse.fill_(torch.nan)
                    selected.fill_(-2)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)
                    graphs["decode"].replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for field in ("allocation.all.allocated", "allocation.all.freed",
                                  "segment.all.allocated", "segment.all.freed"):
                        assert before[field] == after[field], field
                    all_selected = [None] * TP
                    dist.all_gather_object(all_selected, selected.cpu().tolist())
                    expected_count = min(visible, TOPK)
                    # Tied selections need consensus inside a DCP group, not
                    # identical tie-breaking between independent groups.
                    selections = all_selected[members[0]:members[-1] + 1]
                    sets = []
                    for peer in selections:
                        assert all(token == -1 for token in peer[0])
                        valid = [token for token in peer[1] if token >= 0]
                        assert len(valid) == len(set(valid)) == expected_count
                        assert all(token < visible for token in valid)
                        sets.append(set(valid))
                    assert all(s == sets[0] for s in sets)
                    chosen = torch.tensor(sorted(sets[0]), dtype=torch.int64)
                    all_scores = _scores(torch.arange(visible), concentrated)
                    cutoff = all_scores.topk(expected_count).values[-1]
                    assert (all_scores[chosen] >= cutoff).all()
                    assert set(torch.where(all_scores > cutoff)[0].tolist()) <= sets[0]
                    assert counts[0].item() == 0
                    assert counts[1].item() == sum(token % DCP == local_rank for token in sets[0])
                    records = _records(chosen.to(device)).to(cache.dtype).float()
                    expected = (query[1].float() @ records.T * 192 ** -.5).softmax(-1) @ records[:, :512]
                    torch.testing.assert_close(output[1].float(), expected, rtol=.03, atol=.002)
                    assert not output[0].count_nonzero() and torch.isfinite(output).all()
                    assert torch.equal(cache.view(torch.uint8), saved_cache)
                    assert bool((reduced == sum(range(TP))).all())
                    print(json.dumps(dict(**layout, visible=visible, concentrated=concentrated,
                        repetition=repetition, parity=True, inactive_exact=True, cache_preserved=True)), flush=True)
                    dist.barrier()
        finally:
            ownership.hold(dict(locals()))
    assert ownership.drained
    print(json.dumps(dict(rank=args.rank, drained=True, cases=3, repetitions=3)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--rendezvous")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--exclusive-fleet-confirmed", action="store_true")
    args = parser.parse_args()
    layouts = check_geometry()
    if args.check:
        print(json.dumps(dict(geometry=layouts, gpu_execution=False)))
        return
    if args.rank is None or args.rendezvous is None or args.extension is None:
        parser.error("--rank, --rendezvous and --extension are required")
    run(args)


if __name__ == "__main__":
    main()
