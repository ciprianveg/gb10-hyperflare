# SPDX-License-Identifier: Apache-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 multi-host preparation gates, weight-read routing, and plan dispatch."""

from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.recipes.kimi3 import bootstrap, recipe
from ignition.recipes.declaration import Speculation


def test_eight_hosts_may_each_use_their_own_gpu_zero():
    config = bootstrap.PreparationConfig(devices=(0,) * 8, hosts=tuple(f"h{i}" for i in range(8)),
                                         roce_ring=True)
    assert config.devices == (0,) * 8
    with pytest.raises(ValueError, match="placements"):
        bootstrap.PreparationConfig(devices=(0,) * 8)
    with pytest.raises(ValueError, match="placements"):
        bootstrap.PreparationConfig(devices=(0,) * 8, hosts=("hertz",) * 8)


@pytest.mark.parametrize("remote", [False, True])
def test_recipe_dispatches_the_complete_plan_to_the_rank_group(monkeypatch, remote, tmp_path):
    from ignition.preparation import rank_group
    from ignition.recipes.kimi31 import recipe as kimi31

    gpus = tuple(planning.Gpu(i, "GB10", f"GPU-{i}", 1 << 40, 1 << 40, (12, 1)) for i in range(8))
    plan = planning.fit(kimi31.RECIPE, kimi31.RECIPE.checkpoints[0], gpus, concurrency=32,
        reserve_bytes=0, speculation=Speculation(draft_tokens=7, adaptive=True, token_capacity=128,
            packed_verification=True, graph_budget=64, method="dflash2"),
        roce_ring=remote, context_parallel=True, language_model_only=True)
    assert plan.world_size == 8 and plan.roce_ring == remote and not plan.roce_chain
    # The draft resolves inside the snapshot (site 7): peers fetch it through the source.
    snapshot = tmp_path / "checkpoint"
    (snapshot / "dflash2").mkdir(parents=True)
    ranks = SimpleNamespace(hosts=tuple(f"h{i}" for i in range(8))) if remote else None
    calls, result, storage = [], object(), object()
    monkeypatch.setattr(rank_group, "prepare_group", lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    assert kimi31.prepare(str(snapshot), plan, compile_forwards=True, kernel_cache=None, progress=None,
                          core_storage=storage, ranks=ranks) is result
    (snapshot_arg, config, forward), kwargs = calls[0]
    assert snapshot_arg == str(snapshot) and forward is bootstrap.prepare_rank
    assert config.devices == ((0,) * 8 if remote else tuple(range(8)))
    assert config.hosts == (ranks.hosts if remote else None)
    assert config.roce_ring == remote and not config.roce_chain
    assert config.minimum_pages == plan.minimum_pages
    assert config.context == plan.context and config.context_parallel
    assert config.draft_checkpoint == str(snapshot / "dflash2")
    assert config.prefill_chunk_size == plan.prefill_chunk
    assert kwargs["ranks"] is ranks and kwargs["checkpoint"] is plan.checkpoint
    assert kwargs["core_storage"] is storage


def test_multi_host_gate_reports_no_unported_exchange_classes(monkeypatch):
    coordinator = SimpleNamespace(spans_hosts=True)
    # Stage 5 ported the two prefill classes, so the gate is empty and never raises.
    bootstrap._require_ported_exchanges(coordinator)
    bootstrap._require_ported_exchanges(None)
    bootstrap._require_ported_exchanges(SimpleNamespace(spans_hosts=False))
    # The mechanism stays: gating any class again brings its report back.
    monkeypatch.setitem(bootstrap.REMOTE_EXCHANGES_PORTED, "shared prefill staging", False)
    monkeypatch.setitem(bootstrap.REMOTE_EXCHANGES_PORTED, "IPC continuation-page peers", False)
    with pytest.raises(NotImplementedError) as info:
        bootstrap._require_ported_exchanges(coordinator)
    message = str(info.value)
    assert "shared prefill staging" in message and "IPC continuation-page peers" in message
    assert "port stage 5" in message
    monkeypatch.setitem(bootstrap.REMOTE_EXCHANGES_PORTED, "shared prefill staging", True)
    with pytest.raises(NotImplementedError) as info:
        bootstrap._require_ported_exchanges(coordinator)
    shrunk = str(info.value)
    assert "shared prefill staging" not in shrunk and "IPC continuation-page peers" in shrunk


def test_weight_loading_skips_cross_host_ipc_column_reads():
    group, storage = object(), SimpleNamespace(peer_storage=object())
    # Ranks on one host keep staging column reads through CUDA IPC.
    assert bootstrap._shared_column_peers(group, storage, ("hertz",) * 8) == (group, storage.peer_storage)
    assert bootstrap._shared_column_peers(group, storage, None) == (group, storage.peer_storage)
    # Ranks on separate hosts cannot open each other's IPC handles.
    assert bootstrap._shared_column_peers(group, storage, tuple(f"h{i}" for i in range(8))) is None
    assert bootstrap._shared_column_peers(None, storage, None) is None
    assert bootstrap._shared_column_peers(group, SimpleNamespace(peer_storage=None), None) is None


@pytest.mark.parametrize("world,heads,hidden", [(8, 96, 7168), (16, 96, 7168), (16, 32, 8), (16, 96, 12)])
def test_sparse_attention_accepts_eight_or_sixteen_context_ranks(world, heads, hidden):
    from ignition.recipes.kimi31.attention import validate_context_ranks

    # Sub-rank synthetic widths skip divisibility; preparation asserts the real config.
    validate_context_ranks(world, heads, hidden)


@pytest.mark.parametrize("world,heads,hidden", [(4, 96, 7168), (2, 96, 7168), (8, 95, 7168), (8, 96, 7164)])
def test_sparse_attention_rejects_undividable_context_ranks(world, heads, hidden):
    from ignition.recipes.kimi31.attention import validate_context_ranks

    with pytest.raises(ValueError, match="context ranks"):
        validate_context_ranks(world, heads, hidden)


def test_remote_backbone_runtime_sizes_one_roce_channel(monkeypatch):
    config = bootstrap.PreparationConfig(devices=(0,) * 8, hosts=tuple(f"h{i}" for i in range(8)),
        roce_ring=True, max_active=32, speculative_widths=tuple(range(1, 9)),
        speculative_token_capacity=128)
    calls, runtime, storage = [], object(), object()
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
        lambda group, device, **kwargs: calls.append(kwargs) or runtime)
    model_config = SimpleNamespace(hidden=7168, vocabulary=163840, text={"num_attention_heads": 96}, draft=None)
    shard = SimpleNamespace(world_size=8)
    assert bootstrap._remote_backbone_runtime(config, None, None, core_storage=storage,
        model_config=model_config, shard=shard, channel_rows=1024, sample_rows=256, slots=32) is runtime
    only = calls[0]
    assert only["core_storage"] is storage and only["ring"] and not only["ring_open"]
    # Decode and verification rows reduce one-shot; prefill chunks pipeline two-shot.
    # The one-shot window plans the speculative round's physical rows in from
    # construction (glm53's sample capacity): the dflash2 draft block runs every
    # configured position per request, so the window grows past the reduce rows when
    # the drafted envelope demands it and never shrinks below them.
    assert only["max_tokens"] == 1024
    assert only["pipe_tokens"] == max(config.prefill_chunk_size, config.decode_tokens)
    assert only["hidden_size"] == 7168
    # One 16 MiB gather step covers every per-rank contribution the port stages move
    # onto the runtime: LSE slices (128 x 43,176 x 2 ~ 11 MiB), vocabulary partitions
    # (256 x 20,480 x 2 ~ 10.5 MiB), candidates and selections (4 MiB each).
    assert only["gather_bytes"] == 16 << 20
    # Stage 5's varlen exchanges: one 128-row part's selected records in the worst
    # case (every selection owned by one rank) and one 65,536-position index window.
    assert only["varlen_bytes"] == max(128 * 2048 * 576, 65536 * 132)
    # Larger configs grow the gather channel in 16 MiB steps, never below a term, and
    # the one-shot window covers the sample rows.
    terms = (128 * 43176 * 2, 2000 * (163840 // 8) * 2, 128 * 8192 * 2, 512 * 4096 * 2)
    assert only["gather_bytes"] >= max(terms[:1] + terms[2:])
    sized = bootstrap._remote_backbone_runtime(config, None, None, core_storage=storage,
        model_config=model_config, shard=shard, channel_rows=1024, sample_rows=2000, slots=32)
    assert calls[1]["gather_bytes"] == -(-max(terms) // (16 << 20)) * (16 << 20)
    assert calls[1]["max_tokens"] == 2000
    # A DSpark draft adds the markov transition bias gather (FP32 logits as BF16
    # pairs: slots x vocabulary/world x 4) and a 96-head LSE combine term; a large
    # request capacity makes the markov gather the largest contribution.
    drafted = SimpleNamespace(hidden=7168, vocabulary=163840, text={"num_attention_heads": 96},
                              draft=SimpleNamespace(heads=96))
    bootstrap._remote_backbone_runtime(config, None, None, core_storage=storage,
        model_config=drafted, shard=shard, channel_rows=1024, sample_rows=2000, slots=4000)
    markov = 4000 * (163840 // 8) * 4
    assert calls[2]["gather_bytes"] == -(-markov // (16 << 20)) * (16 << 20)


def test_remote_backbone_runtime_threads_the_pipe_chunk_override(monkeypatch):
    """A configured pipe chunk rides the backbone construction; none keeps b12x's 128 KiB chunks."""
    config = bootstrap.PreparationConfig(devices=(0,) * 8, hosts=tuple(f"h{i}" for i in range(8)),
        roce_ring=True, max_active=32, roce_pipe_chunk_bytes=524288)
    calls, runtime, storage = [], object(), object()
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
        lambda group, device, **kwargs: calls.append(kwargs) or runtime)
    model_config = SimpleNamespace(hidden=7168, vocabulary=163840, text={"num_attention_heads": 96}, draft=None)
    shard = SimpleNamespace(world_size=8)
    assert bootstrap._remote_backbone_runtime(config, None, None, core_storage=storage,
        model_config=model_config, shard=shard, channel_rows=1024, sample_rows=256, slots=32) is runtime
    assert calls[0]["pipe_chunk_bytes"] == 524288
    plain = bootstrap.PreparationConfig(devices=(0,) * 8, hosts=tuple(f"h{i}" for i in range(8)), roce_ring=True)
    bootstrap._remote_backbone_runtime(plain, None, None, core_storage=storage,
        model_config=model_config, shard=shard, channel_rows=1024, sample_rows=256, slots=32)
    assert "pipe_chunk_bytes" not in calls[1]
    with pytest.raises(ValueError, match="multiple of 16"):
        bootstrap.PreparationConfig(devices=(0,) * 8, hosts=tuple(f"h{i}" for i in range(8)),
            roce_ring=True, roce_pipe_chunk_bytes=1000)


@pytest.mark.parametrize("env,expected", [("262144", 262144), ("1048576", 1048576), (None, None)])
def test_create_roce_runtime_reads_the_pipe_chunk_environment(monkeypatch, env, expected):
    """FLARE_ROCE_PIPE_CHUNK_BYTES sizes the pipelined chunks like the varlen override does."""
    import sys
    from ignition.preparation.collectives import create_roce_runtime

    class Runtime:
        def __init__(self, **options):
            self.options = options

    monkeypatch.setitem(sys.modules, "b12x.comm", SimpleNamespace(roce=SimpleNamespace(AllReduce=Runtime)))
    if env is None:
        monkeypatch.delenv("FLARE_ROCE_PIPE_CHUNK_BYTES", raising=False)
    else:
        monkeypatch.setenv("FLARE_ROCE_PIPE_CHUNK_BYTES", env)
    runtime = create_roce_runtime(object(), "cuda:0", max_tokens=32, pipe_tokens=4096,
                                  hidden_size=4096, gather_bytes=4096)
    if expected is None:
        assert "pipe_chunk_bytes" not in runtime.options
    else:
        assert runtime.options["pipe_chunk_bytes"] == expected
    # An explicit argument beats the environment.
    runtime = create_roce_runtime(object(), "cuda:0", max_tokens=32, pipe_tokens=4096,
                                  hidden_size=4096, gather_bytes=4096, pipe_chunk_bytes=524288)
    assert runtime.options["pipe_chunk_bytes"] == 524288
    with pytest.raises(ValueError, match="multiple of 16"):
        create_roce_runtime(object(), "cuda:0", max_tokens=32, pipe_tokens=4096,
                            hidden_size=4096, gather_bytes=4096, pipe_chunk_bytes=1000)


def test_row_sharded_wrappers_bind_the_shard_geometry(monkeypatch):
    """The RS/AG pair mirrors one monolithic reduction: shard rows, rank-order gather surfaces."""
    import torch
    from b12x.preparation import PreparedCall
    from ignition.components import collectives

    source = torch.zeros((4096, 7168), dtype=torch.bfloat16)
    output = torch.zeros((4096, 7168), dtype=torch.bfloat16)
    requests, surfaces = [], []
    plan = SimpleNamespace(request=lambda **kwargs: requests.append(kwargs) or object())
    runtime = SimpleNamespace(world_size=16, pipe_bytes=4096 * 7168 * 2, _group=object(),
        reduce_scatter=lambda inp, out=None, plan=None: out.copy_(inp[:out.shape[0]]),
        gather_shards=lambda inp, out=None, plan=None: out.fill_(3))
    monkeypatch.setattr(collectives, "_roce", lambda candidate: candidate is runtime)
    monkeypatch.setattr(collectives, "_roce_plan",
                        lambda candidate, coordinator, surface: surfaces.append(surface) or plan)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _group: list(range(16)))

    pair = collectives.PreparedRowShardedAllReduce(source, output, runtime=runtime, session=object(),
        coordinator=object(), name="layer.reduce")
    assert pair.shard.shape == (256, 7168)
    assert surfaces == ["AllReduce.reduce_scatter", "AllReduce.gather_shards"]
    assert [request["name"] for request in requests] == ["layer.reduce.scatter", "layer.reduce.gather"]
    assert all(request["collective"].ranks == tuple(range(16)) for request in requests)
    # The prepare calls prime the runtime's prepared RS/AG states with the bound tensors.
    state = SimpleNamespace(reduce_scatter=lambda inp, out=None: out.fill_(1),
                            gather_shards=lambda inp, out=None: out.fill_(2))
    scatter_call = requests[0]["prepare_call"](state)
    gather_call = requests[1]["prepare_call"](state)
    assert isinstance(scatter_call, PreparedCall) and isinstance(gather_call, PreparedCall)
    assert scatter_call.owners == (runtime, source, pair.shard)
    assert gather_call.owners[2] is output
    scatter_call.run()
    assert pair.shard[0, 0].item() == 1
    gather_call.run()
    assert output[0, 0].item() == 2
    # The eager call drives both halves through the custom ops in emission order.
    source.fill_(7)
    assert pair(source) is output
    assert torch.equal(pair.shard, source[:256])
    assert output[0, 0].item() == 3 and output[-1, 0].item() == 3


def test_row_sharded_wrappers_reject_ineligible_geometry(monkeypatch):
    import torch
    from ignition.components import collectives

    runtime = SimpleNamespace(world_size=16, pipe_bytes=4096 * 7168 * 2, _group=object())
    monkeypatch.setattr(collectives, "_roce", lambda candidate: candidate is runtime)
    monkeypatch.setattr(collectives, "_roce_plan", lambda *args: SimpleNamespace(request=lambda **k: object()))
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _group: list(range(16)))
    # Rows must divide into world shards.
    with pytest.raises(ValueError, match="world shards"):
        collectives.PreparedRowShardedAllReduce(torch.zeros((4095, 7168), dtype=torch.bfloat16),
            torch.zeros((4095, 7168), dtype=torch.bfloat16), runtime=runtime, session=object(),
            coordinator=object(), name="reduce")
    # The whole message must stay within the pipelined capacity.
    with pytest.raises(ValueError, match="pipelined capacity"):
        collectives.PreparedRowShardedAllReduce(torch.zeros((8192, 7168), dtype=torch.bfloat16),
            torch.zeros((8192, 7168), dtype=torch.bfloat16), runtime=runtime, session=object(),
            coordinator=object(), name="reduce")
    # The wrappers exist only for the RoCE runtime.
    with pytest.raises(ValueError, match="RoCEnante"):
        collectives.PreparedRowShardedAllReduce(torch.zeros((4096, 7168), dtype=torch.bfloat16),
            torch.zeros((4096, 7168), dtype=torch.bfloat16), runtime=object(), session=object(),
            coordinator=object(), name="reduce")


@pytest.mark.parametrize("flag,world,rows,expected", [
    (False, 16, 4096, False), (True, 16, 4096, True), (True, 16, 4095, False), (True, None, 4096, False)])
def test_row_sharded_prefill_flag_gates_the_emission(monkeypatch, flag, world, rows, expected):
    from ignition.components import collectives

    if flag:
        monkeypatch.setenv("FLARE_ROCE_ROW_SHARDED_PREFILL", "1")
    else:
        monkeypatch.delenv("FLARE_ROCE_ROW_SHARDED_PREFILL", raising=False)
    runtime = None if world is None else SimpleNamespace(world_size=world, _group=object())
    monkeypatch.setattr(collectives, "_roce", lambda candidate: candidate is runtime and runtime is not None)
    assert collectives.row_sharded_prefill(runtime, rows) is expected


def test_multi_host_speculation_captures_the_glm53_width_ladder(tmp_path):
    """Multi-host exports capture one packed round per configured verification width.

    glm53's span-hosts template captures a graph for every width in the ladder, so the
    serving core always finds a captured width for the requested draft depth. K3.1's
    packed rounds adopt the same envelope (batch x width within the token capacity);
    single-host preparation keeps the packed single-width catalog.
    """
    from ignition.components.capacity import parallel_speculative_shapes

    widths = tuple(range(1, 9))
    draft = str(tmp_path / "dflash2")
    single = bootstrap.PreparationConfig(devices=tuple(range(8)), max_active=32, draft_checkpoint=draft,
        speculative_widths=widths, speculative_token_capacity=128)
    # Single-host preparation keeps the byte-identical packed single-width catalog.
    assert single.hosts is None
    assert single.speculative_round_shapes == parallel_speculative_shapes(
        single.max_active, widths, 128, 64)
    multi = bootstrap.PreparationConfig(devices=(0,) * 8, hosts=tuple(f"h{i}" for i in range(8)),
        roce_ring=True, max_active=32, draft_checkpoint=draft, speculative_widths=widths,
        speculative_token_capacity=128)
    shapes = multi.speculative_round_shapes
    assert {width for _, width, _ in shapes} == set(widths)
    # Every width verifies inside the token capacity envelope; width-one rounds stay
    # plain decode rounds at every batch.
    assert all(batch * width == packed for batch, width, packed in shapes)
    assert all(packed <= 128 for _, _, packed in shapes)
    assert {(batch, 1, batch) for batch in (1, 2, 4, 8, 16, 32)} <= set(shapes)
    assert multi.decode_samples == max(batch * width for batch, width, _ in shapes)
    assert multi.draft_tokens == 7 and multi.completion_token_capacity == 8
    # The ladder is deterministic and honors the graph budget.
    assert shapes == multi.speculative_round_shapes
    with pytest.raises(ValueError, match="graph budget of at least"):
        bootstrap.PreparationConfig(devices=(0,) * 8, hosts=tuple(f"h{i}" for i in range(8)),
            roce_ring=True, max_active=32, draft_checkpoint=draft, speculative_widths=widths,
            speculative_token_capacity=128, speculative_graph_budget=16)
    # The round envelope sizes the remote runtime's one-shot window: the draft block
    # runs all eight positions of the largest drafted batch (32 x 8 rows), beyond the
    # packed verification rows a single-width catalog planned.
    assert max(batch for batch, width, _ in shapes if width > 1) * (multi.draft_tokens + 1) == 256


def test_roce_backbone_ports_every_exchange_class():
    # Stage 5 flipped the last two entries (shared prefill staging and continuation
    # pages): the gate is empty under multi-host and the port is structurally complete.
    assert all(bootstrap.REMOTE_EXCHANGES_PORTED.values())
    assert bootstrap._unported_exchange_classes() == []


def test_remote_context_parallel_carries_the_stage_five_geometry():
    from ignition.recipes.kimi31.dcp import (INDEX_WINDOW_RECORDS, RECORD_BYTES,
                                             REMOTE_RECORD_ROWS, TOPK)
    from ignition.recipes.kimi31.query_selection import PreparedQuerySelection  # noqa: F401

    # The backbone's variable-length area covers both stage-5 windows exactly: one
    # 128-row part's selected records in the worst case, one index-history window.
    assert bootstrap.REMOTE_RECORD_ROWS == REMOTE_RECORD_ROWS == 128
    assert bootstrap.INDEX_WINDOW_POSITIONS == INDEX_WINDOW_RECORDS
    assert REMOTE_RECORD_ROWS * TOPK * RECORD_BYTES > INDEX_WINDOW_RECORDS * 132
    # Under multi-host the context carries no shared pages, staged slabs, or signals:
    # the remote context parallel is constructed with an empty peer map and the
    # exchange buffers are allocated per decoder, never through CUDA IPC.
    from ignition.recipes.kimi3.dcp import ContextShards
    from ignition.recipes.kimi31.dcp import ContextParallel as RemoteContextParallel

    context = RemoteContextParallel(ContextShards.on(0, 8, "cpu"), heads=object(), candidates=None,
                                    coordinator=None, attention_constants=None, peers={},
                                    remote=True)
    assert context.remote and context.peers == {}
    assert context.staging is None and context.staged_records is None
    assert context.index_staging is None and context.stage_signals == ()


def test_prompt_scoring_across_hosts_requires_the_remote_runtime():
    from ignition.components.prompt_logprobs import PromptScoring

    config = SimpleNamespace(prompt_logprobs=True, prefill_chunk_size=128)
    with pytest.raises(AssertionError, match="remote runtime"):
        PromptScoring.allocate(config, "cpu", {}, frame=object(), group=object(),
            ipc=object(), runtime=None, coordinator=SimpleNamespace(spans_hosts=True))
    # Single-rank scoring still constructs without any group or runtime.
    scoring = PromptScoring.allocate(config, "cpu", {}, frame=object())
    assert scoring is not None and scoring.runtime is None
    scoring.close()
