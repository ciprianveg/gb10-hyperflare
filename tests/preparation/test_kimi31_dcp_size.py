"""CPU configuration, descriptor geometry and capture-owner checks; no CUDA execution."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from ignition import planning
from ignition.cli.serve import _export_options
from ignition.preparation import capture
from ignition.preparation.execution import GraphSignature
from ignition.recipes import resolve
from ignition.recipes.kimi3 import bootstrap
from ignition.recipes.kimi3.config import ModelConfig
from ignition.recipes.kimi3.pool import continuation_layout, page_bytes
from ignition.recipes.kimi31.gqa_dspark_config import DSparkGQAConfig


def plan(size=None):
    model = resolve("kimi-k3.1-pro")
    devices = tuple(planning.Gpu(i, "GB10 metadata fixture", str(i), 256 << 30,
                                256 << 30, (12, 1)) for i in range(16))
    return planning.plan(model.recipe, model.checkpoint, devices, selected=tuple(range(16)),
        concurrency=2, reserve_bytes=2 << 30,
        speculation=replace(model.recipe.speculation, draft_tokens=6),
        image_concurrency=2, prefill_chunk=4096, dcp_size=size)


@pytest.mark.parametrize("size,global_page", [(None, 2048), (16, 2048), (8, 1024)])
def test_plan_descriptor_geometry_and_policy(size, global_page):
    value = plan(size)
    assert value.world_size == 16 and value.page_tokens == global_page
    assert value.to_dict()["dcp_size"] == size
    assert value.concurrency == 2 and value.prefill_chunk == 4096 and value.context == 1048576
    assert value.image_concurrency == 2 and value.speculation.widths == tuple(range(1, 8))
    assert value.recipe.sampling_defaults.temperature == 1
    assert value.recipe.sampling_defaults.top_p == .95
    fixture = Path(__file__).parents[1] / "fixtures/final_dspark"
    target = ModelConfig.from_dict(json.loads((fixture/"target-config.json").read_text()))
    target = replace(target, draft=DSparkGQAConfig.load(fixture, target))
    assert page_bytes(target) == value.costs.page_bytes
    layout = continuation_layout(target, 16, speculative_tokens=7)
    assert layout.row_bytes == value.costs.state_row_bytes
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("bad", [0, 4, 12, 32, True])
def test_invalid_plan_size_rejected(bad):
    with pytest.raises(planning.UsageError, match="dcp-size"):
        plan(bad)


def test_config_and_recipe_propagation(monkeypatch):
    from ignition.preparation import rank_group
    from ignition.recipes.kimi3 import recipe
    value = replace(plan(8), speculation=None)
    received = []
    monkeypatch.setattr(rank_group, "prepare_group", lambda snapshot, config, *a, **kw: received.append(config))
    recipe.prepare(Path("no-weight-read"), value, compile_forwards=False, kernel_cache=None,
                   progress=None, ranks=NS(hosts=tuple(f"host{i}" for i in range(16))),
                   recipe=value.recipe)
    config, = received
    assert config.dcp_size == 8 and config.page_tokens == 128 * 8
    assert config.devices == (0,) * 16 and config.context == value.context
    assert config.sampling_defaults["top_p"] == .95
    for kwargs in (dict(dcp_size=8), dict(dcp_size=8, context_parallel=False),
                   dict(dcp_size=12)):
        with pytest.raises(ValueError, match="DCP size"):
            bootstrap.PreparationConfig(**kwargs)
    assert bootstrap.PreparationConfig(devices=tuple(range(8)), context_parallel=False).page_tokens == 128
    assert bootstrap.PreparationConfig().page_tokens == 128 * 16


@pytest.mark.parametrize("rank", range(16))
def test_all_ranks_create_identical_groups_and_select_local_rank(monkeypatch, rank):
    calls = []
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(range(16)))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    def create(*, ranks, backend):
        assert backend == "gloo"
        calls.append(tuple(ranks))
        return tuple(ranks)
    monkeypatch.setattr(torch.distributed, "new_group", create)
    group, local, members = bootstrap.prepare_context_group("tp", rank=rank, world=16, size=8)
    assert calls == [tuple(range(8)), tuple(range(8, 16))]
    assert group == members == calls[rank // 8] and local == rank % 8
    calls.clear()
    assert bootstrap.prepare_context_group("tp", rank=rank, world=16, size=16) == (
        "tp", rank, tuple(range(16)))
    assert calls == []


def test_bad_membership_is_rejected_before_group_creation(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(reversed(range(16))))
    with pytest.raises(ValueError, match="ordered"):
        bootstrap.prepare_context_group("tp", rank=0, world=16, size=8)
    with pytest.raises(ValueError, match="outside"):
        bootstrap.prepare_context_group("tp", rank=16, world=16, size=8)
    with pytest.raises(ValueError, match="divide"):
        bootstrap.context_groups(16, 12)


def test_export_options_never_silently_rebuild_dcp8(monkeypatch):
    from ignition.preparation import source_identity
    monkeypatch.setattr(source_identity, "source_identity", lambda: "cpu-source")
    model = NS(directory=None)
    options = dict(rebuild_manifest={"stale": True}, keep="static")
    for size in (8, 16):
        assert _export_options(options, model, plan(size), compiled_forwards=True) == {"keep": "static"}
    result = _export_options(options, model, plan(), compiled_forwards=True)
    assert result["rebuild_manifest"]["context_parallel"] is True
    assert result["rebuild_manifest"]["tensor_parallel"] == 16
    assert options["rebuild_manifest"] == {"stale": True}


@pytest.mark.parametrize("plural", [False, True])
def test_one_graph_is_protected_by_all_runtime_and_session_guards(monkeypatch, plural):
    events, active = [], set()
    @contextmanager
    def guard(name):
        active.add(name)
        events.append(("enter", name))
        try:
            yield
        finally:
            active.remove(name)
            events.append(("exit", name))
    stream = NS(wait_stream=lambda other: None, synchronize=lambda: None)
    for name, value in dict(current_stream=lambda device: stream, stream=lambda stream: nullcontext(),
                            synchronize=lambda device: None, graph=lambda *a, **kw: guard("graph"),
                            mem_get_info=lambda device: (1024, 2048), memory_allocated=lambda device: 0,
                            memory_reserved=lambda device: 0).items():
        monkeypatch.setattr(torch.cuda, name, value)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None)
    monkeypatch.setattr(capture, "capture_graph", lambda **kw: "one-graph")
    tp = NS(capture=lambda: guard("tp-session"))
    dcp = NS(capture=lambda: guard("dcp-session"))
    runtime = NS(capture=lambda **kw: guard("tp-runtime"))
    dcp_runtime = NS(capture=lambda **kw: guard("dcp-runtime"))
    invocations = []
    def forward():
        invocations.append(set(active))
    graphs, _ = capture.capture_forwards({"decode": forward},
        {"decode": GraphSignature("decode", 1, 1)}, session=tp, device="cpu", stream=stream,
        pool=None, collective=runtime, compile_decoder=False,
        additional_sessions=(dcp,) if plural else (),
        additional_collectives=(dcp_runtime,) if plural else ())
    expected = {"tp-session", "tp-runtime", "graph"}
    if plural:
        expected |= {"dcp-session", "dcp-runtime"}
    assert graphs == {"decode": "one-graph"}
    assert invocations == [set(), expected]
    assert active == set()
    assert not torch.cuda.is_initialized()
