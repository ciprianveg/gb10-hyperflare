# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-3 remote decode route exchange: transport round trips and selection parity.

Every rewired site is checked with fake runtimes: a RoCEnante backbone is a plain object
(the real transport is exercised by the two-host RoCE test), and the exchange inside the
paired-column gather is simulated with the protocol's rank-major row layout.

Selection parity: the remote path cannot run the PCIe channel's fused ``kimi_topk16``, and
b12x's general prepared router is NOT a drop-in for it — the general kernel keeps the
highest expert id on exact score ties and rounds sigmoid differently — so the remote select
reimplements the fused kernel's semantics in tensor primitives. The reference below derives
the same semantics independently: a stable descending sort keeps the ascending expert order
of equal scores, which is exactly the fused kernel's complemented packed key.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3 import bootstrap, routing
from ignition.recipes.kimi3.routing import REMOTE_ROUTE_ROWS, PreparedRouteGather, kimi_route_topk16


def reference_route_select(logits, bias, top=16):
    """The fused kimi_topk16 semantics through an independent formulation.

    Selection score = tanh-form sigmoid + correction bias; non-finite inputs score -inf
    with zero weight; exact score ties resolve to the lower expert id (a stable descending
    sort preserves the ascending expert order of equal scores); weights are the unbiased
    sigmoid scores renormalized by a fold-in-halves sum with a 1e-20 floor.
    """
    finite = torch.isfinite(logits) & torch.isfinite(bias)
    unbiased = torch.where(finite, 0.5 * torch.tanh(0.5 * logits) + 0.5,
                           torch.zeros((), dtype=logits.dtype))
    selection = torch.where(finite, unbiased + bias, torch.full((), float("-inf"), dtype=logits.dtype))
    selection = torch.where(selection == 0, torch.zeros_like(selection), selection)
    order = torch.sort(selection, dim=-1, descending=True, stable=True).indices[:, :top].to(torch.int64)
    selected = torch.gather(unbiased, 1, order)
    folded = selected
    while folded.shape[1] > 1:
        half = folded.shape[1] // 2
        folded = folded[:, :half] + folded[:, half:]
    return order.to(torch.int32), selected * (1.0 / (folded[:, 0] + 1e-20)).unsqueeze(1)


@pytest.mark.parametrize("rows,seed", [(1, 3), (4, 5), (16, 7)])
def test_route_select_matches_kimi_topk16_semantics(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    experts = 896
    logits = torch.randn((rows, experts), generator=generator) * 3
    bias = torch.randn((experts,), generator=generator) * 0.7
    ids = torch.zeros((rows, 16), dtype=torch.int32)
    weights = torch.zeros((rows, 16), dtype=torch.float32)
    kimi_route_topk16(logits, bias, weights, ids)
    expected_ids, expected_weights = reference_route_select(logits, bias)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)
    # Slot order is descending selection score; the tanh-form sigmoid tracks the ordinary
    # one away from its rounding edges.
    selection = 0.5 * torch.tanh(0.5 * logits) + 0.5 + bias
    scores = torch.gather(selection, 1, ids.to(torch.int64))
    assert torch.equal(scores, scores.sort(dim=-1, descending=True).values)
    torch.testing.assert_close(0.5 * torch.tanh(0.5 * logits) + 0.5, torch.sigmoid(logits),
                               rtol=1e-6, atol=1e-6)


def test_route_select_breaks_exact_ties_toward_the_lower_expert():
    # Duplicate logits make the selection scores bitwise equal: the fused kernel's packed
    # key complements the expert index, so of two experts tied for the last slot the
    # lower-numbered one is selected.
    experts = 896
    bias = torch.zeros(experts)
    logits = torch.full((1, experts), -6.0)
    logits[0, :15] = torch.linspace(3.0, 0.8, 15)
    logits[0, 100] = logits[0, 500] = 0.0
    ids = torch.zeros((1, 16), dtype=torch.int32)
    weights = torch.zeros((1, 16), dtype=torch.float32)
    kimi_route_topk16(logits, bias, weights, ids)
    assert 100 in ids[0].tolist() and 500 not in ids[0].tolist()
    assert torch.equal(ids, reference_route_select(logits, bias)[0])
    # Exact ties manufactured through the correction bias alone.
    bias_b = torch.full((experts,), -6.0)
    bias_b[200] = bias_b[700] = 0.0
    for index in range(15):
        bias_b[300 + index] = 1.0
    logits_b = torch.zeros((1, experts))
    logits_b[0, 300:315] = torch.linspace(1.0, 0.1, 15)
    ids_b = torch.zeros((1, 16), dtype=torch.int32)
    kimi_route_topk16(logits_b, bias_b, weights, ids_b)
    assert 200 in ids_b[0].tolist() and 700 not in ids_b[0].tolist()
    assert torch.equal(ids_b, reference_route_select(logits_b, bias_b)[0])


def test_route_select_handles_non_finite_scores_like_the_fused_kernel():
    # Non-finite logits or biases score -inf with zero weight; the leftover slots fill
    # with the lowest-numbered unusable experts at exactly zero weight.
    experts = 896
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn((2, experts), generator=generator) * 0.5
    bias = torch.randn((experts,), generator=generator) * 0.3
    logits[0, :886] = float("nan")
    biased = bias.clone()
    biased[:886] = float("inf")
    ids = torch.zeros((2, 16), dtype=torch.int32)
    weights = torch.zeros((2, 16), dtype=torch.float32)
    kimi_route_topk16(logits, biased, weights, ids)
    expected_ids, expected_weights = reference_route_select(logits, biased)
    assert torch.equal(ids, expected_ids) and torch.equal(weights, expected_weights)
    for row in range(2):
        # Experts 886..895 are the only usable ones; six -inf slots fill with the
        # lowest-numbered unusable experts, at exactly zero weight.
        assert set(ids[row, :10].tolist()) == set(range(886, 896))
        assert ids[row, 10:].tolist() == [0, 1, 2, 3, 4, 5]
        assert torch.equal(weights[row, 10:], torch.zeros(6))
    assert torch.isfinite(weights).all()


class FakeExchange:
    """Simulates the backbone all-gather: rank-major packed rows into the receive buffer."""

    packed_rows = ()

    def __init__(self, source, output, **kwargs):
        self.output = output

    def __call__(self, source):
        self.output.copy_(torch.cat(list(self.packed_rows), dim=1)[:, :self.output.shape[1]])


@pytest.mark.parametrize("world,rows", [(2, 16), (8, 8), (8, 1), (16, 3)])
def test_remote_route_exchange_round_trips_full_rows(monkeypatch, world, rows):
    from ignition.components import column_pair_gather

    latent_width, experts = 3584, 896
    first_width, second_width = latent_width // world, experts // world
    columns = first_width + 2 * second_width
    generator = torch.Generator().manual_seed(rows * 31 + world)
    full_latent = torch.randn((rows, latent_width), generator=generator).to(torch.bfloat16)
    full_logits = torch.randn((rows, experts), generator=generator)
    bias = torch.randn((experts,), generator=generator)

    def packed(rank):
        row = torch.zeros((rows, columns), dtype=torch.bfloat16)
        row[:, :first_width].copy_(full_latent[:, rank * first_width:(rank + 1) * first_width])
        row[:, first_width:].copy_(
            full_logits.view(torch.bfloat16)[:, rank * 2 * second_width:(rank + 1) * 2 * second_width])
        return row

    FakeExchange.packed_rows = tuple(packed(rank) for rank in range(world))
    monkeypatch.setattr(column_pair_gather, "PreparedAllGather", FakeExchange)
    latent = torch.zeros((rows, latent_width), dtype=torch.bfloat16)
    weights = torch.zeros((rows, 16), dtype=torch.float32)
    ids = torch.zeros((rows, 16), dtype=torch.int32)
    operation = PreparedRouteGather(full_latent[:, :first_width].contiguous(),
        full_logits[:, :second_width].contiguous(), bias,
        latent, weights, ids, runtime=SimpleNamespace(world_size=world), session=None,
        coordinator=None, name="route", remote=True, transport={})
    assert operation.gather.columns == columns and operation.gather.send.shape == (rows, columns)
    assert operation.gather.received.shape == (rows, world * columns)
    operation()
    # The latent arrives bitwise; the FP32 router bits arrive as exact BF16 lane pairs.
    assert torch.equal(latent, full_latent)
    assert torch.equal(operation.logits.view(torch.int32), full_logits.view(torch.int32))
    # Selection over the gathered logits agrees with the fused kernel's semantics.
    expected_ids, expected_weights = reference_route_select(full_logits, bias)
    assert torch.equal(ids, expected_ids) and torch.equal(weights, expected_weights)


@pytest.mark.parametrize("world", [2, 8, 16])
def test_remote_route_exchange_rejects_row_overflow(world):
    rows = REMOTE_ROUTE_ROWS + 1
    first_width, second_width = 3584 // world, 896 // world
    with pytest.raises(ValueError, match=f"1-{REMOTE_ROUTE_ROWS} row"):
        PreparedRouteGather(torch.zeros((rows, first_width), dtype=torch.bfloat16),
            torch.zeros((rows, second_width)), torch.zeros(896),
            torch.zeros((rows, 3584), dtype=torch.bfloat16), torch.zeros((rows, 16)),
            torch.zeros((rows, 16), dtype=torch.int32),
            runtime=SimpleNamespace(world_size=world), session=None, coordinator=None,
            name="route", remote=True, transport={})
    with pytest.raises(ValueError, match="shared transport"):
        PreparedRouteGather(torch.zeros((1, first_width), dtype=torch.bfloat16),
            torch.zeros((1, second_width)), torch.zeros(896),
            torch.zeros((1, 3584), dtype=torch.bfloat16), torch.zeros((1, 16)),
            torch.zeros((1, 16), dtype=torch.int32),
            runtime=SimpleNamespace(world_size=world), session=None, coordinator=None,
            name="route", remote=True)


def test_pcie_route_gather_keeps_its_channel_paths_and_row_cap(monkeypatch):
    from b12x.comm import pcie

    surfaces = []

    def fake_query(runtime, *, surface, call):
        surfaces.append((surface, dict(call)))
        return SimpleNamespace(surface=surface, call=call)

    class Plan:
        def request(self, *args, **kwargs):
            return object()

    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: (0, 1))
    monkeypatch.setattr(pcie, "query_from_runtime", fake_query)
    monkeypatch.setattr(pcie, "plan", lambda query, *, runtime: Plan())
    monkeypatch.setattr(routing, "declare", lambda *args, **kwargs: None)
    runtime = SimpleNamespace(world_size=2, exchange_group=object())

    def build(rows):
        return PreparedRouteGather(torch.zeros((rows, 1792), dtype=torch.bfloat16),
            torch.zeros((rows, 448)), torch.zeros(896),
            torch.zeros((rows, 3584), dtype=torch.bfloat16), torch.zeros((rows, 16)),
            torch.zeros((rows, 16), dtype=torch.int32),
            runtime=runtime, session=None, coordinator=None, name="route")

    build(1)
    assert surfaces[-1][0] == "DcpAllToAll.all_gather_pair_kimi_topk"
    build(8)
    assert [surface for surface, _ in surfaces[-2:]] == ["DcpAllToAll.all_gather_pair", "DcpAllToAll.kimi_topk16"]
    # The PCIe channel keeps its eight-row capacity: our pinned b12x compiles no
    # sixteen-row fused top-16 (voipmonitor #15 stays a remote-path algorithm).
    with pytest.raises(ValueError, match="1-8 row"):
        build(9)
    assert len(surfaces) == 3


class Plan:
    def memory_requirements(self):
        return SimpleNamespace(scratch_nbytes=0)


class Workspace:
    overlap = True

    def __init__(self):
        self.auxiliary_stream = SimpleNamespace(cuda_stream=0)

    def register_parallel(self, *args):
        pass

    def views(self, *specs):
        return None, self


class StubProjection(SimpleNamespace):
    def __call__(self, source):
        return self.output


def _latent_moe(monkeypatch, remote, workspace, world=8):
    from ignition.recipes.kimi3 import decoder as decoder_module

    gathers, chains = [], []

    def projection(source, weight, output, **kwargs):
        return StubProjection(output=output, plan=Plan(),
                              parts=(weight, torch.zeros(1), None))

    def route_gather(local_down, local_router, correction, latent, weights, ids, **kwargs):
        gathers.append(kwargs)
        return StubProjection(output=latent, plan=Plan())

    def chain(*args, **kwargs):
        chains.append(kwargs)
        return SimpleNamespace(start=lambda *a: None, finish=lambda *a: None)

    class StubRouterProjection:
        def __init__(self, source, weight, output, **kwargs):
            self.plan, self.output = Plan(), output

        def __call__(self, source):
            return self.output

    monkeypatch.setattr(decoder_module, "prepare_projection", projection)
    monkeypatch.setattr(decoder_module, "UnquantizedProjection", StubRouterProjection)
    monkeypatch.setattr(decoder_module, "PreparedChunkedAllReduce",
                        lambda source, output, **kwargs: StubProjection(output=output))
    monkeypatch.setattr(decoder_module, "PreparedRoutedExperts",
                        lambda source, expert_weights, ids, weights, output, **kwargs:
                        StubProjection(output=output, plan=Plan()))
    monkeypatch.setattr(decoder_module, "PreparedRouter",
                        lambda logits, scratch, **kwargs: StubProjection(output=logits))
    monkeypatch.setattr(decoder_module, "PreparedRouteGather", route_gather)
    monkeypatch.setattr(decoder_module, "ConcurrentProjectionChain",
                        chain if not remote else
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("overlap constructed")))
    weights = {
        "latent_up": torch.zeros((64, 64), dtype=torch.bfloat16),
        "latent_down_rows": torch.zeros((64, 64), dtype=torch.bfloat16),
        "latent_norm": torch.ones((64,)),
        "router": torch.zeros((896 // world,)),
        "correction": torch.zeros((896,)),
        "experts": object(),
        "shared": {"gate_up": torch.zeros((64, 64), dtype=torch.bfloat16),
                   "down": torch.zeros((64, 64), dtype=torch.bfloat16)},
    }
    config = SimpleNamespace(text={"routed_expert_hidden_size": 64, "num_experts": 896,
        "num_experts_per_token": 16, "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0}, hidden=64)
    shard = SimpleNamespace(world_size=world)
    storage = SimpleNamespace(hidden_columns=torch.arange(64), latent_begin=torch.zeros(1, dtype=torch.int32))
    buffers = {}
    operation = decoder_module.PreparedLatentMoE(weights, torch.zeros((4, 64), dtype=torch.bfloat16),
        config, shard, runtime=object(), router_runtime=object(), coordinator=None, session=None,
        name="moe", workspace=workspace, buffers=buffers, storage=storage,
        route_runtime=SimpleNamespace(world_size=world), route_remote=remote)
    return gathers, chains, buffers


def test_prepared_latent_moe_keeps_the_route_exchange_off_the_auxiliary_stream(monkeypatch):
    gathers, chains, buffers = _latent_moe(monkeypatch, remote=True, workspace=Workspace())
    assert len(gathers) == 1
    assert gathers[0]["remote"] is True
    # One shared transport pair serves every layer serially.
    assert gathers[0]["transport"] is buffers["route_transport"]
    # The auxiliary-stream forks never exist under the single ordered RoCE channel.
    assert chains == []


def test_prepared_latent_moe_keeps_pcie_overlaps_on_one_host(monkeypatch):
    gathers, chains, buffers = _latent_moe(monkeypatch, remote=False, workspace=Workspace())
    assert len(gathers) == 1
    assert gathers[0].get("remote", False) is False
    assert len(chains) == 1


def test_bootstrap_flips_only_the_stage_four_exchange_classes():
    # The stage-5 registry state: every exchange class rides the backbone — the
    # stage-4 classes plus the two prefill classes — and the gate is empty.
    ported = {name for name, value in bootstrap.REMOTE_EXCHANGES_PORTED.items() if value}
    assert ported == {bootstrap.HEAD_EXCHANGE_CLASS, "the decode route top-16 exchange",
                      "the DSA candidate and owner exchanges", "shared prefill staging",
                      "IPC continuation-page peers"}
    assert bootstrap._unported_exchange_classes() == []
