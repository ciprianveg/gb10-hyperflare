# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 decode routing: latent-down and router shards gathered with expert selection.

Each rank projects its rows of the routed-expert down projection and its experts' router logits.
One b12x exchange gathers both shards; a single decode row also selects Kimi's 16 experts inside
that exchange, and two to eight rows select them with b12x's prepared top-16 afterwards. This
replaces the router all-gather, the separate top-k, and the latent all-reduce of the general path.

Ranks on separate hosts cannot share the PCIe channel. The remote variant (remote-CP stage 3)
packs both shards into one BF16 send row over the shared RoCE backbone — the latent columns
native, every FP32 router bit as a BF16 lane pair — and selects experts afterwards with the
fused kernel's exact semantics, single row or not. It keeps the route gather for nine to sixteen
decode rows, the verify batches of two concurrent speculative streams, instead of falling back
to the general path (voipmonitor/flare #15's algorithm); the PCIe channel keeps its eight-row
capacity because the pinned b12x compiles no sixteen-row Kimi top-16.
"""

import itertools
import weakref

import torch
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.components.column_pair_gather import PreparedColumnPairGather
from ignition.preparation.tuning import declare

MAX_ROWS = 8
# The remote exchange rides the RoCE backbone, whose gather step is byte-capacity bound, so the
# row limit is the selection's, not a channel's: sixteen rows keep two-stream verify batches on
# the route path (voipmonitor #15).
REMOTE_ROUTE_ROWS = 16
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def create_route_runtime(group, device, *, ipc=None):
    """Allocate the b12x exchange that carries 1-8 rows of K3 latent and router shards."""
    import torch.distributed as dist
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    if world_size < 2 or 3584 % world_size or 896 % world_size:
        raise ValueError("K3 route exchange requires an even latent and expert partition")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device, max_batch_size=MAX_ROWS,
        total_heads=world_size, head_dim=8, query_head_dim=(3584 * 2 + 896 * 4) // world_size,
        stream_affine=False, ipc=ipc)


def kimi_route_topk16(logits, correction_bias, weights, ids):
    """Select Kimi's top-16 experts with the fused b12x kernel's exact semantics.

    The remote exchange cannot run the PCIe kernel's fused selection, and b12x's general
    prepared router differs from it on exact score ties (the general kernel keeps the highest
    expert id; the fused kernel's packed key keeps the lowest) and in its sigmoid rounding, so
    the selection is expressed in tensor primitives instead:

    the selection score is sigmoid(logit) + correction bias with sigmoid in the kernel's tanh
    form; non-finite logits or biases score -inf with zero unbiased weight; -0.0 normalizes to
    +0.0 so both zeros share one tie group; exact score ties resolve to the lower expert id
    (the packed key complements the index, so an integer max breaks ties downward); route
    weights are the unbiased sigmoid scores renormalized by a fold-in-halves sum over the
    sixteen slots with a 1e-20 floor, in descending-score order.
    """
    top = ids.shape[1]
    finite = torch.isfinite(logits) & torch.isfinite(correction_bias)
    # tanh(0.5x)/2 + 0.5 is the kernel's sigmoid; non-finite inputs score zero weight.
    unbiased = torch.where(finite, 0.5 * torch.tanh(0.5 * logits) + 0.5,
                           torch.zeros((), dtype=logits.dtype, device=logits.device))
    selection = torch.where(finite, unbiased + correction_bias,
                            torch.full((), float("-inf"), dtype=logits.dtype, device=logits.device))
    selection = torch.where(selection == 0, torch.zeros_like(selection), selection)
    bits = selection.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    keys = torch.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits | 0x80000000)
    experts = torch.arange(logits.shape[1], dtype=torch.int64, device=logits.device)
    # A plain integer max over (score order key, complemented expert): better candidates are
    # larger keys, and equal scores keep the lower expert id. The key rides the high 48 bits
    # — a 32-bit shift would overflow the signed integer topk runs on.
    packed = keys << 16 | (0xFFFF - experts)
    chosen = 0xFFFF - (torch.topk(packed, top, dim=-1).values & 0xFFFF)
    ids.copy_(chosen.to(torch.int32))
    selected = torch.gather(unbiased, 1, chosen)
    folded = selected
    while folded.shape[1] > 1:
        half = folded.shape[1] // 2
        folded = folded[:, :half] + folded[:, half:]
    weights.copy_(selected * (1.0 / (folded[:, 0] + 1e-20)).unsqueeze(1))


@torch.library.custom_op("flare::kimi3_route_select", mutates_args=("weights", "ids"))
def _route_select(logits: torch.Tensor, correction_bias: torch.Tensor, weights: torch.Tensor,
                  ids: torch.Tensor) -> None:
    kimi_route_topk16(logits, correction_bias, weights, ids)


@_route_select.register_fake
def _fake_route_select(logits, correction_bias, weights, ids) -> None:
    pass


class PreparedRouteGather:
    """Gather latent and FP32 router shards into full rows, then select 16 experts per row.

    The latent, expert weights and expert IDs are caller-owned outputs read by the routed experts.
    Every rank must declare the same operation names and replay them serially. Ranks on separate
    hosts set ``remote``: both shards pack into one BF16 send row over the shared RoCE backbone
    (the shared ``transport`` buffers serve every layer serially) and the fused selection runs
    afterwards on the main stream, so single-row and multi-row forwards take one path.
    """

    def __init__(self, local_down, local_router, correction_bias, latent, weights, ids, *, runtime, session,
                 coordinator, name, remote=False, transport=None):
        import torch.distributed as dist
        from b12x.comm import pcie

        rows = local_down.shape[0]
        world = runtime.world_size
        max_rows = REMOTE_ROUTE_ROWS if remote else MAX_ROWS
        expected = ((local_down, (rows, 3584 // world), torch.bfloat16), (local_router, (rows, 896 // world), torch.float32),
                    (correction_bias, (896,), torch.float32), (latent, (rows, 3584), torch.bfloat16),
                    (weights, (rows, 16), torch.float32), (ids, (rows, 16), torch.int32))
        wrong = [(tuple(t.shape), t.dtype, t.is_contiguous(), shape, dtype) for t, shape, dtype in expected
                 if t.shape != shape or t.dtype != dtype or not t.is_contiguous()]
        if not 1 <= rows <= max_rows or wrong:
            raise ValueError(f"K3 route gather requires contiguous 1-{max_rows} row latent, router and route tensors: {wrong}")
        self.runtime, self.rows, self.remote = runtime, rows, remote
        self.local_down, self.local_router, self.correction_bias = local_down, local_router, correction_bias
        self.latent, self.weights, self.ids = latent, weights, ids
        if remote:
            if transport is None:
                raise ValueError("the remote route gather needs shared transport buffers")
            self.logits = torch.zeros((rows, 896), dtype=torch.float32, device=latent.device)
            # The send row carries the latent columns native and every FP32 router bit as a BF16
            # lane pair; one whole-row gather fits the backbone's byte capacity.
            self.gather = PreparedColumnPairGather(local_down, local_router, latent, self.logits,
                runtime=runtime, session=session, coordinator=coordinator, name=name, buffers=transport,
                columns=local_down.shape[1] + 2 * local_router.shape[1], max_rows=REMOTE_ROUTE_ROWS)
            return
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        requests = []
        if rows == 1:
            self.fused = dict(local_down=local_down, local_router=local_router, correction_bias=correction_bias,
                              out_down=latent, topk_weights=weights, topk_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair_kimi_topk",
                                                          call=self.fused), runtime=runtime)
            requests.append(self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.fused), output=latent, owners=(runtime, latent, weights, ids)),
                collective=CollectiveRequirement(key=name, ranks=ranks)))
        else:
            self.logits = torch.zeros((rows, 896), dtype=torch.float32, device=latent.device)
            self.pair = dict(local_first=local_down, local_second=local_router, out_first=latent,
                             out_second=self.logits, threads=512, row_blocks=2 if rows == 8 else 1)
            self.select = dict(router_logits=self.logits, correction_bias=correction_bias,
                               output_weights=weights, output_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair",
                                                          call=self.pair), runtime=runtime)
            self.select_plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.kimi_topk16",
                                                                 call=self.select), runtime=runtime)
            requests.append(self.plan.request(name=name + "/gather", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.pair), output=latent, owners=(runtime, latent, self.logits)),
                collective=CollectiveRequirement(key=name + "/gather", ranks=ranks)))
            requests.append(self.select_plan.request(name=name + "/select", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.select), output=weights, owners=(runtime, self.logits, weights, ids))))
        declare(session, tuple(requests), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self):
        if self.remote:
            self.gather()
            _route_select(self.logits, self.correction_bias, self.weights, self.ids)
            return self.latent
        _route_gather(self.local_down, self.local_router, self.latent, self.weights, self.ids, self.handle)
        return self.latent


@torch.library.custom_op("flare::kimi3_route_gather", mutates_args=("latent", "weights", "ids"))
def _route_gather(local_down: torch.Tensor, local_router: torch.Tensor, latent: torch.Tensor,
                  weights: torch.Tensor, ids: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    runtime = owner.runtime
    if owner.rows == 1:
        runtime.all_gather_pair_kimi_topk(local_down, local_router, owner.correction_bias, latent, weights, ids,
                                          plan=owner.plan)
        return
    runtime.all_gather_pair(local_down, local_router, latent, owner.logits, plan=owner.plan, threads=512)
    runtime.kimi_topk16(owner.logits, owner.correction_bias, weights, ids, plan=owner.select_plan)


@_route_gather.register_fake
def _fake_route_gather(local_down, local_router, latent, weights, ids, handle):
    pass
