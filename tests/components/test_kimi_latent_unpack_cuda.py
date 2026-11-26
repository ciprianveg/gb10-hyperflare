"""Opt-in GPU regression for PR277's rank-interleaved Q/KV normalization."""
import importlib.util
import os
from pathlib import Path
import unittest

if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
    raise unittest.SkipTest("requires an explicitly released CUDA component window")

import torch

SOURCE = Path(__file__).resolve().parents[2] / "ignition/recipes/kimi3/pointwise.py"
spec = importlib.util.spec_from_file_location("kimi_pointwise_component", SOURCE)
pointwise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pointwise)

WORLD, Q, KV, TAIL = 16, 1536, 512, 64
LOCAL_Q, LOCAL_KV = Q // WORLD, (KV + TAIL) // WORLD
EPS = 1e-6


def inputs(rows, transposed=False, weight_dtype=torch.bfloat16):
    width = LOCAL_Q + LOCAL_KV
    if transposed:
        owner = torch.full((WORLD, rows + 2, width + 5), 17., device="cuda", dtype=torch.bfloat16)
        gathered = owner[:, 1:rows + 1, :width].transpose(0, 1)
    else:
        owner = torch.full((rows + 2, WORLD, width), 17., device="cuda", dtype=torch.bfloat16)
        gathered = owner[1:rows + 1]
    gathered.copy_(torch.randn(gathered.shape, device="cuda", dtype=torch.bfloat16))
    qw = torch.linspace(-1.5, 1.5, Q, device="cuda", dtype=weight_dtype)
    kw = torch.linspace(.25, 1.75, KV, device="cuda", dtype=weight_dtype)
    return owner, gathered, qw, kw


def outputs(rows, tail=TAIL):
    # Disjoint output views share a retained owner; input/output overlap is not supported.
    owner = torch.full((rows + 2, Q + KV + tail + 16), 13., device="cuda", dtype=torch.bfloat16)
    q = owner[1:rows + 1, 4:4 + Q]
    latent = owner[1:rows + 1, Q + 8:Q + 8 + KV + tail]
    return owner, q, latent


def reference(gathered, qw, kw, tail=TAIL, eps=EPS):
    # Canonical rank-interleaved unpack, followed by the unchanged separate norms.
    rows = gathered.shape[0]
    query = torch.cat([gathered[:, rank, :LOCAL_Q] for rank in range(WORLD)], dim=1)
    kv = torch.cat([gathered[:, rank, LOCAL_Q:LOCAL_Q + LOCAL_KV] for rank in range(WORLD)], dim=1)
    qout = torch.empty((rows, Q), device="cuda", dtype=torch.bfloat16)
    latent = torch.empty((rows, KV + tail), device="cuda", dtype=torch.bfloat16)
    pointwise.rms_norm_columns(query, qw, qout, eps)
    pointwise.rms_norm_columns(kv[:, :KV], kw, latent[:, :KV], eps)
    latent[:, KV:].copy_(kv[:, KV:KV + tail])
    return qout, latent, query, kv


class LatentUnpackTests(unittest.TestCase):
    def assert_same(self, a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def check_guards(self, owner, rows, tail):
        self.assertTrue(torch.all(owner[0] == 13).item())
        self.assertTrue(torch.all(owner[-1] == 13).item())
        for begin, end in ((0, 4), (Q + 4, Q + 8), (Q + 8 + KV + tail, owner.shape[1])):
            self.assertTrue(torch.all(owner[1:rows + 1, begin:end] == 13).item())

    def test_tp16_widths_c2_strides_and_padded_rows(self):
        torch.manual_seed(277)
        for batch in (1, 2):
            for width in range(1, 8):
                for transposed in (False, True):
                    for dtype in (torch.bfloat16, torch.float32):
                        rows = batch * width
                        with self.subTest(batch=batch, width=width, transposed=transposed, weight_dtype=dtype):
                            src_owner, gathered, qw, kw = inputs(rows, transposed, dtype)
                            # The operator has no live-row mask: padded zero rows must match
                            # the old sequence, while rows outside the supplied views stay untouched.
                            if rows > 1:
                                gathered[-1].zero_()
                            saved = src_owner.clone()
                            qref, lref, _, _ = reference(gathered, qw, kw)
                            owner, qout, latent = outputs(rows)
                            pointwise.unpack_and_rms_norm_qkv(
                                gathered, LOCAL_Q, LOCAL_KV, qw, kw, qout, latent, EPS)
                            self.assert_same(qout, qref)
                            self.assert_same(latent, lref)
                            self.assert_same(src_owner, saved)
                            self.check_guards(owner, rows, TAIL)

    def test_prefill_rows(self):
        for rows in (17, 4096):
            with self.subTest(rows=rows):
                src_owner, gathered, qw, kw = inputs(rows, transposed=True)
                saved = src_owner.clone()
                qref, lref, _, _ = reference(gathered, qw, kw)
                owner, qout, latent = outputs(rows)
                pointwise.unpack_and_rms_norm_qkv(
                    gathered, LOCAL_Q, LOCAL_KV, qw, kw, qout, latent, EPS)
                self.assert_same(qout, qref)
                self.assert_same(latent, lref)
                self.assert_same(src_owner, saved)
                self.check_guards(owner, rows, TAIL)

    def test_partial_tail_and_canonical_rounding(self):
        for tail in (0, 13, 64):
            _, gathered, qw, kw = inputs(7, transposed=True)
            # Exact-power inputs remove reduction-order ambiguity from the independent
            # PyTorch reference while exercising both BF16 rounding points and tail routing.
            signs = torch.arange(gathered.numel(), device="cuda").reshape(gathered.shape) % 2
            gathered.copy_(torch.where(signs == 0, 2., -2.))
            owner, qout, latent = outputs(7, tail)
            qref, lref, query, kv = reference(gathered, qw, kw, tail, eps=0.)
            def canonical(x, weight):
                x = x.float()
                normalized = (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True))).to(torch.bfloat16)
                return (normalized.float() * weight.float()).to(torch.bfloat16)
            self.assert_same(qref, canonical(query, qw))
            self.assert_same(lref[:, :KV], canonical(kv[:, :KV], kw))
            pointwise.unpack_and_rms_norm_qkv(
                gathered, LOCAL_Q, LOCAL_KV, qw, kw, qout, latent, 0.)
            self.assert_same(qout, qref)
            self.assert_same(latent, lref)
            self.check_guards(owner, 7, tail)

    def test_graph_replay_owners_overwrite_and_allocation_stability(self):
        for rows in (7, 14):
            src_owner, gathered, qw, kw = inputs(rows, transposed=True)
            out_owner, qout, latent = outputs(rows)
            def fused():
                pointwise.unpack_and_rms_norm_qkv(
                    gathered, LOCAL_Q, LOCAL_KV, qw, kw, qout, latent, EPS)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fused()
                fused()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused()
            addresses = (gathered.data_ptr(), qout.data_ptr(), latent.data_ptr())
            for epoch in range(3):
                gathered.copy_(torch.randn_like(gathered) * (epoch + 1))
                saved = src_owner.clone()
                qref, lref, _, _ = reference(gathered, qw, kw)
                qout.fill_(float("nan"))
                latent.fill_(float("nan"))
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(torch.cuda.memory_allocated(), allocated)
                self.assertEqual(addresses, (gathered.data_ptr(), qout.data_ptr(), latent.data_ptr()))
                self.assert_same(qout, qref)
                self.assert_same(latent, lref)
                self.assert_same(src_owner, saved)
                self.check_guards(out_owner, rows, TAIL)

    def test_noncontiguous_columns_rejected(self):
        _, gathered, qw, kw = inputs(3)
        _, qout, latent = outputs(3)
        invalid = torch.empty((3, WORLD, 2 * (LOCAL_Q + LOCAL_KV)),
                              device="cuda", dtype=torch.bfloat16)[:, :, ::2]
        with self.assertRaises(ValueError):
            pointwise.unpack_and_rms_norm_qkv(
                invalid, LOCAL_Q, LOCAL_KV, qw, kw, qout, latent, EPS)


if __name__ == "__main__":
    unittest.main()
