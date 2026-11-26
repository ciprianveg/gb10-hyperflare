"""Future GPU release only: existing confidence operator versus pinned CPU logits.

This is a confidence component check, not full draft or target qualification.
"""
import importlib.util
import json
import os
from pathlib import Path
import struct
import time
from decimal import Decimal, localcontext
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires separate parent GPU/compilation release")


def _sigmoid_copy(src, dst):
    dst.copy_(src.sigmoid())


def _sigmoid_source_paths(since_ns):
    """Report relevant newly written sources only; cache hits may write nothing."""
    root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    found = []
    if root:
        for path in sorted(Path(root).rglob("*.py")):
            if path.stat().st_mtime_ns < since_ns:
                continue
            lines = path.read_text(errors="replace").splitlines()
            excerpts = [(i + 1, line) for i, line in enumerate(lines)
                        if any(word in line for word in ("sigmoid", "exp(", "exp2(", "rcp(", "reciprocal"))]
            if excerpts:
                found.append(dict(path=str(path), expressions=excerpts[:24]))
            if len(found) == 12:
                break
    return dict(inductor_cache=root, triton_cache=os.environ.get("TRITON_CACHE_DIR"),
                newly_written_sources=found,
                limitation="bounded source candidates, not kernel provenance; cache hits may emit no source")


def _sigmoid_diagnostic(logits, expected_logits, output, eager, standalone, repeat_outputs,
                        repeat_logits, standalone_repeats, eager_repeats, *, batch, proposals, value, sources):
    """Decimal math and FP32 bit distances are observations, not acceptance bounds."""
    def bits(number):
        return struct.unpack("<I", struct.pack("<f", number))[0]

    def ordered(number):
        word = bits(number)
        return ~word & 0xffffffff if word & 0x80000000 else word | 0x80000000

    def flat(tensor):
        return tensor.detach().cpu().reshape(-1).tolist()

    values = {name: flat(tensor) for name, tensor in (
        ("logits", logits), ("expected_logits", expected_logits), ("full", output),
        ("eager", eager), ("standalone", standalone))}
    rows = []
    with localcontext() as ctx:
        ctx.prec = 80
        for i, x in enumerate(values["logits"]):
            exact_input = Decimal.from_float(x)
            mathematical = Decimal(1) / (Decimal(1) + (-exact_input).exp())
            # Probabilities here are positive normal FP32 values. Report error
            # in units of adjacent FP32 spacing, not a chosen tolerance.
            nearest_word = bits(float(mathematical))
            adjacent = [struct.unpack("<f", struct.pack("<I", w))[0]
                        for w in (nearest_word - 1, nearest_word, nearest_word + 1)]
            nearest = min(adjacent, key=lambda f: (abs(Decimal.from_float(f) - mathematical), bits(f) & 1))
            spacing = Decimal.from_float(struct.unpack("<f", struct.pack("<I", bits(nearest) + 1))[0])
            spacing -= Decimal.from_float(nearest)
            row = dict(index=i, mathematical_sigmoid_decimal80=str(mathematical),
                       values={k: v[i] for k, v in values.items()},
                       fp32_bits={k: f"0x{bits(v[i]):08x}" for k, v in values.items()})
            row["math_error"] = {}
            for name in ("full", "eager", "standalone"):
                error = abs(Decimal.from_float(values[name][i]) - mathematical)
                row["math_error"][name] = dict(abs=str(error), relative=str(error / abs(mathematical)),
                                               ulp_spacing_units=str(error / spacing),
                                               distance_to_nearest_fp32=abs(ordered(values[name][i]) - ordered(nearest)))
            row["nearest_math_fp32_bits"] = f"0x{bits(nearest):08x}"
            row["pairwise"] = {}
            for left, right in (("full", "eager"), ("standalone", "eager"), ("full", "standalone")):
                a, b = values[left][i], values[right][i]
                row["pairwise"][f"{left}_vs_{right}"] = dict(
                    abs=abs(a - b), relative=abs(a - b) / abs(b) if b else None,
                    ulp_distance=abs(ordered(a) - ordered(b)))
            rows.append(row)
        maxima = {
            name: {metric: max(row["pairwise"][name][metric] for row in rows)
                   for metric in ("abs", "relative", "ulp_distance")}
            for name in rows[0]["pairwise"]}
        math_maxima = {
            name: {metric: str(max(Decimal(row["math_error"][name][metric]) for row in rows))
                   for metric in ("abs", "relative", "ulp_spacing_units")}
            for name in ("full", "eager", "standalone")}
    repeats = {}
    for name, baseline, samples in (
        ("full", output, repeat_outputs), ("logits", logits, repeat_logits),
        ("standalone", standalone, standalone_repeats), ("eager", eager, eager_repeats)):
        original = flat(baseline)
        repeats[name] = [
            dict(fp32_bits=[f"0x{bits(x):08x}" for x in flat(sample)],
                 max_abs=max(abs(a - b) for a, b in zip(original, flat(sample))),
                 max_ulp=max(abs(ordered(a) - ordered(b)) for a, b in zip(original, flat(sample))))
            for sample in samples]
    print("CONFIDENCE_SIGMOID_DIAGNOSTIC " + json.dumps(dict(
        batch=batch, proposals=proposals, hidden_value=value, rows=rows,
        pairwise_maxima=maxima, mathematical_error_maxima=math_maxima,
        repeatability=repeats, source_candidates=sources,
        criterion="exact pinned logits and compiled sigmoid integration; math errors are diagnostic"), sort_keys=True), flush=True)


@pytest.mark.parametrize("start", [0, 1, 2047, 2048, 2049, 1048570])
def test_existing_private_window_staging_does_not_mutate_accepted_ring(start):
    from ignition.recipes.kimi3.dspark import stage_windows
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    window, block, device = 2048, 6, "cuda:0"
    history = torch.arange(2 * window * 2, dtype=torch.float32, device=device).view(2, window, 1, 2)
    original = history.clone()
    proposal = torch.arange(2 * block * 2, dtype=torch.float32, device=device).view(2 * block, 1, 2) + 100000
    output = torch.empty((2, window + block, 1, 2), device=device)
    metadata = NS(starts=torch.tensor([start, start], device=device),
                  state_rows=torch.tensor([0, 1], device=device),
                  query_start_loc=torch.tensor([0, block, 2 * block], device=device),
                  lengths=torch.tensor([block, 0], device=device))
    stage_windows(history, proposal, output, metadata)
    prefix = min(start, window)
    expected = torch.zeros_like(output)
    positions = torch.arange(start - prefix, start, device=device) % window
    expected[0, :prefix] = history[0, positions]
    expected[0, prefix:prefix + block] = proposal[:block]
    assert torch.equal(output, expected)
    assert torch.equal(history, original)
    torch.cuda.synchronize()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("proposals", range(1, 7))
@torch.no_grad()
def test_confidence_logits_bias_predecessors_and_captured_storage(batch, proposals):
    from ignition.components.split_draft import PreparedDraftConfidence
    from ignition.preparation.tuning import RecordedPreparationSession
    path = Path("/tmp/opencode/k31-speculators-05f9ba2324fe-reference/repo/"
                "src/speculators/models/dspark/model_definitions.py")
    spec = importlib.util.spec_from_file_location("pinned_dspark_confidence", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "GPU assignment must be explicit"
    device = torch.device("cuda:0")
    head = reference.ConfidenceHead(7680)
    head.proj.weight.copy_((torch.arange(7680).remainder(7).float() - 3).view(1, -1) / 32)
    head.proj.bias.fill_(1 / 32)
    hidden = torch.zeros((batch * proposals, 7168), dtype=torch.bfloat16, device=device)
    previous = torch.zeros((batch, 512), dtype=torch.bfloat16, device=device)
    output = torch.empty((batch, proposals), dtype=torch.float32, device=device)
    weight = head.proj.weight.to(device, torch.bfloat16)
    bias = head.proj.bias.to(device, torch.bfloat16)
    with RecordedPreparationSession(device=device, checkpoint="dspark-confidence-synthetic",
            workload_schema="hidden-plus-predecessor", numerical_policy="bf16-dyadic-exact",
            compile_workers=1) as session:
        source = NS(owner=NS(block_size=proposals, logit_offset=0, session=session),
                    batch=batch, width=proposals + 1,
                    draft=NS(head_hidden=hidden), markov=NS(embedding=previous))
        operation = PreparedDraftConfidence(source, weight, output, bias=bias, name="confidence")
        for position in range(proposals):
            previous.fill_((position + 1) / 32)
            operation.record_embedding(position, previous)
        session.tune(show_progress=False)
        compile_started_ns = time.time_ns()
        forward = torch.compile(operation, fullgraph=True)
        forward()
        # Match layout/dtype without changing production inputs or compiler flags.
        standalone_src = torch.empty_strided(operation.logits.shape, operation.logits.stride(),
                                             dtype=operation.logits.dtype, device=device)
        standalone_dst = torch.empty_strided(output.shape, output.stride(), dtype=output.dtype, device=device)
        standalone_forward = torch.compile(_sigmoid_copy, fullgraph=True)
        standalone_src.copy_(operation.logits)
        standalone_forward(standalone_src, standalone_dst)
        torch.cuda.synchronize()
        source_candidates = _sigmoid_source_paths(compile_started_ns)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        try:
            for value in (1 / 32, -1 / 32, 3 / 32):
                hidden.fill_(value)
                # Overwrite scratch: saved predecessors must remain position-specific.
                previous.fill_(99)
                features = torch.cat((hidden.cpu().float().view(batch, proposals, 7168),
                    operation.embeddings.cpu().float()), dim=-1)
                expected_logits = head(features)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                assert torch.equal(operation.logits.cpu(), expected_logits)
                # Diagnostics are outside the unchanged replay allocation guard.
                # Preserve its original outputs so repeat replays cannot mask a failure.
                original_output, original_logits = output.clone(), operation.logits.clone()
                eager = expected_logits.to(device).sigmoid()
                # Independent pinned-reference logits, not the operator's logits.
                standalone_src.copy_(expected_logits)
                standalone_forward(standalone_src, standalone_dst)
                standalone_result = standalone_dst.clone()
                standalone_repeats, repeated_outputs, repeated_logits, eager_repeats = [], [], [], []
                for _ in range(2):
                    eager_repeats.append(expected_logits.to(device).sigmoid().cpu())
                    standalone_forward(standalone_src, standalone_dst)
                    standalone_repeats.append(standalone_dst.cpu())
                    graph.replay()
                    torch.cuda.synchronize()
                    repeated_outputs.append(output.cpu())
                    repeated_logits.append(operation.logits.cpu())
                output.copy_(original_output)
                operation.logits.copy_(original_logits)
                _sigmoid_diagnostic(original_logits, expected_logits, original_output, eager,
                    standalone_result, repeated_outputs, repeated_logits, standalone_repeats,
                    eager_repeats,
                    batch=batch, proposals=proposals, value=value, sources=source_candidates)
                assert all(torch.equal(x, original_output.cpu()) for x in repeated_outputs)
                assert all(torch.equal(x, original_logits.cpu()) for x in repeated_logits)
                assert all(torch.equal(x, standalone_result.cpu()) for x in standalone_repeats)
                # Preserve the compiled Torch primitive; eager CUDA need not be bit-identical.
                assert torch.equal(output, standalone_result)
                assert torch.isfinite(output).all()
                assert ((output >= 0) & (output <= 1)).all()
        finally:
            torch.cuda.synchronize()
            graph.reset()
