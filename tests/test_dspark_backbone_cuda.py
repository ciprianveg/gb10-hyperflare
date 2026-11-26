"""Draft-only diagnostic with canonical computation in a separate environment.

Collection imports no Torch or Speculators. Preflight imports the canonical
package in a CPU-only subprocess. This is not distributed/state qualification.
"""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

PIN = "05f9ba2324fef505cef8700669f2bdc20f8ae219"
REFERENCE = Path("/tmp/opencode/k31-speculators-05f9ba2324fe-reference/repo")
REFERENCE_PYTHON = Path("/tmp/opencode/k31-speculators-reference-env/bin/python")
REFERENCE_HELPER = Path(__file__).with_name("dspark_reference_process.py")
CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires coordinated Ops GPU assignment and compilation release")


def preflight():
    """Inspect engine metadata, then import the reference in a CPU-only process."""
    versions, missing = {}, []
    for module, distribution in (
        ("torch", "torch"), ("transformers", "transformers"),
        ("b12x", "b12x"), ("flashinfer", "flashinfer-python"),
        ("safetensors", "safetensors"), ("triton", "triton"),
    ):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
        try:
            versions[module] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[module] = "distribution metadata absent"
    # This parent-supplied pinned source is an archive extraction, not a Git
    # checkout. Revision identity is an input prerequisite, not inferred from
    # its directory name or falsely certified by these availability checks.
    assert (REFERENCE / "src/speculators/models/dflash/model_definitions.py").is_file()
    versions["reference_revision_required"] = PIN
    print(json.dumps(dict(dependencies=versions, missing=missing), sort_keys=True))
    assert not missing, f"missing dependencies: {missing}; installation is not authorized"
    assert versions["transformers"] == "5.21.0.dev0", "do not replace the preparation environment"
    assert REFERENCE_PYTHON.is_file(), "isolated canonical reference environment is required"
    result = subprocess.run([str(REFERENCE_PYTHON), "-B", str(REFERENCE_HELPER), "--preflight"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True, text=True, check=True)
    print(result.stdout)
    return versions


def test_dependency_preflight():
    before = set(sys.modules)
    print(json.dumps(preflight(), sort_keys=True))
    assert not ({"torch", "speculators", "flashinfer", "b12x"} & (set(sys.modules) - before))


def test_real_weight_memory_plan():
    """Tensor-only prerequisites, from the local metadata fixture; not admission."""
    raw = json.loads((Path(__file__).parent / "fixtures/final_dspark/config.json").read_text())
    c = raw["transformer_layer_config"]
    h, intermediate, vocabulary = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    layers, q, kv, dim = (c[k] for k in
                         ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim"))
    rank = raw["markov_rank"]
    # Same BF16 tensor shapes as dspark_weights.weight_shapes, excluding the
    # absent verifier-owned vocabulary matrices. Packed context KV is additional.
    checkpoint = 2 * (len(raw["aux_hidden_state_layer_ids"]) * h * h + 2 * h
        + layers * ((q + 2 * kv) * dim * h + h * q * dim
                    + 2 * dim + 2 * h + 3 * h * intermediate)
        + 2 * vocabulary * rank + h + rank + 1)
    packed_context = 2 * layers * 2 * kv * dim * h
    verifier = 2 * 2 * vocabulary * h
    print(json.dumps(dict(draft_checkpoint_bf16=checkpoint, packed_context_kv=packed_context,
                         verifier_embedding_and_head_bf16=verifier,
                         shared_owner_tensor_floor=checkpoint + packed_context + verifier,
                         duplicate_reference_tensor_floor=2 * checkpoint + packed_context + verifier,
                         excludes="activations, scratch, loading temporaries, tuning, graphs, compiler")))
    assert checkpoint > 0 and packed_context > 0 and verifier == 4697620480


def require_gpu():
    assert os.environ.get("FLARE_RUN_CUDA_TESTS") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Ops must explicitly assign one GPU"
    assert "," not in os.environ["CUDA_VISIBLE_DEVICES"], "single-device component only"
    print(json.dumps(preflight(), sort_keys=True))
    assert "speculators" not in sys.modules, "canonical imports belong to the isolated subprocess"


def compare(name, actual, expected, records):
    import torch
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    delta = actual.float() - expected.float()
    records[name] = dict(max_abs=delta.abs().max().item() if delta.numel() else 0,
                         rms=delta.square().mean().sqrt().item() if delta.numel() else 0,
                         exact=torch.equal(actual, expected))


@CUDA
@pytest.mark.parametrize("context", [0, 1, 2047, 2048, 2049])
@pytest.mark.parametrize("block", [1, 2, 3, 4, 5, 6])
def test_synthetic_five_layer_backbone(context, block, tmp_path):
    require_gpu()
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3DecoderLayer
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.paged_state import PagedStateStorage
    from ignition.components.state import StateLayout
    from ignition.components.weights import MLPWeights
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dspark import PreparedContext, PreparedDraft, DraftScratch
    from ignition.recipes.kimi3.dspark_weights import DraftWeights, DraftLayerWeights

    with torch.no_grad():
        torch.manual_seed(3105)
        device, dtype = "cuda:0", torch.bfloat16
        cfg = Qwen3Config(hidden_size=256, intermediate_size=512, num_hidden_layers=5,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                         vocab_size=256, attention_bias=False, rms_norm_eps=1e-5,
                         sliding_window=2048, rope_theta=10000.,
                         layer_types=["sliding_attention"] * 5)
        cfg._attn_implementation = "eager"
        d = NS(text=cfg.to_dict(), heads=4, kv_heads=2, head_dim=64, layers=5,
               window=2048, theta=10000., markov_rank=64)
        config = NS(hidden=256, vocabulary=256, draft=d)
        shard = TensorParallelShard(0, 1)
        # These modules supply synthetic weight shapes only. The engine forward
        # is PreparedDraft; the canonical forward runs in the reference process.
        layers = [Qwen3DecoderLayer(cfg, i).to(device, dtype).eval() for i in range(5)]
        fc = torch.nn.Linear(5 * 256, 256, bias=False).to(device, dtype)
        hn = Qwen3RMSNorm(256, eps=cfg.rms_norm_eps).to(device, dtype)
        norm = Qwen3RMSNorm(256, eps=cfg.rms_norm_eps).to(device, dtype)
        embedding = torch.nn.Embedding(256, 256).to(device, dtype)
        head = torch.nn.Linear(256, 256, bias=False).to(device, dtype)
        packed, kv = [], []
        for layer in layers:
            a, m = layer.self_attn, layer.mlp
            kv.extend((a.k_proj.weight, a.v_proj.weight))
            packed.append(DraftLayerWeights(
                torch.cat((a.q_proj.weight, a.k_proj.weight, a.v_proj.weight)),
                a.o_proj.weight, a.q_norm.weight, a.k_norm.weight,
                layer.input_layernorm.weight, layer.post_attention_layernorm.weight,
                MLPWeights(torch.cat((m.gate_proj.weight, m.up_proj.weight)), m.down_proj.weight)))
        draft_weights = DraftWeights(fc.weight, hn.weight, torch.cat(kv), norm.weight,
                                     embedding.weight, None,
                                     torch.zeros(256, 64, device=device, dtype=dtype),
                                     torch.zeros(256, 64, device=device, dtype=dtype), tuple(packed))
        weights = NS(config=config, shard=shard, draft=draft_weights, head=head.weight)
        layout = StateLayout.from_specs([
            (f"draft{i}.{kind}", (2048, 2, 64), dtype)
            for i in range(5) for kind in ("key", "value")])
        pool = PagedStateStorage.allocate(layout, [], 2, 1, device)
        auxiliary = torch.randn(max(1, context), 1280, device=device, dtype=dtype)
        def state(module):
            return {k: v.detach().cpu() for k, v in module.state_dict().items()}
        input_path, output_path = tmp_path / "synthetic-input.pt", tmp_path / "canonical.safetensors"
        torch.save(dict(config=cfg.to_dict(), context=context, layers=[state(x) for x in layers],
            fc=state(fc), hn=state(hn), norm=state(norm), embedding=state(embedding),
            head=state(head), auxiliary=auxiliary.cpu()), input_path)
        result = subprocess.run([
            str(REFERENCE_PYTHON), "-B", str(REFERENCE_HELPER),
            "--input", str(input_path), "--output", str(output_path)],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            capture_output=True, text=True)
        (tmp_path / "canonical-process.log").write_text(result.stdout + result.stderr)
        print(result.stdout)
        assert result.returncode == 0, result.stderr
        from safetensors.torch import load_file
        expected = load_file(str(output_path), device=device)
        cm = NS(positions=torch.arange(max(1, context), device=device),
                starts=torch.zeros(1, dtype=torch.long, device=device),
                requests=torch.zeros(max(1, context), dtype=torch.long, device=device),
                state_rows=torch.ones(1, dtype=torch.long, device=device))
        counts = torch.tensor([context], device=device)
        tokens = torch.tensor([3] + [255] * (block - 1), device=device)
        dm = NS(token_ids=tokens, positions=torch.arange(context, context + block, device=device),
                starts=torch.tensor([context], device=device),
                lengths=torch.tensor([block], dtype=torch.int32, device=device),
                state_rows=cm.state_rows, sample_indices=torch.arange(block, device=device),
                query_start_loc=torch.tensor([0, block], dtype=torch.int32, device=device),
                live_counts=torch.tensor([block], dtype=torch.int32, device=device))
        scratch = DraftScratch.allocate(config, shard, block, block, device)
        records, actual_layers = {}, {}
        with RecordedPreparationSession(
            device=torch.device(device), checkpoint="synthetic-five-layer-gqa",
            workload_schema=f"draft-only-{block}", numerical_policy="bf16-diagnostic-exact",
            compile_workers=1,
        ) as session:
            common = dict(session=session, workspace=None, runtime=None, coordinator=None)
            context_op = PreparedContext(weights, auxiliary, cm, pool, name="context",
                                         buffers={}, **common)
            draft_op = PreparedDraft(weights, scratch, dm, pool, block=block,
                                     vocabulary_runtime=None, embedding_runtime=None,
                                     name="draft", **common)
            session.tune(show_progress=False)
            context_op(auxiliary, counts)
            compare("context.fc", context_op.projected, expected["context.fc"], records)
            compare("context.norm", context_op.normalized, expected["context.norm"], records)
            compare("context.packed_kv", context_op.kv[:context], expected["context.packed_kv"], records)

            # Observe actual module outputs; no replacement forwards or attention.
            handles = []
            for i, layer in enumerate(draft_op.layers):
                def observe(module, args, output, index=i):
                    actual_layers[index] = (output.clone(), scratch.qkv.clone(),
                                            scratch.heads_output.clone())
                handles.append(layer.register_forward_hook(observe))
            draft_op()
            for handle in handles:
                handle.remove()
            first = scratch.logits.clone()
            draft_op()
            assert torch.equal(first, scratch.logits), "prepared repeatability"

            for i in range(5):
                compare(f"layer{i}.qkv", actual_layers[i][1], expected[f"layer{i}.qkv"][:block], records)
                compare(f"layer{i}.attended", actual_layers[i][2], expected[f"layer{i}.attended"][:block], records)
                compare(f"layer{i}.hidden", actual_layers[i][0], expected[f"layer{i}.hidden"][:block], records)
            compare("head.normalized", scratch.hidden, expected["head.normalized"][:block], records)
            compare("head.logits", scratch.logits, expected["head.logits"][:block], records)
            print(json.dumps(dict(context=context, block=block, reference_block=7,
                                  comparisons=records), sort_keys=True))
            # Exactness is reported, not used as a blanket numerical contract.
            # No cross-kernel acceptance envelope has been justified yet.
            pytest.fail("diagnostic completed; numerical qualification remains blocked on "
                        "reviewed BF16 projection/reduction and padded-attention bounds. "
                        "Do not derive acceptance by widening tolerances to this result.")
