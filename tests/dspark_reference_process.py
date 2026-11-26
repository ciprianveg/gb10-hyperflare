"""Pinned canonical computation in its own Transformers environment."""
import argparse
import json
import os
from pathlib import Path
import sys

SOURCE = Path("/tmp/opencode/k31-speculators-05f9ba2324fe-reference/repo")


def imports():
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "hs_connectors/src")]
    import torch
    import transformers
    import speculators
    import triton
    import pydantic
    assert transformers.__version__ == "5.19.0"
    assert torch.__version__ == "2.14.1+cu132"
    assert str(Path(torch.__file__).resolve()).startswith("/home/ciprian/k3-src/flare-k31/.venv/")
    assert Path(speculators.__file__).resolve().is_relative_to(SOURCE)
    assert not torch.cuda.is_initialized()
    return torch, dict(transformers=transformers.__version__, torch=torch.__version__,
                      triton=triton.__version__, pydantic=pydantic.__version__,
                      torch_file=torch.__file__, transformers_file=transformers.__file__,
                      speculators_file=speculators.__file__, cuda_initialized=False)


def forward(input_path, output_path):
    assert os.environ.get("FLARE_RUN_CUDA_TESTS") == "1", "separate GPU release required"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Ops device assignment required"
    torch, versions = imports()
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding
    from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
    from speculators.models.dflash.attention import create_anchor_block_mask_mod
    from speculators.models.attention import create_float_mask
    from safetensors.torch import save_file

    data = torch.load(input_path, map_location="cpu", weights_only=True)
    cfg = Qwen3Config(**data["config"])
    cfg._attn_implementation = "eager"
    context, device, dtype = data["context"], "cuda:0", torch.bfloat16
    with torch.no_grad():
        layers = [Qwen3DFlashDecoderLayer(cfg, i).to(device, dtype).eval() for i in range(5)]
        for layer, state in zip(layers, data["layers"], strict=True):
            layer.load_state_dict(state, strict=True)
        fc = torch.nn.Linear(1280, 256, bias=False).to(device, dtype)
        hn = Qwen3RMSNorm(256, eps=cfg.rms_norm_eps).to(device, dtype)
        norm = Qwen3RMSNorm(256, eps=cfg.rms_norm_eps).to(device, dtype)
        embedding = torch.nn.Embedding(256, 256).to(device, dtype)
        head = torch.nn.Linear(256, 256, bias=False).to(device, dtype)
        for name, module in (("fc", fc), ("hn", hn), ("norm", norm),
                             ("embedding", embedding), ("head", head)):
            module.load_state_dict(data[name], strict=True)
        auxiliary = data["auxiliary"].to(device)
        rotary = Qwen3RotaryEmbedding(cfg).to(device)
        mod, qlen, klen = create_anchor_block_mask_mod(
            torch.zeros(context + 1, dtype=torch.long, device=device), context + 1,
            torch.tensor([context], device=device), 7, True, 2048)
        mask = create_float_mask(mod, B=1, H=1, Q_LEN=qlen, KV_LEN=klen, device=device, dtype=dtype)
        mask = torch.cat((mask[..., :context], mask[..., context + 1:]), -1)
        positions = torch.arange(context + 7, device=device).unsqueeze(0)
        tokens = torch.tensor([3] + [255] * 6, device=device)

        def run():
            out = {}
            projected = fc(auxiliary)
            target = hn(projected)[:context].unsqueeze(0)
            out["context.fc"] = projected
            out["context.norm"] = hn(projected)
            out["context.packed_kv"] = torch.cat([
                p(target) for layer in layers
                for p in (layer.self_attn.k_proj, layer.self_attn.v_proj)], -1).squeeze(0)
            hidden = embedding(tokens).unsqueeze(0)
            rope = rotary(hidden, positions)
            for i, layer in enumerate(layers):
                observed = {}
                from functools import partial

                def save(module, args, output, name):
                    observed.setdefault(name, []).append(output.clone())

                handles = [getattr(layer.self_attn, name).register_forward_hook(partial(save, name=name))
                           for name in ("q_proj", "k_proj", "v_proj")]
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                    lambda module, args: observed.update(attended=args[0].clone())))
                hidden = layer(hidden_states=hidden, target_hidden=target,
                               position_embeddings=rope, attention_mask=mask)
                for handle in handles:
                    handle.remove()
                out[f"layer{i}.qkv"] = torch.cat([
                    observed[name][-1] for name in ("q_proj", "k_proj", "v_proj")], -1).squeeze(0)
                out[f"layer{i}.attended"] = observed["attended"].squeeze(0)
                out[f"layer{i}.hidden"] = hidden.squeeze(0)
            out["head.normalized"] = norm(hidden).squeeze(0)
            out["head.logits"] = head(out["head.normalized"])
            return {k: v.cpu().contiguous() for k, v in out.items()}

        first, second = run(), run()
        repeatability = {k: (first[k].float() - second[k].float()).abs().max().item()
                         if first[k].numel() else 0. for k in first}
        save_file(first, output_path)
        print(json.dumps(dict(environment=versions, repeatability_max_abs=repeatability)))
        assert all(v == 0 for v in repeatability.values()), "canonical repeatability requires review"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.preflight:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        _, versions = imports()
        print(json.dumps(versions, sort_keys=True))
    else:
        assert args.input and args.output
        forward(args.input, args.output)
