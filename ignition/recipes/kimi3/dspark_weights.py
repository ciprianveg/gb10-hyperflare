# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sharded Kimi K3 DSpark draft: Qwen3 sliding-window layers, context projections and the Markov head."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint


def weight_shapes(draft):
    """Checkpoint shapes before TP slicing; per-head norm vectors replicate."""
    hidden, intermediate = draft.text["hidden_size"], draft.text["intermediate_size"]
    vocabulary = draft.text["vocab_size"]
    shapes = {"fc.weight": (hidden, len(draft.auxiliary_layers) * hidden),
        "hidden_norm.weight": (hidden,), "norm.weight": (hidden,),
        "embed_tokens.weight": (vocabulary, hidden), "lm_head.weight": (vocabulary, hidden),
        "markov_head.markov_w1.weight": (vocabulary, draft.markov_rank),
        "markov_head.markov_w2.weight": (vocabulary, draft.markov_rank)}
    for index in range(draft.layers):
        prefix = f"layers.{index}."
        shapes.update({prefix + name + ".weight": shape for name, shape in {
            "self_attn.q_proj": (draft.heads * draft.head_dim, hidden),
            "self_attn.k_proj": (draft.kv_heads * draft.head_dim, hidden),
            "self_attn.v_proj": (draft.kv_heads * draft.head_dim, hidden),
            "self_attn.o_proj": (hidden, draft.heads * draft.head_dim),
            "self_attn.q_norm": (draft.head_dim,), "self_attn.k_norm": (draft.head_dim,),
            "input_layernorm": (hidden,), "post_attention_layernorm": (hidden,),
            "mlp.gate_proj": (intermediate, hidden), "mlp.up_proj": (intermediate, hidden),
            "mlp.down_proj": (hidden, intermediate)}.items()})
    if draft.raw.get("enable_confidence_head") is True:
        shapes.update({"confidence_head.proj.weight": (1, hidden + draft.markov_rank),
                       "confidence_head.proj.bias": (1,)})
    return shapes


def weight_extents(draft, shard):
    """Actual loader slices; vectors and the Markov input embedding are replicated."""
    draft.validate_shard(shard)
    extents = {name: (slice(None),) * len(shape) for name, shape in weight_shapes(draft).items()}
    def partition(name, axis, size, alignment):
        value = list(extents[name])
        value[axis] = slice(*shard.partition(size, alignment=alignment))
        extents[name] = tuple(value)
    hidden, intermediate, vocabulary = (draft.text[k] for k in ("hidden_size", "intermediate_size", "vocab_size"))
    partition("fc.weight", 0, hidden, 64)
    partition("embed_tokens.weight", 1, hidden, 64)
    partition("lm_head.weight", 0, vocabulary, 64)
    partition("markov_head.markov_w2.weight", 0, vocabulary, 64)
    for index in range(draft.layers):
        prefix = f"layers.{index}."
        for name, heads in (("q_proj", draft.heads), ("k_proj", draft.kv_heads), ("v_proj", draft.kv_heads)):
            partition(prefix + "self_attn." + name + ".weight", 0, heads * draft.head_dim, draft.head_dim)
        partition(prefix + "self_attn.o_proj.weight", 1, draft.heads * draft.head_dim, draft.head_dim)
        for name in ("gate_proj", "up_proj"):
            partition(prefix + "mlp." + name + ".weight", 0, intermediate, 32)
        partition(prefix + "mlp.down_proj.weight", 1, intermediate, 32)
    return extents


def validate_checkpoint(manifest, draft):
    from ignition.recipes.kimi31.gqa_dspark_config import DSparkGQAConfig
    shapes = weight_shapes(draft)
    optional = {"embed_tokens.weight", "lm_head.weight"}
    missing = set(shapes) - set(manifest.tensors)
    if isinstance(draft, DSparkGQAConfig) and missing != optional:
        raise ValueError("Kimi K3.1 GQA DSpark requires verifier-owned absent embedding and head")
    if missing - optional or set(manifest.tensors) - set(shapes):
        raise ValueError("DSpark checkpoint tensor names differ from validated geometry")
    if missing and not draft.raw.get("speculators_config", {}).get("verifier"):
        raise ValueError("absent draft vocabulary weights require a declared verifier")
    for name, tensor in manifest.tensors.items():
        if tensor.dtype != "BF16" or tuple(tensor.shape) != shapes[name]:
            raise ValueError(f"DSpark checkpoint requires BF16 {name} with shape {shapes[name]}")
    return missing


def target_vocabulary_weights(config, shard, device, target_embedding, target_head):
    hidden = config.hidden
    column = shard.partition(hidden, alignment=64)
    row = shard.partition(config.vocabulary, alignment=64)
    selected = torch.device(device)
    for name, value, shape in (("embedding", target_embedding, (config.vocabulary, column[1] - column[0])),
                              ("head", target_head, (row[1] - row[0], hidden))):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16
                or tuple(value.shape) != shape or value.device.type != selected.type
                or selected.index is not None and value.device != selected):
            raise ValueError(f"DSpark verifier {name} must be BF16 with shape {shape} on {device}")
    if target_embedding.device != target_head.device:
        raise ValueError("DSpark verifier embedding and head must share their device")
    return target_embedding, target_head


@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: torch.Tensor
    output: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class DraftWeights:
    """One rank's draft weights.

    ``projection`` holds this rank's rows of the auxiliary-state combination, gathered across ranks;
    ``context_kv`` stacks every layer's key and value rows, so accepted target states reach all draft
    caches through one projection. The embedding is column-sharded like the target's. The output head
    is the target's when the checkpoint's head is identical, and the Markov transition keeps its
    embedding replicated while its projection is vocabulary-sharded like the head. A plan storing
    online MXFP8 holds the combination, attention outputs and MLPs as MXFP8; queries, keys, values,
    the embedding and the Markov transition keep the checkpoint's BF16.
    """

    projection: torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: torch.Tensor
    norm: torch.Tensor
    embedding: torch.Tensor
    head: torch.Tensor | None
    markov_embedding: torch.Tensor
    markov_head: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]
    confidence_head: torch.Tensor | None = None
    confidence_bias: torch.Tensor | None = None

    @classmethod
    def load(cls, snapshot, config, device, shard, *, target_head, target_embedding=None,
             progress=None, online_mxfp8=False):
        from ignition.components.weights import quantize_bf16_linear_as_mxfp8 as mxfp8_rows

        stored = mxfp8_rows if online_mxfp8 else lambda weight: weight
        d = config.draft
        if d.raw.get("enable_confidence_head") is True and online_mxfp8:
            raise ValueError("confidence-bearing DSpark qualification requires checkpoint BF16 weights")
        d.validate_shard(shard)
        c, hidden = d.text, config.hidden
        manifest = inspect_indexed_checkpoint(snapshot)
        missing = validate_checkpoint(manifest, d)
        extents = weight_extents(d, shard)
        if missing:
            target_embedding, target_head = target_vocabulary_weights(
                config, shard, device, target_embedding, target_head)
        layers, context = [], []
        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            def matrix(name):
                value = loader.load_tensor(name + ".weight", extent=extents[name + ".weight"])
                if value.dtype != torch.bfloat16 or value.ndim != 2:
                    raise ValueError(f"Kimi K3 DSpark requires BF16 matrices: {name}")
                return value

            projection = matrix("fc")
            if projection.shape[1] != len(d.auxiliary_layers) * hidden:
                raise ValueError("DSpark fc must combine one hidden state per auxiliary layer")
            projection = stored(projection)
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            for index in range(d.layers):
                prefix = f"layers.{index}"
                key = matrix(prefix + ".self_attn.k_proj")
                value = matrix(prefix + ".self_attn.v_proj")
                qkv = torch.cat((matrix(prefix + ".self_attn.q_proj"), key, value))
                context.extend((key, value))
                output = stored(matrix(prefix + ".self_attn.o_proj"))
                query_norm, key_norm, input_norm, post_norm = loader.load_tensors(tuple(prefix + name for name in
                    (".self_attn.q_norm.weight", ".self_attn.k_norm.weight", ".input_layernorm.weight",
                     ".post_attention_layernorm.weight")))
                mlp = MLPWeights(stored(torch.cat([matrix(prefix + ".mlp." + name)
                                                    for name in ("gate_proj", "up_proj")])),
                                 stored(matrix(prefix + ".mlp.down_proj")))
                layers.append(DraftLayerWeights(qkv, output, query_norm, key_norm, input_norm, post_norm, mlp))
            embedding = target_embedding if "embed_tokens.weight" in missing else matrix("embed_tokens")
            head = None if "lm_head.weight" in missing else matrix("lm_head")
            # Kimi K3's DSpark draft reuses the target's output head; keep a copy only for drafts that differ.
            head = None if head is None or torch.equal(head, target_head) else head
            markov_embedding = matrix("markov_head.markov_w1")
            markov_head = matrix("markov_head.markov_w2")
            if markov_embedding.shape[1] != d.markov_rank or markov_head.shape[1] != d.markov_rank:
                raise ValueError("DSpark Markov projections disagree with markov_rank")
            confidence, bias = None, None
            if d.raw.get("enable_confidence_head") is True:
                confidence = matrix("confidence_head.proj")
                bias = loader.load_tensor("confidence_head.proj.bias")
        return cls(projection, hidden_norm, torch.cat(context), norm, embedding, head, markov_embedding,
                   markov_head, tuple(layers), confidence, bias)
