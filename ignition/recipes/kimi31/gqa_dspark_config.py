# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1's causal Qwen3 DSpark over plain target residual boundaries."""
from ignition.recipes.kimi3.config import DSparkConfig

DSPARK_REPOSITORY = "Inferact/Kimi-K3.1-Pro-DSpark"
DSPARK_REVISION = "e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f"
DSPARK_DIRECTORY = "dspark-final"


class DSparkGQAConfig(DSparkConfig):
    def validate_storage_precision(self, *, online_mxfp8, online_mxfp8_scope):
        if online_mxfp8 and online_mxfp8_scope == "all":
            raise ValueError("Kimi K3.1 GQA DSpark requires BF16 draft storage; use target-only kda-input scope")

    @classmethod
    def from_dict(cls, raw, target):
        result = super().from_dict(raw, target)
        expected = dict(block_size=7, sample_from_anchor=True, full_attention_causal=True,
            sliding_window_non_causal=False, mask_token_id=163839, draft_vocab_size=163840,
            markov_rank=512, markov_head_type="vanilla", enable_confidence_head=True,
            confidence_head_with_markov=True, dtype="bfloat16", tie_word_embeddings=False,
            aux_hidden_state_layer_ids=[76, 80, 84, 88, 92])
        nested = dict(hidden_size=7168, intermediate_size=14336, num_hidden_layers=5,
            num_attention_heads=96, num_key_value_heads=16, head_dim=64, sliding_window=2048,
            max_position_embeddings=1048576, rms_norm_eps=1e-5,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
        for values, required in ((raw, expected), (result.text, nested)):
            for name, value in required.items():
                if type(values.get(name)) is not type(value) or values[name] != value:
                    raise ValueError(f"unsupported Kimi K3.1 GQA DSpark {name}")
        if target.get("num_hidden_layers") != 93:
            raise ValueError("Kimi K3.1 GQA DSpark requires 93 target layers")
        return result

    @property
    def auxiliary_stream(self):
        # vLLM c5993c0471bd captures prefix_sum + pending MLP at layer_idx + 1.
        # Checkpoint aux IDs name those boundaries directly, not preview layer IDs.
        return "prefix"
