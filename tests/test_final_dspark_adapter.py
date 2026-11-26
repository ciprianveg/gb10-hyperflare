"""CPU metadata/partition checks, not draft forward or captured-replay qualification."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.state import StateStorage
from ignition.preparation.checkpoint import TensorMetadata
from ignition.recipes.kimi3.config import DSparkConfig, draft_geometry
from ignition.recipes.kimi3.dspark_weights import (
    target_vocabulary_weights, validate_checkpoint, weight_extents, weight_shapes,
)
from ignition.recipes.kimi3.pool import continuation_layout, page_bytes

FIXTURE = Path(__file__).parent / "fixtures/final_dspark"
TARGET = dict(hidden_size=7168, vocab_size=163840, num_hidden_layers=93,
              num_attention_heads=96, max_position_embeddings=1048576)


def config():
    return DSparkConfig.load(FIXTURE, NS(text=TARGET))


def manifest():
    report = json.loads((FIXTURE / "header-report.json").read_text())
    return NS(tensors={name: TensorMetadata(name, "not-opened.safetensors", item["byte_range"][0],
        item["byte_range"][1] - item["byte_range"][0], tuple(item["shape"]), item["dtype"])
        for name, item in report["tensors"].items()})


def test_actual_config_reader_and_kimi31_prefix_contract():
    d = config()
    assert (d.layers, d.heads, d.kv_heads, d.head_dim, d.window, d.theta) == (5, 96, 16, 64, 2048, 10000.)
    assert d.block_size == 7 and d.auxiliary_layers == (76, 80, 84, 88, 92)
    assert draft_geometry(d) == "gqa"
    assert draft_geometry(None) is None
    with pytest.raises(ValueError):
        draft_geometry(NS())
    assert d.auxiliary_stream == "prefix"
    old = copy.deepcopy(d.raw)
    old["enable_confidence_head"] = False
    old["speculators_config"]["verifier"]["architectures"] = ["KimiForCausalLM"]
    assert DSparkConfig.from_dict(old, TARGET).auxiliary_stream == "attn_res"
    for proposals in range(7):
        d.validate_proposals(proposals)
        assert proposals + 1 <= 7
    with pytest.raises(ValueError):
        d.validate_proposals(8)
    assert not torch.cuda.is_initialized()


def test_preview_and_dflash_dispatch_keep_distinct_validators():
    preview = dict(architectures=["K3DSparkModel"], model_type="k3_dspark",
        hidden_act="silu", hidden_size=7168, intermediate_size=14336,
        num_hidden_layers=5, num_attention_heads=96, num_key_value_heads=96,
        q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
        qk_rope_head_dim=64, qk_head_dim=192, v_head_dim=128, head_dim=64,
        attention_bias=False, attention_dropout=0.0, rms_norm_eps=1e-5,
        rope_interleave=True, rope_theta=1_000_000.0,
        layer_types=["full_attention"] * 5, sliding_window=None,
        use_sliding_window=False, full_attention_causal=True,
        sliding_window_non_causal=False, sample_from_anchor=True,
        vocab_size=163840, draft_vocab_size=163840, mask_token_id=163839,
        markov_rank=512, markov_head_type="vanilla", block_size=7,
        num_target_layers=5, target_hidden_size=7168, target_num_hidden_layers=93,
        max_position_embeddings=1048576, tie_word_embeddings=False, torch_dtype="bfloat16",
        enable_confidence_head=True, confidence_head_with_markov=True,
        target_layer_ids=[75, 79, 83, 87, 91], dflash_config={"causal": True},
        rope_parameters=dict(rope_type="yarn", rope_theta=1_000_000.0, factor=32.0,
            original_max_position_embeddings=32768, beta_fast=32, beta_slow=1,
            mscale=1.0, mscale_all_dim=0.0, attn_factor=0.7426255848312643))
    parsed = DSparkConfig.from_dict(preview, TARGET)
    assert draft_geometry(parsed) == "mla" and parsed.auxiliary_stream == "prefix"
    assert parsed.auxiliary_layers == config().auxiliary_layers
    from ignition.recipes.kimi31.dspark_config import DSparkConfig as MLA
    with pytest.raises(ValueError):
        MLA.from_dict(config().raw, TARGET)
    dflash = dict(architectures=["DFlash2DraftModel"], model_type="qwen3", dtype="bfloat16",
        hidden_act="silu", attention_bias=False, attention_dropout=0., is_causal=True,
        use_sliding_window=True, tie_word_embeddings=False, mla_use_nope=False,
        rope_interleave=True, q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
        qk_rope_head_dim=64, qk_head_dim=192, v_head_dim=128, hidden_size=7168,
        num_attention_heads=96, vocab_size=163840, draft_vocab_size=163840,
        max_position_embeddings=1048576, num_hidden_layers=5, intermediate_size=14336,
        num_target_layers=93, sliding_window=1024, layer_types=["sliding_attention"] * 5,
        rms_norm_eps=1e-5, dflash_config=dict(attention_mode="mla", aux_hidden_stream="attn_res",
            block_size=5, conv_kernel_size=2, conv_group_size=16, selector_rank=256,
            selector_top_k=32, mask_token_id=163824, target_layer_ids=[7, 23, 51, 67, 83, 91]),
        rope_parameters=dict(rope_type="yarn", factor=32., rope_theta=50000.,
            original_max_position_embeddings=32768, mscale=1., mscale_all_dim=1.,
            beta_fast=32, beta_slow=1))
    parsed = DSparkConfig.from_dict(dflash, TARGET)
    assert draft_geometry(parsed) == "dflash2" and parsed.auxiliary_stream == "attn_res"


@pytest.mark.parametrize("field,value", [
    ("num_key_value_heads", 0), ("num_attention_heads", 95), ("head_dim", 32),
    ("sliding_window", 2049), ("attention_dropout", .1), ("mlp_bias", True),
    ("rope_parameters", {"rope_type": "yarn", "rope_theta": 10000.}),
    ("layer_types", ["full_attention"] * 5),
])
def test_bad_nested_geometry_rejected(field, value):
    raw = copy.deepcopy(config().raw)
    raw["transformer_layer_config"][field] = value
    with pytest.raises(ValueError):
        DSparkConfig.from_dict(raw, TARGET)


def test_header_contract_confidence_and_expected_absent_weights():
    d, m = config(), manifest()
    assert len(m.tensors) == 62
    assert validate_checkpoint(m, d) == {"embed_tokens.weight", "lm_head.weight"}
    assert weight_shapes(d)["confidence_head.proj.weight"] == (1, 7168 + 512)
    assert weight_shapes(d)["confidence_head.proj.bias"] == (1,)
    for removed in ("confidence_head.proj.bias", "layers.0.self_attn.q_norm.weight"):
        bad = NS(tensors=dict(m.tensors)); del bad.tensors[removed]
        with pytest.raises(ValueError):
            validate_checkpoint(bad, d)
    bad = NS(tensors=dict(m.tensors))
    t = bad.tensors["layers.0.self_attn.q_norm.weight"]
    bad.tensors[t.name] = TensorMetadata(t.name, t.shard, t.offset, t.nbytes, (4,), t.dtype)
    with pytest.raises(ValueError):
        validate_checkpoint(bad, d)
    bad = NS(tensors=dict(m.tensors))
    bad.tensors["embed_tokens.weight"] = TensorMetadata("embed_tokens.weight", "not-opened", 0, 0,
                                                        (163840, 7168), "BF16")
    with pytest.raises(ValueError, match="verifier-owned"):
        validate_checkpoint(bad, d)


def test_tp16_actual_loader_slices_reconstruct_every_partition_axis():
    d = config()
    shapes = weight_shapes(d)
    extents = [weight_extents(d, TensorParallelShard(rank, 16)) for rank in range(16)]
    for name, shape in shapes.items():
        for axis, length in enumerate(shape):
            slices = [e[name][axis] for e in extents]
            whole = torch.arange(length)
            if slices[0] == slice(None):
                assert all(torch.equal(whole[s], whole) for s in slices)
            else:
                assert torch.equal(torch.cat([whole[s] for s in slices]), whole), (name, axis)
        local = torch.empty(shape, device="meta")[extents[0][name]]
        if name.endswith(("q_norm.weight", "k_norm.weight")):
            assert local.shape == (64,)
    assert torch.empty(shapes["layers.0.self_attn.q_proj.weight"], device="meta")[
        extents[0]["layers.0.self_attn.q_proj.weight"]].shape == (384, 7168)
    assert torch.empty(shapes["layers.0.self_attn.k_proj.weight"], device="meta")[
        extents[0]["layers.0.self_attn.k_proj.weight"]].shape == (64, 7168)


def test_verifier_owners_retained_and_shapes_checked_without_payload():
    d = config()
    target = NS(hidden=7168, vocabulary=163840, draft=d)
    embedding = torch.empty((163840, 448), device="meta", dtype=torch.bfloat16)
    head = torch.empty((10240, 7168), device="meta", dtype=torch.bfloat16)
    a, b = target_vocabulary_weights(target, TensorParallelShard(0, 16), "meta", embedding, head)
    assert a is embedding and b is head
    with pytest.raises(ValueError):
        target_vocabulary_weights(target, TensorParallelShard(0, 16), "meta", embedding, head.float())
    with pytest.raises(ValueError):
        target_vocabulary_weights(target, TensorParallelShard(0, 16), "meta", None, head)


def test_confidence_declared_as_weights_and_predecessors_retained_per_position():
    from ignition.recipes.kimi3.dspark_weights import DraftWeights
    from ignition.preparation.rebuild_manifest import declare_weights, declared_views
    from ignition.components.split_draft import PreparedDraftConfidence
    scalar = torch.ones((1,), dtype=torch.bfloat16)
    draft = DraftWeights(scalar, scalar, scalar, scalar, scalar, None, scalar, scalar, (),
                         torch.ones((1, 7680), dtype=torch.bfloat16), scalar)
    declaration = declare_weights({"draft": draft}, components=("draft",))
    names = {name for name, *_ in declared_views(declaration)}
    assert "draft.confidence_head" in names and "draft.confidence_bias" in names
    # Exercise only the existing CPU copy method, not an unqualified projection.
    observer = object.__new__(PreparedDraftConfidence)
    observer.canonical_embedding = None
    observer.embeddings = torch.empty((2, 6, 512), dtype=torch.bfloat16)
    reused = torch.empty((2, 512), dtype=torch.bfloat16)
    for position in range(6):
        reused.fill_(position + 1)
        observer.record_embedding(position, reused)
    reused.fill_(-1)
    for position in range(6):
        assert torch.equal(observer.embeddings[:, position],
                           torch.full_like(reused, position + 1))
    assert not torch.cuda.is_initialized()


def test_actual_descriptor_accounting_ring_and_target_pages():
    d = config()
    target = NS(draft=d, hidden=7168, text={"linear_attn_config": {"num_heads": 64}},
                layers=("dsa",) * 24, dsa=NS(head_dim=128))
    assert page_bytes(target) == 24 * 128 * (576 + 132) == 2174976
    layout = continuation_layout(target, 16, speculative_tokens=6)
    rings = [r for r in layout.regions if r.name.startswith("draft")]
    assert len(rings) == 10
    assert sum(r.nbytes for r in rings) == 2621440
    assert sum(r.nbytes for r in rings) * 40 == 100 << 20
    assert all(r.shape == (2048, 1, 64) and r.dtype == torch.bfloat16 for r in rings)
    # Actual CPU views share a single allocation. Count backing once, not each alias.
    storage = StateStorage.allocate(layout, 2, "cpu")
    views = [storage.region(r.name) for r in rings]
    unique = {view.untyped_storage().data_ptr(): view.untyped_storage().nbytes() for view in views}
    assert list(unique.values()) == [storage.backing.nbytes]
    assert storage.backing.nbytes == 2 * layout.row_bytes
    assert not torch.cuda.is_initialized()
