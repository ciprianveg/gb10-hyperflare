"""CPU path selection, policy and descriptor budgets; no model payloads or GPU."""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from ignition import planning
from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import MLPWeights
from ignition.preparation.checkpoint_source import RootSource
from ignition.preparation.rebuild_manifest import declare_weights, declared_views
from ignition.recipes import resolve
from ignition.recipes.kimi3.config import ModelConfig
from ignition.recipes.kimi3.dspark_weights import DraftLayerWeights, DraftWeights, weight_shapes, weight_extents
from ignition.recipes.kimi3.pool import continuation_layout, page_bytes
from ignition.recipes.kimi3.recipe import _draft_checkpoint
from ignition.recipes.kimi31 import recipe
from ignition.recipes.kimi31.gqa_dspark_config import (
    DSPARK_DIRECTORY, DSPARK_REPOSITORY, DSPARK_REVISION, DSparkGQAConfig,
)

FIXTURE = Path(__file__).parent / "fixtures/final_dspark"


def target():
    return ModelConfig.from_dict(json.loads((FIXTURE / "target-config.json").read_text()))


def final():
    return DSparkGQAConfig.load(FIXTURE, target())


def small_plan(scope="kda-input"):
    return NS(context_parallel=True, speculation=replace(recipe.RECIPE.speculation, draft_tokens=6),
              online_mxfp8=True, online_mxfp8_scope=scope, hub_cache=None)


def stage_metadata(root, final_config=True):
    root.mkdir()
    (root / "config.json").write_text((FIXTURE / "target-config.json").read_text())
    (root / "dspark").mkdir()
    (root / "dspark/config.json").write_text('{"architectures":["K3DSparkModel"]}')
    if final_config:
        (root / DSPARK_DIRECTORY).mkdir()
        (root / DSPARK_DIRECTORY / "config.json").write_text((FIXTURE / "config.json").read_text())


def test_preview_directory_cannot_satisfy_final_default(tmp_path):
    root = tmp_path / "target"
    stage_metadata(root, final_config=False)
    with pytest.raises(ValueError, match="preview dspark/ is not a substitute"):
        recipe.prepare(root, small_plan(), compile_forwards=False, kernel_cache=None, progress=None)


def test_final_config_selected_before_gpu_preparation(tmp_path, monkeypatch):
    root = tmp_path / "target"
    stage_metadata(root)
    reached = []
    def preparation_boundary(snapshot, plan, **kwargs):
        # Only GPU preparation is replaced; both production path selectors execute.
        selected = _draft_checkpoint(snapshot, plan, ranks=kwargs["ranks"],
            draft_source=kwargs["draft_source"], draft_directory=kwargs["draft_directory"])
        reached.append((selected, kwargs["draft_source"]))
        return selected
    monkeypatch.setattr(recipe, "prepare_kimi", preparation_boundary)
    result = recipe.prepare(root, small_plan(), compile_forwards=False, kernel_cache=None,
                            progress=None, ranks=NS(hosts=("host",) * 16))
    assert result == str(root / DSPARK_DIRECTORY)
    assert reached == [(str(root / DSPARK_DIRECTORY), (DSPARK_REPOSITORY, DSPARK_REVISION))]
    (root / DSPARK_DIRECTORY / "config.json").write_text((root / "dspark/config.json").read_text())
    with pytest.raises(ValueError):
        recipe.prepare(root, small_plan(), compile_forwards=False, kernel_cache=None, progress=None)
    assert len(reached) == 1


def test_precision_request_is_rejected_not_silently_changed(tmp_path, monkeypatch):
    root = tmp_path / "target"
    stage_metadata(root)
    with pytest.raises(ValueError, match="BF16 draft storage"):
        recipe.prepare(root, small_plan("all"), compile_forwards=False, kernel_cache=None, progress=None)
    parsed = final()
    with pytest.raises(ValueError, match="checkpoint BF16"):
        DraftWeights.load("not-opened", replace(target(), draft=parsed), "cpu",
                          TensorParallelShard(0, 16), target_head=None, online_mxfp8=True)
    parsed.validate_storage_precision(online_mxfp8=True, online_mxfp8_scope="kda-input")
    parsed.validate_storage_precision(online_mxfp8=False, online_mxfp8_scope="all")


def test_no_spec_and_dflash_keep_their_selection(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(recipe, "prepare_kimi", lambda snapshot, plan, **kw: received.append(kw))
    plan = small_plan()
    plan.speculation = None
    recipe.prepare(tmp_path / "absent", plan, compile_forwards=False, kernel_cache=None, progress=None)
    plan.speculation = replace(recipe.RECIPE.speculation, method="dflash2")
    recipe.prepare(tmp_path / "absent", plan, compile_forwards=False, kernel_cache=None, progress=None)
    from ignition.recipes.kimi31.dflash_config import REPOSITORY, REVISION
    assert received[-1]["draft_directory"] == "dflash2"
    assert received[-1]["draft_source"] == (REPOSITORY, REVISION)


def test_file_aliases_enter_real_source_catalog_without_copying_payload(tmp_path):
    owner = tmp_path / "owner"; owner.mkdir()
    (owner / "config.json").write_text((FIXTURE / "config.json").read_text())
    header = json.dumps({"tiny": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    payload = len(header).to_bytes(8, "little") + header + b"\x80\x3f"
    (owner / "model.safetensors").write_bytes(payload)
    root = tmp_path / "target"; root.mkdir()
    selected = root / DSPARK_DIRECTORY; selected.mkdir()
    for name in ("config.json", "model.safetensors"):
        (selected / name).symlink_to(owner / name)
    (root / "directory-alias").symlink_to(owner, target_is_directory=True)
    source = RootSource(root, NS(repository="synthetic", revision="cpu-test"))
    try:
        names = {entry["path"] for entry in source.catalog["files"]}
        assert names == {f"{DSPARK_DIRECTORY}/config.json", f"{DSPARK_DIRECTORY}/model.safetensors"}
        assert source.read(f"{DSPARK_DIRECTORY}/model.safetensors", 0, len(payload)) == payload
        assert Path(_draft_checkpoint(root, small_plan(), ranks=NS(),
            draft_source=(DSPARK_REPOSITORY, DSPARK_REVISION), draft_directory=DSPARK_DIRECTORY)) == selected
    finally:
        source.close()


def assembled_meta_weights(draft, world):
    shapes, extents = weight_shapes(draft), weight_extents(draft, TensorParallelShard(0, world))
    def tensor(name):
        shape = tuple(len(range(*part.indices(size))) for size, part in zip(shapes[name], extents[name]))
        return torch.empty(shape, dtype=torch.bfloat16, device="meta")
    layers, context = [], []
    for i in range(draft.layers):
        p = f"layers.{i}."
        key, value = tensor(p + "self_attn.k_proj.weight"), tensor(p + "self_attn.v_proj.weight")
        context.extend((key, value))
        layers.append(DraftLayerWeights(torch.cat((tensor(p + "self_attn.q_proj.weight"), key, value)),
            tensor(p + "self_attn.o_proj.weight"), tensor(p + "self_attn.q_norm.weight"),
            tensor(p + "self_attn.k_norm.weight"), tensor(p + "input_layernorm.weight"),
            tensor(p + "post_attention_layernorm.weight"),
            MLPWeights(torch.cat((tensor(p + "mlp.gate_proj.weight"), tensor(p + "mlp.up_proj.weight"))),
                       tensor(p + "mlp.down_proj.weight"))))
    verifier_embedding = tensor("embed_tokens.weight")
    weights = DraftWeights(tensor("fc.weight"), tensor("hidden_norm.weight"), torch.cat(context),
        tensor("norm.weight"), verifier_embedding, None, tensor("markov_head.markov_w1.weight"),
        tensor("markov_head.markov_w2.weight"), tuple(layers),
        tensor("confidence_head.proj.weight"), tensor("confidence_head.proj.bias"))
    return weights, verifier_embedding


@pytest.mark.parametrize("world,resident,row6,row7", [
    (16, 476635394, 34804480, 35122432), (8, 785309954, 67419392, 68055296)])
def test_actual_target_descriptors_and_unique_retained_weight_owners(world, resident, row6, row7):
    model, draft = target(), final()
    assert model.layers.count("kda") == 69 and model.layers.count("dsa") == 24
    adapted = replace(model, draft=draft)
    assert continuation_layout(adapted, world, speculative_tokens=6).row_bytes == row6
    assert continuation_layout(adapted, world, speculative_tokens=7).row_bytes == row7
    assert recipe.RECIPE.costs(world, recipe.RECIPE.speculation).state_row_bytes == row7
    assert page_bytes(adapted) == recipe.RECIPE.ranks[world].page_bytes == 2174976
    assert continuation_layout(model, world).row_bytes == recipe.RECIPE.plain_ranks[world].state_row_bytes
    weights, verifier = assembled_meta_weights(draft, world)
    owners = {}
    for _, _, tensor, *_ in declared_views(declare_weights({"draft": weights}, components=("draft",))):
        if tensor is not verifier:
            storage = tensor.untyped_storage()
            owners[storage._cdata] = storage.nbytes()
    assert sum(owners.values()) == resident
    # Independent header/slice sum plus the separately retained context-KV packing.
    shapes, extents = weight_shapes(draft), weight_extents(draft, TensorParallelShard(0, world))
    sizes = {name: 2 * math.prod(len(range(*s.indices(n))) for n, s in zip(shape, extents[name]))
             for name, shape in shapes.items() if name not in ("embed_tokens.weight", "lm_head.weight")}
    extra_kv = sum(size for name, size in sizes.items()
                   if name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")))
    assert sum(sizes.values()) + extra_kv == resident
    allowance = recipe.DSPARK_DEVICE_BYTES // world
    assert allowance >= resident
    assert allowance - resident == (0 if world == 16 else 167960834)
    cp = recipe.RECIPE.checkpoints[0]
    target_bytes = 1_491_026_420_236 + 16 * 357_052_416
    assert cp.device_bytes - cp.draft_bytes == target_bytes
    assert recipe.RECIPE.weights_bytes(cp, world, None) == -(-target_bytes // world)
    dflash = replace(recipe.RECIPE.speculation, method="dflash2")
    assert recipe.RECIPE.weights_bytes(cp, world, dflash) == -(-(target_bytes + 5569622016) // world)
    assert recipe.RECIPE.costs(world, dflash).state_row_bytes == (35450112 if world == 16 else 65761536)
    assert not torch.cuda.is_initialized()


def test_actual_recipe_and_plan_serialize_moonshot_policy():
    model = resolve("kimi-k3.1-pro")
    devices = tuple(planning.Gpu(i, "GB10 CPU fixture", str(i), 256 << 30, 256 << 30, (12, 1))
                    for i in range(16))
    plan = planning.plan(model.recipe, model.checkpoint, devices, selected=tuple(range(16)),
        concurrency=2, reserve_bytes=2 << 30, speculation=replace(model.recipe.speculation, draft_tokens=6),
        image_concurrency=2, prefill_chunk=4096)
    assert plan.recipe.sampling_defaults.as_dict() == dict(seed=0, temperature=1., top_p=.95,
        top_k=-1, min_p=0., repetition_penalty=1., presence_penalty=0., frequency_penalty=0.)
    encoded = plan.to_dict()
    assert encoded["speculation"]["draft_tokens"] == 6
    assert json.loads(json.dumps(plan.recipe.sampling_defaults.as_dict()))["top_p"] == .95
    from ignition.preparation.core_proto import serving
    from ignition.preparation.runtime_core_pb2 import ModelIndex
    index = ModelIndex(version=4)
    serving(index, dict(model_id=plan.recipe.served_model_name,
        max_sequence_tokens=plan.context, prefill_chunk_size=plan.prefill_chunk,
        sampling_defaults=plan.recipe.sampling_defaults.as_dict(),
        frontend=dict(tokenizer={}, eos_token_ids=[163586]),
        assets={"tokenizer": {}, "chat_template": {}}))
    assert index.sampling_defaults.temperature == 1.
    assert index.sampling_defaults.top_p == torch.tensor(.95, dtype=torch.float32).item()
    assert not index.sampling_defaults.HasField("ranges")
    assert plan.speculation.draft_tokens == 6 and plan.concurrency == 2 and plan.context == 1048576
    assert plan.speculation.widths == tuple(range(1, 8))
    assert plan.image_concurrency == 2 and plan.prefill_chunk == 4096
    from ignition.recipes.kimi3.recipe import RECIPE as K3
    assert K3.sampling_defaults.top_p == 1.
