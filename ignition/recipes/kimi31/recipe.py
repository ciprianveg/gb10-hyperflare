# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 Pro MXFP4 preparation and serving defaults."""

from types import MappingProxyType
from pathlib import Path

from ignition.recipes.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, Speculation, SpeculativeMethod
from ignition.recipes.kimi3.recipe import RECIPE as KIMI3, prepare as prepare_kimi, usage
from .gqa_dspark_config import DSPARK_DIRECTORY, DSPARK_REPOSITORY, DSPARK_REVISION, DSparkGQAConfig

# Aggregate retained BF16 tensor bytes at TP16, including replicated Markov/norm/
# confidence weights and the additional packed context-KV projection. The shared
# checkpoint estimator divides by world size, so this also upper-bounds TP8.
# Arena alignment, compiler scratch and capture memory remain separate headroom.
DSPARK_DEVICE_BYTES = 7_626_166_304


def final_draft_checkpoint(snapshot, plan):
    """Require the separately staged final draft and validate it before target loading."""
    from ignition.recipes.kimi3.config import ModelConfig
    path = Path(snapshot) / DSPARK_DIRECTORY
    if not (path / "config.json").is_file():
        raise ValueError(f"stage the already downloaded final DSpark under {path}; preview dspark/ is not a substitute")
    draft = DSparkGQAConfig.load(path, ModelConfig.load(snapshot))
    draft.validate_storage_precision(online_mxfp8=plan.online_mxfp8,
                                     online_mxfp8_scope=plan.online_mxfp8_scope)
    return path


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    if not plan.context_parallel:
        raise ValueError("Kimi K3.1 preparation requires context parallelism")
    if plan.speculation is not None and plan.speculation.method == "dflash2":
        from .dflash_config import REPOSITORY, REVISION
        return prepare_kimi(snapshot, plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache,
            progress=progress, retain_graphs=retain_graphs, core_storage=core_storage, ranks=ranks,
            recipe=RECIPE, draft_source=(REPOSITORY, REVISION), draft_directory="dflash2")
    if plan.speculation is not None and (plan.speculation.draft_head_quantization is not None
                                         or plan.speculation.draft_body_quantization is not None):
        raise ValueError("Kimi draft conversion requires --speculative-method dflash2")
    if plan.speculation is not None:
        final_draft_checkpoint(snapshot, plan)
    return prepare_kimi(snapshot, plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache,
        progress=progress, retain_graphs=retain_graphs, core_storage=core_storage, ranks=ranks,
        recipe=RECIPE, draft_source=(DSPARK_REPOSITORY, DSPARK_REVISION), draft_directory=DSPARK_DIRECTORY)


RECIPE = Recipe(
    name="kimi31", served_model_name="Kimi-K3.1-Pro", prepare=prepare, usage=usage,
    # Moonshot evaluation defaults: https://huggingface.co/moonshotai/Kimi-K3.1-Pro#3-evaluation-results
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    checkpoints=(Checkpoint(alias="kimi-k3.1-pro", repository="moonshotai/Kimi-K3.1-Pro",
        revision="d9e99b0792a5e93b78ea9bea62a68167e36ddd9e", architecture="KimiK31ForConditionalGeneration",
        stored_bytes=1_561_217_377_280,
        # K3's matching target layout plus 357,052,416 BF16 indexer bytes replicated on sixteen ranks.
        device_bytes=1_491_026_420_236 + 16 * 357_052_416 + DSPARK_DEVICE_BYTES,
        draft_bytes=DSPARK_DEVICE_BYTES),),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64),
    max_draft_tokens=7, verification_rows=0, max_images=4, image_bytes=32 << 20,
    online_mxfp8=True, online_mxfp8_scope="kda-input", online_mxfp8_scopes=("all", "kda-input"),
    draft_head_quantizations=("mxfp8", "nvfp4"), draft_body_quantizations=("mxfp8",),
    speculative_methods=MappingProxyType({"dflash2": SpeculativeMethod(MappingProxyType({
        16: RankCosts(state_row_bytes=35_450_112, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128*16),
        # Derived from the continuation layout at TP8: KDA convolution and recurrent rows
        # double (96/8 heads); page records and the DFlash2 window rings stay per-rank constant.
        8: RankCosts(state_row_bytes=65_761_536, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128*8),
    }), draft_bytes=5_569_622_016)}),
    stream_overlap=True, language_model_only=True,
    b12x_environment=MappingProxyType({**KIMI3.b12x_environment, "B12X_PCIE_DCP_THREADS": "0"}),
    # Graph storage is a planning allowance; actual graph memory is reported after preparation.
    # Production descriptors: 69 KDA layers at seven-draft convolution capacity,
    # 24 DSA partial pages, five complete per-rank GQA K/V rings and last_hidden.
    # Six-draft initial catalogs fit within this seven-draft state allowance.
    ranks=MappingProxyType({
        16: RankCosts(state_row_bytes=35_122_432, page_bytes=2_174_976,
             graph_bytes=9 << 29, page_tokens=128 * 16),
        8: RankCosts(state_row_bytes=68_055_296, page_bytes=2_174_976,
             graph_bytes=9 << 29, page_tokens=128 * 8)}),
    plain_ranks=MappingProxyType({
        16: RankCosts(state_row_bytes=30_275_328, page_bytes=2_174_976,
            graph_bytes=9 << 29, page_tokens=128 * 16),
        8: RankCosts(state_row_bytes=58_361_088, page_bytes=2_174_976,
            graph_bytes=9 << 29, page_tokens=128 * 8)}),
)
