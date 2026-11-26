# Kimi-K3.1-Pro on Flare — 16x DGX Spark GB10 (TP16 over RoCE)

Published package: `ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16` — Kimi-K3.1-Pro
prepared as a Flare runtime core for a 16-node NVIDIA DGX Spark (GB10 / SM121)
cluster serving tensor-parallel 16 over the RoCE fabric.

## Performance

~700 tok/s prefill; 24 tok/s prose; 30 tok/s coding; 40 tok/s structured.

**Context**: 2.2M tokens total KV capacity — 550K per stream at C4 (4 concurrent streams).

## Runtime binary

Use the bundled `flare-linux-x86_64` from this repository — it carries the GB10
TP16 transport customizations this core requires. Upstream flare binaries will
fail trust verification or transport connection against this core.

## Download

```sh
flare download ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16
```

or the Docker equivalent (mount the models directory and run the flare binary
from the image, exactly as the runtime's own run guide describes):

```sh
docker run --rm --pull never --network host \
  --mount type=bind,src=/home/ciprian/models,dst=/models,readonly \
  --entrypoint /qualification/flare <flare-image> \
  download ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16
```

## Run — default (fixed NST3)

```sh
flare run ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16 \
  --mtp-draft-tokens 3 --no-mtp-adaptive-verification --no-history-drafts
```

## Run — optional adaptive speculation

For adaptive speculation use `--mtp-draft-tokens 7 --mtp-adaptive-verification` instead of the fixed-NST3 flags.

## Credits

- **Model**: [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) (base model) and [Inferact/Kimi-K3.1-Pro-DSpark](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark) (the DSpark draft/preview checkpoint the speculative preparation builds on).
- **Upstream runtime**: [NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference) (the hyperflare git — the Flare runtime this preparation was built with; our fork carries serving customizations on top).
- **LIL**: local-inference-lab (the recipes/flashinfer ecosystem the preparation pipeline draws from).

## 16x GB10 SM121 special improvements

Implementation-level notes on what this package's serving stack adds for a
16-node GB10 cluster. (C2/C4 detail sections to be added later.)

- **Dual-rail RoCE fabric.** Each node carries four RoCE interfaces (two rails x
  two ports, MTU 4096) wired across two switches; the serving stack pins exactly
  two HCAs — one per rail — and relies on transport defaults for rail selection,
  which connect 2 rails per peer for both the tp16 and dcp8 communicators.

- **Multi-communicator transport schema.** The runtime-core schema carries a
  per-rank list of RoCE communicators (`helios/src/runtime_core/roce_communicators.rs`,
  proto fields core f18 / runtime_core f13) instead of a single legacy transport
  record, and the serving path selects transport via the multi-communicator
  accessor (`require_cluster_transport` over `roce_communicators(0)`). This is
  what lets one core carry both the tp16 world-16 and the dcp8 world-8
  communicators and boot cleanly.

- **Customized ABI-14 RoCE proxy.** The serving binary embeds a static C proxy
  (`helios/native/roce_proxy.c` + `roce_peer_order.h` + `roce_peer_wave.h`) with
  three serving-path customizations: cyclic peer ordering for ring transfer,
  per-peer pipe-chunk posting on the receive side, and receive-side
  completion pacing that keeps slow peers from stalling the round.

- **Metadata-limit raise for large rank-metadata cores.** The speculative
  preparation of this model produces a 617 MB signed rank-metadata body; the
  core/schema limits were raised to carry it end to end
  (`helios/src/core/schema.rs` MAX_BODY 512 MiB, MAX_SELECTED_RANK_METADATA 2 GiB),
  enforced identically at export sign time and at serving decode time.

- **Early forward-compile pool for GB10 UMA exports.** The export preparation
  engages the forward-compile pool during capture (`@with_autotune_effort` on the
  recipe prepare path) — the documented GB10 unified-memory requirement; exports
  without it fail their compile stage on GB10.

- **Merged-build lineage with git reproducibility.** Serving binaries are built
  from a single merged source-of-truth tree (the same lineage that produced the
  published binary), committed as a git baseline before carrying upstream picks,
  so every published binary's provenance is a diff against a reviewable commit
  rather than a set of hand-applied patches.

- **Identical-env serving defaults.** The serving standard is transport defaults
  with exactly two pinned env knobs — the RoCE HCA pair and the shared proxy
  cache directory — established by an A/B that measured the retired overrides as
  a prefill regression; every published start command uses these defaults.

*C2/C4 detail sections to be added later.*
