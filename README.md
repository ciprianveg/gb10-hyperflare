# Kimi-K3.1-Pro — Flare runtime for 16x DGX Spark GB10 (TP16 over RoCE)

This repository is a delivery fork of
[NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)
(upstream README preserved at [README-upstream.md](README-upstream.md)). The
serving tree below carries the GB10 TP16 multi-host customizations described in
the guide; the prepared runtime is published at
[huggingface.co/ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16](https://huggingface.co/ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16).

---
# Kimi-K3.1-Pro on Flare — 16x DGX Spark GB10 (TP16 over RoCE)

Published package: `ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16` — Kimi-K3.1-Pro
prepared as a Flare runtime core for a 16-node NVIDIA DGX Spark (GB10 / SM121)
cluster serving tensor-parallel 16 over the RoCE fabric.

## Performance

~700 tok/s prefill; 24 tok/s prose; 30 tok/s coding; 40 tok/s structured.

**Context**: 2.2M tokens total KV capacity — 550K per stream at C4 (4 concurrent streams).

## Runtime binary

Download `flare-linux-x86_64` from this repository and install it as `flare` on
**every one of the 16 nodes** (`chmod +x flare-linux-x86_64 && sudo cp flare-linux-x86_64 /usr/local/bin/flare`).
It carries the GB10 TP16 transport customizations this core requires — upstream
flare binaries fail trust verification or transport connection against this core.
The exact serving source this binary was built from:
`github.com/ciprianveg/gb10-hyperflare` branch `k31-gb10-tp16` (single release commit).

## Download

```sh
flare download ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16
```

Run this on ONE node (or all — the package is cached under `~/.cache/flare` and
shared via NFS if your cluster mounts a common models directory).

## Run — 16-node launch (default: fixed NST3)

Multi-host serving needs one head command and one peer command per node. On the
HEAD node (the one others connect to), pick a rendezvous port (e.g. 29911):

```sh
flare run ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16   --rank 0 --rendezvous HEAD_IP:29911   --host 0.0.0.0 --port 8000 --gpus 0 --memory-limit=-2G   --mtp-draft-tokens 3 --no-mtp-adaptive-verification --no-history-drafts
```

On each of the 15 PEER nodes (HEAD_IP = the head node's IP):

```sh
flare run ciprianv/kimi-k3.1-pro-dspark-flare-gb10-tp16   --rank N --rendezvous HEAD_IP:29911 --gpus 0 --memory-limit=-2G   --mtp-draft-tokens 3 --no-mtp-adaptive-verification --no-history-drafts
```

with `--rank` 1..15, one rank per node. Rank 0 serves the OpenAI-compatible API
on `:8000`. On first load flare prompts to review and accept the publisher —
the signer is `ciprianv` (production key, fingerprint in `publisher.json` at
this repository's root).

## Run — optional adaptive speculation

Replace the three fixed-NST3 flags
(`--mtp-draft-tokens 3 --no-mtp-adaptive-verification --no-history-drafts`)
with `--mtp-draft-tokens 7 --mtp-adaptive-verification` on every node.

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
