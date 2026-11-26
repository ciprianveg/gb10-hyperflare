---
license: mit
base_model: moonshotai/Kimi-K3.1-Pro
draft_model: Inferact/Kimi-K3.1-Pro-DSpark
tags:
- flare
- kimi
- gb10
- sm121
- tp16
- roce
- dgx-spark
library_name: flare
pipeline_tag: text-generation
---

# Kimi-K3.1-Pro — Flare runtime for 16x DGX Spark GB10 (TP16 over RoCE)

Kimi-K3.1-Pro prepared as a Flare runtime core for a 16-node NVIDIA DGX Spark
(GB10 / SM121) cluster serving tensor-parallel 16 over the RoCE fabric.

## Performance

~700 tok/s prefill; 24 tok/s prose; 30 tok/s coding; 40 tok/s structured.

**Context**: 2.2M tokens total KV capacity — 550K per stream at C4 (4 concurrent streams).

## Context depth scaling

Decode tok/s (C1, 2,048-token generations, warm):

| Workload | 4k | 64k | 128k |
|---|---|---|---|
| coding | 29.97 | 27.82 (−7.2%) | 27.37 (−8.7%) |
| prose | 25.73 | 24.33 (−5.4%) | 21.86 (−15.0%) |

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
