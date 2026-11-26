# RoCE varlen counts-staging patch series (k31-tp8, tp14→tp16)

Host-venv patches for the FlashInfer b12x RoCE comm package (the served image
carries only the Rust binary + helios/native/roce_proxy.c; these Python/C files
are used by `flare prepare` and the tests on the prepare hosts).  Apply by
copying over
`.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/comm/roce/`
on every prepare host (originals backed up per host under /tmp/opencode/ as
`_roce_proxy.c.orig`, `_varlen_cute.py.orig`, `*.tp15`).

Contents (state = tp16, re-ported onto the ABI-14 sources of flashinfer aa6c42215
— the 2026-11-14 pin bump; every tp16 feature carried: per-op-kind stats, beacons
(B12X_ROCE_REPORT_SECS), staged-count slots (CTRL_SLOT_SEQ words 14..15, still free
in the ABI-14 ctrl layout), FLARE_ROCE_VARLEN_CHUNK_BYTES handling):
- `_roce_proxy.c` — per-op-kind stats + stall beacons (B12X_ROCE_REPORT_SECS)
  and the sequence-tagged varlen count slot: the kernel publishes its count
  record together with its sequence (CTRL_SLOT_SEQ, word 14+slot) and
  `run_varlen_count_wait` never consumes a stale generation of the seq&1-aliased
  slot (zero tag = pre-tagging kernel, ABI-compatible; no layout change).
- `_varlen_cute.py` — the counts themselves are staged: a 1-block publish kernel
  copies the caller's counts into a runtime-owned double-buffered device slot
  tagged with the op's sequence (release-published), and the gather kernel spins
  (bounded, ERROR_KIND_COUNT_STAGE on timeout) until the tag matches its
  sequence before reading counts — the gather never touches the caller-owned
  counts tensor.  Fixes the misaligned-u32 tag indexing (slot*4 bytes).
- `roce_oneshot.py` — allocates the staging tensors before first use (refused
  inside capture), passes them to the launcher, and names the new error kind in
  check_health.

VALIDATION STATE (world 2, .177.111/.177.112): chunk sweep deterministic green
(512/511/256/33/31/30/2/1, equal + skew, eager + replay).  The layer-sequence
repro (test_kimi31_varlen_chunks_cuda.py::test_varlen_layer_sequence_over_roce)
STILL stalls at its 4th replay op with divergent per-rank staged counts even
with generation-checked staging — the per-op `counts.copy_()` nodes themselves
execute unordered against the kernels in the captured graph.  Next step per the
bounded-iteration directive: pivot to the upstream glm53/dcp.py windowed
PreparedVarlenGather pattern (collectives.py:847-889) instead of a further
generation-tag iteration.
