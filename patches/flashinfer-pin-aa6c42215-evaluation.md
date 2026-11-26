# FlashInfer pin evaluation: 85ea6e766 → aa6c42215 (2026-11-14, k31-main-tp8)

Question: can `pyproject.toml` move `flashinfer-python[b12x]` from
`85ea6e766aa8434f89e587267d044dd852b9a312` (our pin, ABI 11 RoCE proxy) to
`aa6c4221586f7607c2bca64abbcdc0699845cf3b` (the rev #291 requires, ABI 14
RoCE proxy) without breaking our port integration?

**Verdict: BUMPED 2026-11-14 (this branch).** `pyproject.toml`/`uv.lock` now pin
aa6c42215 and the Python/C overlay in `patches/roce-counts-staging/` has been
re-ported onto the ABI-14 sources (three-way merge of the overlay delta onto the
aa6c42215 files; merged `_roce_proxy.c` compiles clean with `gcc -c`, ABI parity
14==14 with aa6c42215's `_proxy.py`, every ctypes-bound symbol present; CPU-only
test batteries show failure-set identity vs the pre-change baseline — GPU
revalidation of the staging machinery is still owed at export, see the end). The bump itself is otherwise compatible (details
below); the blocker is exclusively the overlay re-port + its GPU revalidation,
which cannot run while the fleet is busy.

## (a) patches/roce-counts-staging overlay vs the 16 commits 85ea6e766..aa6c42215

| overlay file | unchanged between revs? | our delta (vs 85ea) | re-port effort |
|---|---|---|---|
| `_varlen_cute.py` | **identical** in both revs | 172 diff-lines (counts staged in the publish kernel) | none — transfers as-is |
| `roce_oneshot.py` | changed (~13 hunks: staging allocation sites, tuning, health) | 39 diff-lines, 5 hunks at ~290/1135/1335/1498/1515 | mechanical rebase; upstream hunks sit at 45/79/243/336/447/465/513/524/1158/1184/1570 — none collide with ours |
| `_roce_proxy.c` | **heavily changed** (384 lines) | 350 diff-lines (per-op-kind stats, stall beacons, `CTRL_SLOT_SEQ 14` staged-count slot) | real re-port required, see below |

What changed in the proxy between the revs (all in upstream aa6c42215):

- `ROCE_ABI_VERSION` **11 → 14** (`_roce_proxy.c:94`, `_proxy.py` `ABI_VERSION`).
- `roce_create` signature changed: adds `pipe_bytes` / `varlen_bytes` region
  sizing parameters.
- New `CTRL_CREDIT 768` credit words + relayed-credit rails appended to the
  ctrl page; `wr_id` packing changed to `wr_id & 255`.
- Lanes / start-of-op credits between proxies (the ABI-14 headline).

Why the overlay re-port is mandatory before any bump: the overlay is applied by
copying over `.venv/.../flashinfer/experimental/b12x/comm/roce/` on every
prepare host. Copying the current ABI-11-based `_roce_proxy.c` next to ABI-14
kernels fails `_proxy.py`'s `lib.roce_abi_version() != ABI_VERSION` check at
best, and silently mismatches the control protocol at worst — the
counts-staging fix would be lost and the layer-sequence stall would return.

Portability note (good news): the overlay's staged-count tag words
(`CTRL_SLOT_SEQ 14`, words `14..14+SLOTS`) remain free in aa6c42215's ctrl
layout — upstream only appends `CTRL_CREDIT 768` and leaves words 14-47
untouched — so the mechanism itself ports; it has never been validated on
ABI-14 code, and its earlier iterations each needed GPU debug cycles.

## (b) ABI-14 vendored proxy (8a228b41, helios/native/roce_proxy.c)

`helios/native/roce_proxy.c` is a self-contained C file compiled into the
helios binary; it has **no build-time dependency on flashinfer-side b12x
Python symbols** (the interop contract is the wire/control-word ABI and the
core's recorded proxy ABI). `helios/src/execution/roce.rs` accepts cores
recording ABI 9/10/11/14 and asserts the runtime provides its vendored ABI 14.

A bump actually *improves* parity: today the prepare-side flashinfer proxy is
ABI 11 (cores record 11; helios tolerates it), after the bump both sides are
ABI 14. No flare-side symbol conflicts found.

## (c) #291's expectations

#291 explicitly pins aa6c42215: "the preparation dependency must also name the
merged FlashInfer sources that implement the selected dynamic KDA and DCP8
collective contracts"; its 62/62 dynamic fused-KDA host tests ran against that
rev (aa6c42215 = "Validate optional plans and bias on dynamic fused KDA calls
(#109)"). Keeping our 85ea6e766 pin does not break #291's *code* — the
dflash2 body-quantization feature is pin-independent — it only leaves the
dynamic-KDA/DCP8 kernel improvements unsynced until the bump.

## Migration path (when the fleet frees up)

1. Re-port the overlay onto aa6c42215 sources: rebase the 5 `roce_oneshot.py`
   hunks (mechanical) and the `_roce_proxy.c` stats/beacons/staged-count delta
   (~350 lines) onto the ABI-14 file; `_varlen_cute.py` copies as-is.
2. Bump `pyproject.toml` + `uv.lock` to aa6c42215, `uv lock`, sync venvs on
   prepare hosts, copy the re-ported overlay.
3. GPU revalidation on .111/.112: RoCE chunk sweep (eager + replay, equal +
   skew) and `test_varlen_layer_sequence_over_roce` 3x, plus the counts-staging
   sentinel checks — before any export that relies on the staging machinery.
4. Cores will then record proxy ABI 14 (was 11); helios accepts both, but mix
   nothing: re-export cores prepared with the ABI-11 overlay after the bump.
