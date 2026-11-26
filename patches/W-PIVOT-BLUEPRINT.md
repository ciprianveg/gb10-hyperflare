# Option W pivot blueprint (windowed varlen, upstream-proven) — preserved 2026-11-13

Option R (record replica) landed and validated first (commit 64130819 + the e2e
local-read test), so W is not required to kill the per-part lockstep crawl — but
this is the fallback/complementary path if the replica's memory budget
(55 layers x 708 B/token) ever binds at very large contexts. Precise pointers
for the W implementation in OUR checkout:

1. ignition/recipes/glm53/dcp.py — the windowed DSA-records pattern: search by
   symbol WINDOW_RECORDS (=65536) and PREFILL_SLICE_ROWS (upstream cites lines
   306-320/344; our tree's numbers differ). Note the 528B latent-record layout
   there vs Kimi's 576B E4M3 latent + 132B planar index — the adaptation is the
   record format, the per-window staging sizing and the window loop carry over.
2. ignition/components/collectives.py — PreparedVarlenGather (upstream :847-889),
   PreparedChunkedAllGather (upstream :197-229), PreparedChunkedAllReduce,
   ConcurrentAllGather (overlap primitive, upstream :518-561). All present in
   our tree — the stage-5 port added to this file without removing them.
3. Working usage: ignition/recipes/deepseek4/bootstrap.py (~:342-366 — runtime
   sizing with varlen_bytes = min(WINDOW_RECORDS, blocks*256)*528 and the
   one-shot decode + pipelined prefill dual budgets) + deepseek4/attention.py
   (prefill_records_per_call=131072) + deepseek4/decoder.py (:235 overlap
   activation).
4. Upstream raw backups: raw.githubusercontent.com/NVIDIA/hyperflare-inference/
   main/ignition/recipes/glm53/dcp.py and .../ignition/components/collectives.py.

The W adaptation = Kimi record formats (576B E4M3 latent + 132B index) into the
windowed device-counted pattern — counts computed in-graph on the same stream,
one-shot varlen per window, stream-serialized — replacing the per-part lockstep
path in kimi31/dcp.py + attention.py.
