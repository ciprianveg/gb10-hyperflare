# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated sparse MLA over globally selected interleaved FP8 latent records."""

import torch
from b12x.attention import dense_mla

from ignition.components.collectives import ConcurrentAllGather, PreparedAllGather, PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.peer_record_gather import PreparedPeerRecordGather
from ignition.components.interleaved_history import PreparedInterleavedHistory
from ignition.recipes.kimi3.attention import FP8, PreparedMLAProjection, SlicedLseCombine
from ignition.recipes.kimi3.dcp import LSE_SLICE_ROWS, ContextShards

from .dcp import RECORD_BYTES, TOPK, selection_slots, write_records
from .indexer import PreparedDSAIndexer, SharedBuffers


# Sparse DSA attention runs over interleaved context shards: sixteen at full width,
# eight on the TP8 fleet (heads 96 / 8 = 12 and hidden 7168 / 8 = 896 both divide).
CONTEXT_RANKS = (8, 16)


def validate_context_ranks(world, heads, hidden):
    """Check one layer's attention geometry against its context-shard count.

    Real layers run at full model width (heads 96, hidden 7168: both divide eight
    and sixteen ranks), so every rank must own integral head and column shares.
    Composition fixtures exercise the exchanges with sub-rank widths; those skip
    the divisibility half, which preparation asserts against the real config.
    """
    if world not in CONTEXT_RANKS:
        raise ValueError("Kimi K3.1 sparse attention requires eight or sixteen context ranks")
    if heads >= world and heads % world:
        raise ValueError("Kimi K3.1 sparse attention requires attention heads that divide"
                         " the context ranks")
    if hidden >= world and hidden % world:
        raise ValueError("Kimi K3.1 sparse attention requires hidden columns that divide"
                         " the context ranks")


class PreparedDSA(PreparedMLAProjection):
    """Kimi latent projections with per-token global top-2048 sparse attention.

    Decode and verification gather absorbed query heads and combine local selected-record
    partials by their LSE. Prefill keeps local heads and gathers only each
    128-query slice's selected records into a reusable one-record-page slab.
    """

    def __init__(self, weights, source, metadata, cache, *, index_cache, eps, mode, session, name, workspace,
                 buffers, latents=None, latent_rows=None, latent_columns=None, coordinator=None, context=None,
                 peers=None, allocator=None, replica_latent=None, replica_index=None,
                 replica_layer=-1):
        if mode not in ("decode", "verify", "prefill"):
            raise ValueError("Kimi K3.1 sparse attention requires decode, verify, or prefill")
        super().__init__(weights, source, metadata, cache, eps=eps, mode=mode, session=session, name=name,
            workspace=workspace, buffers=buffers, latents=latents, latent_rows=latent_rows,
            latent_columns=latent_columns, coordinator=coordinator, context=context, peers=peers, allocator=allocator)
        self.shards = ContextShards() if context is None else context.shards
        context_session = getattr(context, "session", None) or session
        self.remote = getattr(context, "remote", False)
        # A11: this layer's replica slabs; the first DSA layer publishes the chunk's
        # records once per chunk (the selection barrier), every layer reads locally.
        self.replica_latent, self.replica_index = replica_latent, replica_index
        self.replica = getattr(context, "replica", None)
        self.replica_layer = replica_layer
        self.replica_peers = (torch.tensor([replica_latent.data_ptr()] * self.shards.world,
                                           dtype=torch.int64, device=source.device)
                              if replica_latent is not None else None)
        if self.replica_latent is not None:
            # Register this layer's pools for the publish and create the publish
            # gathers once (idempotent across the layers' constructions).
            self.replica.register(self.replica_layer, cache, index_cache, metadata.page_table[0])
            self.replica.bind(context.heads, context_session, context.coordinator, name + ".replica")
        # heads is this rank's shard of the global head count; hidden is the full width.
        validate_context_ranks(self.shards.world, self.absorbed.shape[1] * self.shards.world,
                               source.shape[1])
        if index_cache.shape[0] != 2 * cache.shape[0]:
            raise ValueError("Kimi sparse index cache requires two 64-record subpages per latent page")
        self.indexer = PreparedDSAIndexer(weights, source, self.query_norm, metadata, index_cache, self.shards,
            context=context, eps=eps, mode=mode, session=session, name=name + ".index", workspace=workspace,
            buffers=buffers.setdefault("index", {}), latent_cache=cache, allocator=allocator,
            replica_index=replica_index, replica_slot=replica_layer)
        self.slots = self.buffer("cache_slots", dtype=torch.int64)
        self.mode = mode
        self.history = None
        self.exchange = None
        self.windowed = None
        self.windowed_end = None
        if mode == "prefill" and source.shape[0] >= 4096:
            if self.remote:
                # Stage 5: the replicated index history rides the backbone's windowed
                # varlen exchange (query_selection); no shared pages, no staged slab,
                # and no peer page is ever read under multi-host preparation.
                pass
            elif context.staging is None or context.staged_records is None or metadata.starts.numel() != 1:
                raise ValueError("large sparse prefill requires fixed staging for one request's history")
            else:
                history_storage = SharedBuffers(buffers.setdefault("history", {}), source.device, allocator)
                self.history_end = history_storage.buffer("end", (1,), torch.int64)
                self.history_records = context.staged_records
                self.history = PreparedInterleavedHistory(cache, metadata.page_table[0], self.history_end,
                    self.shards.index, context.staging.tensor.view(-1, RECORD_BYTES), context.staging.addresses,
                    self.history_records, context.stage_signals, runtime=context.stage_runtime,
                    session=context_session, coordinator=context.coordinator, name=name + ".history")
        self.parts = {}
        storage = SharedBuffers(buffers.setdefault("selected_attention", {}), source.device, allocator)
        heads = self.absorbed.shape[1]
        common = dict(device=source.device, kv_dtype=FP8, page_size=1, use_cuda_graph=True,
                      partial_dtype=torch.float32)
        constants = context.attention_constants
        cache_records = cache.view(-1, 1, RECORD_BYTES)
        for start, end in self.indexer.bounds:
            rows = end - start
            if rows in self.parts:
                continue
            selector = None if self.indexer.sharded is not None else self.indexer.selections[rows]
            owners = storage.buffer("owners", (rows, TOPK), torch.int32)
            slots = storage.buffer("slots", (rows, TOPK), torch.int32)
            counts = storage.buffer("counts", (rows,), torch.int32)
            starts, pages = constants.starts[:rows + 1], constants.pages[:rows]
            part = dict(selector=selector, owners=owners, slots=slots, counts=counts)
            if mode == "prefill":
                selected = storage.buffer("records", (rows, TOPK, RECORD_BYTES), FP8)
                if self.remote and self.replica_latent is not None:
                    # A11: selections read this rank's replica by global position —
                    # owners are all zero and slots carry global tokens.
                    part["gather"] = PreparedPeerRecordGather(self.replica_peers, owners, slots,
                        counts, selected, peer_records=self.replica_latent.shape[0],
                        session=session, name=f"{name}.records{start}")
                elif (self.remote and getattr(context, "window_records", 0) > 0
                        and context.staged_records is not None):
                    # Option W: selections read the shared staged slab by global
                    # position (owners all zero); the layer's windowed latent publish
                    # precedes the parts on the same stream.
                    part["gather"] = PreparedPeerRecordGather(context.staged_address, owners,
                        slots, counts, selected, peer_records=context.staged_records.shape[0],
                        session=session, name=f"{name}.records{start}")
                elif not self.remote:
                    if peers is None or peers.shape != (self.shards.world,):
                        raise ValueError("Kimi sparse prefill requires every rank's latent-cache address")
                    local_history = self.history is not None
                    part["gather"] = PreparedPeerRecordGather(context.staged_address if local_history else peers,
                        owners, slots, counts, selected,
                        peer_records=self.history_records.shape[0] if local_history else cache.shape[0] * cache.shape[1],
                        session=session, name=f"{name}.records{start}",
                        dependencies=(self.history_records, self.history.completed) if local_history else ())
                caps = dense_mla.Caps(mode="decode", q_dtype=torch.bfloat16, num_q_heads=heads,
                    max_total_q=rows, max_batch=rows, max_cache_tokens=TOPK, max_page_table_width=TOPK,
                    num_cache_pages=rows * TOPK, **common)
                part["attention"] = PreparedDenseMLA(self.absorbed[start:end], selected.view(-1, 1, RECORD_BYTES),
                    self.attended[start:end], pages, counts, starts, caps=caps, scale=192 ** -.5,
                    session=session, name=f"{name}.attention{start}", workspace=workspace,
                    kv_scale=constants.scale, q_scale=constants.scale)
            else:
                total = heads * self.shards.world
                quantized = storage.buffer("query", (rows, heads, RECORD_BYTES), FP8)
                gathered = storage.buffer("gathered_query", (rows, total, RECORD_BYTES), FP8)
                partial = storage.buffer("partial", (rows, total, 512), torch.bfloat16)
                lse = storage.buffer("lse", (rows, total), torch.float32)
                part.update(quantized=quantized, gathered=gathered, partial=partial, lse=lse)
                part["gather"] = PreparedAllGather(quantized.view(rows, -1), gathered.view(rows, -1),
                    runtime=context.heads, session=context_session, coordinator=context.coordinator,
                    name=name + ".gather_heads")
                caps = dense_mla.Caps(mode="decode", q_dtype=FP8, num_q_heads=total,
                    max_total_q=rows, max_batch=rows, max_cache_tokens=TOPK, max_page_table_width=TOPK,
                    num_cache_pages=cache.shape[0] * cache.shape[1], **common)
                part["attention"] = PreparedDenseMLA(gathered, cache_records, partial,
                    slots, counts, starts, caps=caps, scale=192 ** -.5, session=session,
                    name=name + ".attention", workspace=workspace, lse=lse,
                    kv_scale=constants.scale, q_scale=constants.scale)
                if self.remote:
                    # Stage 2: ranks on separate hosts pack every peer's head shard per LSE
                    # slice and gather the packs over the shared RoCE backbone. One bounded
                    # staging pair serves every slice and serial layer of the graph.
                    from ignition.components.head_exchange import pack_width

                    lse_width = pack_width(self.shards.world, total, 512)
                    send = storage.buffer("lse_send", (LSE_SLICE_ROWS, lse_width), torch.bfloat16)
                    received = storage.buffer("lse_received",
                                              (LSE_SLICE_ROWS, self.shards.world * lse_width), torch.bfloat16)
                    part["combine"] = SlicedLseCombine(partial, lse, self.attended, runtime=context.heads,
                        session=context_session, coordinator=context.coordinator, name=f"{name}.combine{start}",
                        rank=self.shards.index, send=send, received=received)
                else:
                    part["combine"] = PreparedLseReduceScatter(partial, lse, self.attended,
                        runtime=context.heads, session=context_session, coordinator=context.coordinator, name=name + ".combine",
                        threads=512, block_limit=8)
            self.parts[rows] = part
        if self.remote and mode == "prefill" and self.replica_latent is None:
            # Stage 5 / Option W: one shared exchange serves every part and every
            # serial layer. Under W the per-layer windowed latent exchange refills the
            # staged slab (zero replica memory); otherwise the per-part selected-record
            # exchange applies, whose staging row is the worst case (every selection
            # owned by one rank). Buffers live in the decoder's shared storage beside
            # the slab.
            self._prepare_remote_exchange(source, metadata, cache, context, buffers, allocator,
                                          storage, context_session, name)
        self.query_overlap = None
        # The auxiliary-stream query gather would race the ranks' single ordered RoCE
        # channel (glm53's QueryChain runs its RoCE gather on the main stream instead).
        if (mode == "decode" and self.absorbed.shape[0] == 1 and workspace is not None and workspace.overlap
                and not self.remote):
            part, = self.parts.values()
            # Both candidate transports consume the joined query before reusing the channel.
            part["gather"].dependencies.append(cache)
            part["selector"].merge.dependencies.append(part["gather"].output)
            self.query_overlap = ConcurrentAllGather(part["gather"], workspace=workspace, session=session,
                                                     parallel_plans=self.indexer.local_plans)

    def _prepare_remote_exchange(self, source, metadata, cache, context, buffers, allocator,
                                 storage, session, name):
        windowed = getattr(context, "window_records", 0) > 0
        if windowed and (context.staged_records is None or context.staged_address is None):
            raise ValueError("windowed latent exchange requires the full staged slab and its address")
        if windowed:
            from .dcp import PreparedRemoteLatentHistory

            history_storage = SharedBuffers(buffers.setdefault("history", {}), source.device, allocator)
            self.windowed_end = history_storage.buffer("windowed_end", (1,), torch.int64)
            self.windowed = PreparedRemoteLatentHistory(cache, metadata.page_table[0],
                self.windowed_end, self.shards, context.staged_records, history_storage,
                window=context.window_records, runtime=context.heads, session=session,
                coordinator=context.coordinator, name=name + ".latent")
        else:
            from .dcp import REMOTE_RECORD_ROWS, PreparedSelectedRecords

            part_rows = max(end - start for start, end in self.indexer.bounds)
            self.exchange = PreparedSelectedRecords(cache,
                storage.buffer("records", (part_rows, TOPK, RECORD_BYTES), FP8),
                SharedBuffers(buffers.setdefault("selected_attention", {}), source.device, allocator),
                self.shards, runtime=context.heads, session=session, coordinator=context.coordinator,
                name=name + ".selected")
            if part_rows > REMOTE_RECORD_ROWS:
                raise ValueError("remote selected-record exchanges carry one 128-row part")

    def __call__(self, source):
        self.project(source)
        self.slots.copy_(self.shards.token_slots(self.metadata.positions, self.metadata.token_page_table))
        write_records(self.latent, self.slots, self.cache.view(-1, RECORD_BYTES))
        if self.query_overlap is not None:
            part, = self.parts.values()
            part["quantized"].copy_(self.absorbed.float().clamp(-448., 448.).to(FP8))
            self.query_overlap.start(part["quantized"].view(self.absorbed.shape[0], -1))
        self.indexer.project(source)
        if self.replica_latent is not None:
            # A11: publish this layer's records into the replica (the layer's only
            # selection barrier) before any part reads it; each DSA layer publishes
            # its own records right after writing them. The fused publish carries the
            # indexer's index keys (already committed by project() above) in the same
            # varlen gather, so the scorer's own publish disappears.
            if self.replica.fused_publish:
                self.replica.publish(self.metadata, self.replica_layer)
            else:
                self.replica.publish_latents(self.metadata, self.replica_layer)
        if self.windowed is not None:
            # Option W: refill the staged slab with this layer's records (one windowed
            # varlen publish per layer, counts derived in-graph) before any part reads
            # it; the slab is per-layer scratch, so the publish covers [0, end).
            self.windowed_end.copy_((self.metadata.starts + self.metadata.lengths)
                                    * (self.metadata.lengths > 0))
            self.windowed()
        if self.indexer.sharded is not None:
            self.indexer.sharded()
        if self.history is not None:
            self.history_end.copy_((self.metadata.starts + self.metadata.lengths) * (self.metadata.lengths > 0))
            self.history()
        for start, end in self.indexer.bounds:
            part = self.parts[end - start]
            positions = self.metadata.positions[start:end]
            tables = self.metadata.token_page_table[start:end]
            if self.indexer.sharded is not None:
                selected = self.indexer.sharded.selected[start:end]
            elif self.query_overlap is None:
                selected = part["selector"](self.indexer.query[start:end], self.indexer.head_weights[start:end],
                                             positions, tables)
            else:
                selector = part["selector"]
                dependency = selector.prepare_local(self.indexer.query[start:end],
                    self.indexer.head_weights[start:end], positions, tables)
                self.query_overlap.finish(part["quantized"].view(self.absorbed.shape[0], -1), dependency)
                selected = selector.exchange(positions)
            if self.mode == "prefill":
                if self.remote and self.replica_latent is not None:
                    # A11: the layer's records are already in the local replica; the
                    # selection addresses the replica by global position and reads it
                    # locally.
                    selection_slots(selected, positions, tables, part["owners"], part["slots"],
                        part["counts"], self.shards, self.cache.shape[0], local=False,
                        global_records=True)
                    part["gather"]()
                elif self.remote and self.windowed is not None:
                    # Option W: the layer's records are already in the staged slab (the
                    # windowed publish ran before the parts); the selection addresses
                    # the slab by global position and reads it locally.
                    selection_slots(selected, positions, tables, part["owners"], part["slots"],
                        part["counts"], self.shards, self.cache.shape[0], local=False,
                        global_records=True)
                    part["gather"]()
                elif self.remote:
                    # Stage 5: the bit-identical selection drives the varlen exchange;
                    # records arrive in selection order directly into the slab. This
                    # branch reads no part gather — the exchange replaces it.
                    self.exchange(selected, positions, tables)
                else:
                    selection_slots(selected, positions, tables, part["owners"], part["slots"],
                        part["counts"], self.shards, self.cache.shape[0], local=False,
                        global_records=self.history is not None)
                    part["gather"]()
                part["attention"](self.absorbed[start:end], self.attended[start:end])
            else:
                selection_slots(selected, positions, tables, part["owners"], part["slots"],
                    part["counts"], self.shards, self.cache.shape[0], local=True,
                    global_records=False)
                if self.query_overlap is None:
                    part["quantized"].copy_(self.absorbed.float().clamp(-448., 448.).to(FP8))
                    part["gather"](part["quantized"].view(self.absorbed.shape[0], -1))
                part["attention"](part["gathered"], part["partial"])
                part["combine"](part["partial"], part["lse"])
        return self.finish()
