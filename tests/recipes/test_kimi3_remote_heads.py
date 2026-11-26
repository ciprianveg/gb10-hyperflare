# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 remote head exchanges: FP8 transport view, latent gather geometry, sliced LSE combines.

Every rewired site is checked with fake runtimes: a RoCEnante backbone is a plain
object (the real transport is exercised by the two-host RoCE test), and the prepared
components are stubbed to record the declarations their construction makes.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import collectives
from ignition.components.head_exchange import pack_width
from ignition.recipes.kimi3 import attention, bootstrap
from ignition.recipes.kimi3.attention import SlicedLseCombine
from ignition.recipes.kimi3.dcp import LSE_SLICE_ROWS, RECORD_BYTES, ContextShards
from ignition.recipes.kimi31 import attention as kimi31_attention
from ignition.recipes.kimi31.dcp import TOPK
from ignition.recipes.kimi31.indexer import SharedBuffers


def test_roce_transport_view_pairs_fp8_bytes_and_keeps_pcie_native():
    source = torch.randint(0, 200, (4, 576), dtype=torch.uint8).view(torch.float8_e4m3fn)
    viewed = collectives._gather_transport_view(source, roce=True)
    assert viewed.dtype == torch.bfloat16 and viewed.shape == (4, 288)
    # Byte-exact: the BF16 words re-view as the original E4M3 bytes.
    assert torch.equal(viewed.view(torch.uint8), source.view(torch.uint8))
    # The PCIe branch carries E4M3 natively; the FP32 lane is unchanged.
    assert collectives._gather_transport_view(source).dtype == torch.float8_e4m3fn
    fp32 = torch.randn(4, 64)
    assert collectives._gather_transport_view(fp32).dtype == torch.bfloat16
    assert collectives._gather_transport_view(fp32, roce=True).dtype == torch.bfloat16


def test_sliced_lse_combine_bounds_staging_and_declares_rank(monkeypatch):
    declared = []

    def prepared(partial, lse, output, **kwargs):
        declared.append((partial.shape[0], output.shape, kwargs))
        return SimpleNamespace(rows=partial.shape[0], **kwargs)

    monkeypatch.setattr(attention, 'PreparedLseReduceScatter', prepared)
    rows, heads, world = 200, 8, 2
    width = pack_width(world, heads, 512)
    partial = torch.zeros((rows, heads, 512), dtype=torch.bfloat16)
    lse = torch.zeros((rows, heads), dtype=torch.float32)
    output = torch.zeros((rows, heads // world, 512), dtype=torch.bfloat16)
    send = torch.zeros((LSE_SLICE_ROWS, width), dtype=torch.bfloat16)
    received = torch.zeros((LSE_SLICE_ROWS, world * width), dtype=torch.bfloat16)
    rank = torch.tensor([1], dtype=torch.int64)
    combine = SlicedLseCombine(partial, lse, output, runtime=object(), session=None,
                               coordinator=None, name='combine', rank=rank, send=send,
                               received=received)
    # One bounded staging pair serves every row slice; slices never exceed the bound.
    assert [(start, end) for start, end, _ in combine.parts] == [(0, 128), (128, 200)]
    assert [entry[0] for entry in declared] == [128, 72]
    for count, shape, kwargs in declared:
        assert kwargs['rank'] is rank
        assert kwargs['staging'][0].shape == (count, width)
        assert kwargs['staging'][1].shape == (count, world * width)
        assert shape == (count, heads // world, 512)
    # Slices beyond the first carry an index in their declared name.
    assert declared[0][2]['name'] == 'combine.0' and declared[1][2]['name'] == 'combine.1'
    calls = []

    class Part:
        def __init__(self, count):
            self.count = count

        def __call__(self, partial, lse):
            calls.append((self.count, partial.shape[0], lse.shape[0]))

    combine.parts = [(0, 128, Part(128)), (128, 200, Part(72))]
    combine(partial, lse)
    assert calls == [(128, 128, 128), (72, 72, 72)]


class Plan:
    def memory_requirements(self):
        return SimpleNamespace(scratch_nbytes=0)


def _mla_weights(heads, latent_width, local_width, device):
    return {
        'key': torch.zeros((heads, 128), dtype=torch.bfloat16, device=device),
        'value': torch.zeros((heads, 128), dtype=torch.bfloat16, device=device),
        'query_norm': torch.ones((latent_width - RECORD_BYTES,), dtype=torch.bfloat16, device=device),
        'kv_norm': torch.ones((512,), dtype=torch.bfloat16, device=device),
        'latent_gate': SimpleNamespace(out_features=local_width),
        'q_b': object(), 'output': object(),
    }


@pytest.mark.parametrize('remote', [False, True])
def test_prepared_mla_wires_remote_head_rows_and_sliced_combine(monkeypatch, remote):
    gathers, combines, attentions, chunked, latent_views = [], [], [], [], []

    monkeypatch.setattr(attention, 'PreparedAllGather',
                        lambda source, output, **kwargs: gathers.append((source, output, kwargs)) or object())
    monkeypatch.setattr(attention, 'PreparedDenseMLA', lambda *args, **kwargs: attentions.append(args) or object())
    monkeypatch.setattr(attention, 'PreparedChunkedAllGather',
                        lambda source, output, **kwargs: chunked.append((source.shape, output.shape)) or object())
    monkeypatch.setattr(attention, 'prepare_projection',
                        lambda x, w, y, **kwargs: SimpleNamespace(output=y, plan=Plan()))
    declared = []

    def prepared(partial, lse, output, **kwargs):
        declared.append((partial.shape[0], kwargs))
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(attention, 'PreparedLseReduceScatter', prepared)

    device, tokens, heads, world = 'cpu', 8, 4, 2
    latent_width, local_width, head_rows = 2112 // world, 12552, 64
    weights = _mla_weights(heads, latent_width, local_width, device)
    from ignition.components.paged_batch import Metadata

    metadata = Metadata.allocate(tokens, tokens, 4, device)
    cache = torch.zeros((4, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
    # Remote: the shared backbone, with explicit geometry. Single-host: a PCIe channel
    # declaring its row capacity and column pack, read as before.
    latents = (SimpleNamespace(world_size=world) if remote
               else SimpleNamespace(world_size=world, max_batch_size=head_rows,
                                    query_head_dim=latent_width))
    context = SimpleNamespace(shards=ContextShards.on(0, world, device), heads=latents,
                              coordinator=None, remote=remote,
                              head_rows=(head_rows if remote else None))
    buffers = {}
    op = attention.PreparedMLA(weights, torch.zeros((tokens, local_width - latent_width), dtype=torch.bfloat16),
        metadata, cache, eps=1e-5, mode='decode', session=None, name='mla', workspace=None, buffers=buffers,
        latents=latents, latent_rows=(64 if remote else None), latent_columns=(latent_width if remote else None),
        context=context)
    # The latent gather's transport is PreparedChunkedAllGather over the backbone with
    # explicit geometry; the decode/verify head gather gathers E4M3 query heads.
    assert len(chunked) == 1
    assert chunked[0] == ((tokens, latent_width), (tokens, world * latent_width))
    assert len(gathers) == 1
    source, output, kwargs = gathers[0]
    assert source.dtype == torch.float8_e4m3fn and source.shape == (tokens, heads * RECORD_BYTES)
    assert output.shape == (tokens, heads * world * RECORD_BYTES) and kwargs['runtime'] is latents
    # Row capacity comes from head_rows when remote, the channel's declaration otherwise.
    for key in ('quantized', 'gathered', 'partial', 'lse'):
        assert buffers[key].shape[0] == head_rows
    if remote:
        assert isinstance(op.combine, SlicedLseCombine)
        assert [entry[0] for entry in declared] == [tokens]
        assert declared[0][1]['rank'] is context.shards.index
        assert declared[0][1]['staging'][0].shape == (tokens, pack_width(world, heads * world, 512))
        assert declared[0][1]['staging'][1].shape == (tokens, world * pack_width(world, heads * world, 512))
        # Staging lives in the shared buffer dict, bounded by one slice.
        assert buffers['lse_send'].shape == (LSE_SLICE_ROWS, pack_width(world, heads * world, 512))
        assert buffers['lse_received'].shape == (LSE_SLICE_ROWS, world * pack_width(world, heads * world, 512))
    else:
        # The PCIe combine is one whole-row exchange with no rank or staging.
        assert [entry[0] for entry in declared] == [tokens]
        assert 'rank' not in declared[0][1] and 'staging' not in declared[0][1]
        assert declared[0][1]['name'] == 'mla.combine'


@pytest.mark.parametrize('remote', [False, True])
def test_latent_gather_requires_explicit_rows_and_width_together(remote):
    backbone = SimpleNamespace(world_size=2)
    kwargs = dict(latents=backbone)
    if remote:
        kwargs.update(latent_rows=64)
    device = 'cpu'
    weights = _mla_weights(4, 1056, 12552, device)
    from ignition.components.paged_batch import Metadata

    metadata = Metadata.allocate(8, 8, 4, device)
    cache = torch.zeros((4, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
    with pytest.raises(AttributeError if not remote else ValueError) as info:
        attention.PreparedMLAProjection(weights, torch.zeros((8, 4), dtype=torch.bfloat16), metadata, cache,
            eps=1e-5, mode='decode', session=None, name='mla', workspace=None, buffers={}, **kwargs)
    # Remote geometry arrives whole or not at all; a bare backbone declares no channel.
    assert ('width together' in str(info.value)) is remote
    assert ('max_batch_size' in str(info.value)) is not remote


@pytest.mark.parametrize('remote', [False, True])
def test_prepared_dsa_disables_query_overlap_and_slices_remote_combines(monkeypatch, remote):
    rows, heads, world = 200, 6, 16
    projection_kwargs = []

    def projection(self, weights, source, metadata, cache, **kwargs):
        projection_kwargs.append(kwargs)
        self.absorbed = torch.zeros((rows, heads, RECORD_BYTES), dtype=torch.bfloat16)
        self.attended = torch.zeros((rows, heads, 512), dtype=torch.bfloat16)
        self.query_norm = torch.zeros((rows, 1536), dtype=torch.bfloat16)
        self.buffer = lambda name, dtype: torch.zeros(rows, dtype=dtype)

    selector = SimpleNamespace(merge=SimpleNamespace(dependencies=[]))
    plans = (Plan(), Plan())
    monkeypatch.setattr(kimi31_attention.PreparedMLAProjection, '__init__', projection)
    monkeypatch.setattr(kimi31_attention, 'PreparedDSAIndexer', lambda *args, **kwargs:
                        SimpleNamespace(bounds=((0, rows),), selections={rows: selector}, sharded=None,
                                        local_plans=plans))
    monkeypatch.setattr(kimi31_attention, 'PreparedDenseMLA', lambda *args, **kwargs: SimpleNamespace(plan=Plan()))
    monkeypatch.setattr(kimi31_attention, 'PreparedAllGather', lambda source, output, **kwargs:
                        SimpleNamespace(output=output, dependencies=[], plan=Plan(), single=False, roce=remote))
    monkeypatch.setattr(kimi31_attention, 'ConcurrentAllGather',
                        lambda gather, **kwargs: (_ for _ in ()).throw(AssertionError('overlap constructed')))
    declared = []

    def prepared(partial, lse, output, **kwargs):
        declared.append((partial.shape[0], kwargs))
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(attention, 'PreparedLseReduceScatter', prepared)
    # kimi31 imports its own name for the single-host PCIe combine.
    monkeypatch.setattr(kimi31_attention, 'PreparedLseReduceScatter', prepared)
    constants = SimpleNamespace(starts=torch.arange(rows + 1, dtype=torch.int32),
        pages=torch.arange(rows * TOPK, dtype=torch.int32).view(rows, TOPK), scale=torch.ones(1))
    context = SimpleNamespace(shards=ContextShards.on(0, world, 'cpu'), attention_constants=constants,
                              heads=object(), coordinator=None, remote=remote)
    buffers = {}
    cache = torch.zeros((1, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn)
    kimi31_attention.PreparedDSA({}, torch.zeros((rows, 8)), None, cache,
        index_cache=torch.zeros((2, 64 * 132), dtype=torch.uint8), eps=1e-5, mode='decode',
        session=None, name='attention', workspace=SimpleNamespace(overlap=True),
        buffers=buffers, context=context, peers=torch.zeros(world, dtype=torch.int64),
        latent_rows=(rows if remote else None), latent_columns=(132 if remote else None))
    # Latent geometry threads through to the shared projection.
    assert projection_kwargs[-1]['latent_rows'] == (rows if remote else None)
    assert projection_kwargs[-1]['latent_columns'] == (132 if remote else None)
    # The auxiliary-stream query overlap never runs under the single ordered RoCE channel.
    assert selector.merge.dependencies == []
    total = heads * world
    width = pack_width(world, total, 512)
    if remote:
        assert [entry[0] for entry in declared] == [LSE_SLICE_ROWS, rows - LSE_SLICE_ROWS]
        for count, kwargs in declared:
            assert kwargs['rank'] is context.shards.index
            assert kwargs['staging'][0].shape == (count, width)
            assert kwargs['staging'][1].shape == (count, world * width)
        staging = buffers['selected_attention']
        assert staging['lse_send'].shape == (LSE_SLICE_ROWS, width)
        assert staging['lse_received'].shape == (LSE_SLICE_ROWS, world * width)
    else:
        assert len(declared) == 1 and declared[0][0] == rows
        assert declared[0][1].get('threads') == 512 and declared[0][1].get('block_limit') == 8


def test_prepared_dsa_remote_decode_keeps_query_overlap_disabled(monkeypatch):
    rows = 1
    projection = (lambda self, weights, source, metadata, cache, **kwargs: (
        setattr(self, 'absorbed', torch.zeros((rows, 6, RECORD_BYTES), dtype=torch.bfloat16)),
        setattr(self, 'attended', torch.zeros((rows, 6, 512), dtype=torch.bfloat16)),
        setattr(self, 'query_norm', torch.zeros((rows, 1536), dtype=torch.bfloat16)),
        setattr(self, 'buffer', lambda name, dtype: torch.zeros(rows, dtype=dtype)))[-1])
    monkeypatch.setattr(kimi31_attention.PreparedMLAProjection, '__init__', projection)
    monkeypatch.setattr(kimi31_attention, 'PreparedDSAIndexer', lambda *args, **kwargs:
                        SimpleNamespace(bounds=((0, rows),), selections={rows: SimpleNamespace(
                            merge=SimpleNamespace(dependencies=[]))}, sharded=None, local_plans=(Plan(),)))
    monkeypatch.setattr(kimi31_attention, 'PreparedDenseMLA', lambda *args, **kwargs: SimpleNamespace(plan=Plan()))
    monkeypatch.setattr(kimi31_attention, 'PreparedAllGather', lambda source, output, **kwargs:
                        SimpleNamespace(output=output, dependencies=[], plan=Plan(), single=False, roce=True))
    monkeypatch.setattr(kimi31_attention, 'ConcurrentAllGather',
                        lambda gather, **kwargs: (_ for _ in ()).throw(AssertionError('overlap constructed')))
    monkeypatch.setattr(attention, 'PreparedLseReduceScatter',
                        lambda partial, lse, output, **kwargs: SimpleNamespace(**kwargs))
    constants = SimpleNamespace(starts=torch.arange(rows + 1, dtype=torch.int32),
        pages=torch.arange(rows * TOPK, dtype=torch.int32).view(rows, TOPK), scale=torch.ones(1))
    context = SimpleNamespace(shards=ContextShards.on(0, 16, 'cpu'), attention_constants=constants,
                              heads=object(), coordinator=None, remote=True)
    op = kimi31_attention.PreparedDSA({}, torch.zeros((rows, 8)), None,
        torch.zeros((1, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn),
        index_cache=torch.zeros((2, 64 * 132), dtype=torch.uint8), eps=1e-5, mode='decode',
        session=None, name='attention', workspace=SimpleNamespace(overlap=True), buffers={},
        context=context, peers=torch.zeros(16, dtype=torch.int64))
    # Single-row decode with overlap available still never forks under remote.
    assert op.query_overlap is None


def test_column_pair_gather_accepts_explicit_remote_geometry(monkeypatch):
    from ignition.components import column_pair_gather

    gathers = []

    class Gather:
        def __init__(self, source, output, **kwargs):
            gathers.append(kwargs)
            self.output = output

        def __call__(self, source):
            self.output.copy_(torch.cat((source, source), dim=1))

    monkeypatch.setattr(column_pair_gather, 'PreparedAllGather', Gather)
    rows, first_width, second_width = 96, 224, 28
    # The RoCEnante backbone declares neither a column pack nor a row capacity.
    backbone = SimpleNamespace(world_size=2)
    first = torch.randn((rows, first_width), dtype=torch.bfloat16)
    second = torch.randn((rows, second_width))
    operation = column_pair_gather.PreparedColumnPairGather(first, second,
        torch.empty((rows, 2 * first_width), dtype=torch.bfloat16),
        torch.empty((rows, 2 * second_width)), runtime=backbone, session=None, coordinator=None,
        name='columns', buffers={}, columns=first_width + 2 * second_width)
    # Whole rows gather in one chunk within the byte capacity.
    assert len(gathers) == 1 and operation.send.shape == (rows, first_width + 2 * second_width)
    assert operation.received.shape == (rows, 2 * (first_width + 2 * second_width))
    operation()
    # Rank zero's FP32 columns arrive as their exact BF16 lane pairs (compare bit
    # patterns: random FP32 low halves can encode NaN lanes in either view).
    assert torch.equal(operation.output_second.view(torch.bfloat16)[:, :2 * second_width].contiguous()
                      .view(torch.int16), second.view(torch.int16))
    # A wrong explicit pack is rejected exactly as a mismatched PCIe channel is.
    with pytest.raises(ValueError, match='dimensions differ'):
        column_pair_gather.PreparedColumnPairGather(first, second,
            torch.empty((rows, 2 * first_width), dtype=torch.bfloat16),
            torch.empty((rows, 2 * second_width)), runtime=backbone, session=None, coordinator=None,
            name='columns', buffers={}, columns=first_width + 2 * second_width + 8)


def test_latent_exchange_rides_the_backbone_with_explicit_geometry(monkeypatch):
    built = []

    def head_exchange(group, device, **kwargs):
        built.append(kwargs)
        return SimpleNamespace(exchange=kwargs)

    monkeypatch.setattr(bootstrap, 'create_head_exchange', head_exchange)
    backbone = object()
    runtimes, rows, width = bootstrap._latent_exchange(backbone, True, None, None, world_size=8,
        latent=264, channel_rows=1024, max_tokens=4096, ipc=None)
    assert runtimes == (backbone,) and not built
    assert rows == 1024 and width == 264
    runtimes, rows, width = bootstrap._latent_exchange(None, False, object(), 'cpu', world_size=8,
        latent=2112 // 8 + 1, channel_rows=1024, max_tokens=4096, ipc=object())
    assert [call['max_rows'] for call in built] == [8, 1024]
    assert all(call['heads'] == 8 for call in built)
    assert built[0]['head_dim'] == built[1]['head_dim'] == -(-(2112 // 8 + 1) // 8) * 8
    assert rows is None and width is None


def test_draft_exchanges_declare_no_pcie_channel_attributes():
    # Stage-2 declaration: the draft-context, draft-attention, and dynamic-convolution
    # exchanges read no DcpAllToAll attributes and transport BF16 columns, so the
    # shared RoCE backbone carries them unchanged (widths come from the draft config).
    import inspect

    from ignition.components import dflash2, grouped_convolution
    from ignition.recipes.kimi31 import dflash

    for module in (dflash, dflash2, grouped_convolution):
        source = inspect.getsource(module)
        for attribute in ('max_batch_size', 'query_head_dim', 'heads_per_rank', 'total_heads'):
            assert attribute not in source, (module.__name__, attribute)


def test_grouped_convolution_gathers_over_the_backbone(monkeypatch):
    from ignition.components import collectives as collectives_module
    from ignition.components import grouped_convolution

    gathers = []
    # PreparedGroupedConvolution imports the chunked gather lazily from the
    # collectives module, so the stub replaces it there.
    monkeypatch.setattr(collectives_module, 'PreparedChunkedAllGather',
                        lambda source, output, **kwargs: gathers.append((source.shape, output.shape, kwargs))
                        or SimpleNamespace())
    monkeypatch.setattr(grouped_convolution, 'UnquantizedProjection',
                        lambda *args, **kwargs: SimpleNamespace())
    backbone = SimpleNamespace(world_size=2)
    tokens, hidden, taps, block, group = 16, 64, 4, 16, 16
    source = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
    weights = {'base': torch.zeros((2, taps, hidden), dtype=torch.bfloat16), 'projection': object()}
    operation = grouped_convolution.PreparedGroupedConvolution(source, weights, block_size=block,
        group_size=group, session=None, name='conv', buffers={}, runtime=backbone)
    shape, output_shape, kwargs = gathers[0]
    channels = 2 * taps * hidden // group
    # Each rank projects half the coefficient channels; the gather carries them all.
    assert shape == (tokens, channels // 2) and output_shape == (tokens, channels)
    assert kwargs['runtime'] is backbone
