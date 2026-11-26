"""Pinned Speculators leaf heads and adapter predecessor ownership on CPU.

These checks do not compare GPU projection kernels or the attention backbone.
"""
import importlib.util
from pathlib import Path
import sys

import torch

from ignition.components.split_draft import PreparedDraftConfidence

REFERENCE_COMMIT = "05f9ba2324fef505cef8700669f2bdc20f8ae219"
REFERENCE = Path("/tmp/opencode/k31-speculators-05f9ba2324fe-reference/repo/"
                 "src/speculators/models/dspark/model_definitions.py")


def reference_heads():
    spec = importlib.util.spec_from_file_location("pinned_dspark_head_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dyadic(shape, modulus=7):
    count = 1
    for dimension in shape:
        count *= dimension
    return ((torch.arange(count).remainder(modulus) - modulus // 2).float() / 32).view(shape)


@torch.no_grad()
def test_actual_reference_heads_prefixes_and_confidence_predecessors():
    before = set(sys.modules)
    reference = reference_heads()
    assert not any(name == "speculators" or name.startswith("speculators.")
                   for name in set(sys.modules) - before)
    batch, block, hidden_size, rank, vocabulary = 2, 7, 7168, 512, 32
    markov = reference.MarkovHead(verifier_vocab_size=vocabulary, draft_vocab_size=vocabulary,
        markov_rank=rank, hidden_size=hidden_size, head_type="vanilla")
    confidence = reference.ConfidenceHead(hidden_size + rank)
    # BF16-exact stored values, promoted for the canonical FP32 accumulation boundary.
    for parameter in (*markov.parameters(), *confidence.parameters()):
        parameter.copy_(dyadic(parameter.shape).bfloat16().float())
    hidden = dyadic((batch, block, hidden_size)).bfloat16().float()
    # Anchor followed by actual synthetic sampled predecessors, not mask tokens.
    tokens = torch.tensor([[2, 9, 4, 17, 6, 21, 8], [13, 3, 19, 5, 23, 7, 29]])
    embeddings = markov.prev_embeddings(tokens)
    full_bias = markov.block_bias(prev_token_ids=tokens, hidden_states=hidden)
    full_logits = confidence(torch.cat((hidden, embeddings), dim=-1))
    assert full_bias.shape == (batch, block, vocabulary)
    assert full_logits.shape == (batch, block)
    for width in range(1, block + 1):
        prefix_embeddings = markov.prev_embeddings(tokens[:, :width])
        prefix_bias = markov.block_bias(prev_token_ids=tokens[:, :width], hidden_states=hidden[:, :width])
        prefix_logits = confidence(torch.cat((hidden[:, :width], prefix_embeddings), dim=-1))
        assert torch.equal(prefix_bias, full_bias[:, :width])
        assert torch.equal(prefix_logits, full_logits[:, :width])
        assert torch.equal(prefix_logits.sigmoid(), full_logits[:, :width].sigmoid())
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % vocabulary
    changed_embeddings = markov.prev_embeddings(changed)
    changed_bias = markov.block_bias(prev_token_ids=changed, hidden_states=hidden)
    assert torch.equal(changed_bias[:, :4], full_bias[:, :4])
    assert not torch.equal(changed_bias[:, 4:], full_bias[:, 4:])

    # Production observer's CPU copy method snapshots the reusable Markov buffer.
    # Native projection construction is intentionally not invoked on CPU.
    observer = object.__new__(PreparedDraftConfidence)
    observer.canonical_embedding = None
    observer.embeddings = torch.empty((batch, 6, rank), dtype=torch.bfloat16)
    reused = torch.empty((batch, rank), dtype=torch.bfloat16)
    for position in range(6):
        reused.copy_(embeddings[:, position])
        observer.record_embedding(position, reused)
    reused.copy_(changed_embeddings[:, -1])
    assert torch.equal(observer.embeddings.float(), embeddings[:, :6])
    logits_from_retained = confidence(torch.cat((hidden[:, :6], observer.embeddings.float()), dim=-1))
    assert torch.equal(logits_from_retained, full_logits[:, :6])
    # The leaf returns logits. Sigmoid is the existing adapter's forecast boundary.
    probabilities = logits_from_retained.sigmoid()
    assert torch.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert not torch.cuda.is_initialized()
