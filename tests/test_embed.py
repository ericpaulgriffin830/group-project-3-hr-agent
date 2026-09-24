"""app/rag/embed.py -- the shared fastembed client.

Two kinds of tests here, deliberately kept apart:

- Pure logic (_model_name / _cache_dir env resolution) needs no model and no
  network, and runs everywhere, every time.
- Everything that calls the real embedder is marked `@pytest.mark.embedding`.
  These need the actual BAAI/bge-small-en-v1.5 weights, downloaded once from
  Hugging Face and cached under .fastembed_cache/ (see embed.py's module
  docstring on why that download can't happen from a network-restricted
  sandbox). Mocking fastembed out would only prove our own mock behaves the
  way we told it to -- it would not catch a real regression in dimension,
  determinism, or the document/query asymmetry the whole retrieval design
  depends on. So these run for real, against whatever cache/network is
  available wherever the suite runs (dev machine, CI, Render's build step).
"""

from __future__ import annotations

import pytest

import app.rag.embed as embed_module
from app.rag.embed import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL,
    _cache_dir,
    _model_name,
    embed_documents,
    embed_query,
    embedding_dim,
)


# ---------------------------------------------------------------------------
# Pure logic: env var resolution. No model, no network.
# ---------------------------------------------------------------------------


def test_model_name_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    assert _model_name() == DEFAULT_MODEL


def test_model_name_reads_env_override(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "some/other-model")
    assert _model_name() == "some/other-model"


def test_cache_dir_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("EMBED_CACHE_DIR", raising=False)
    assert _cache_dir() == DEFAULT_CACHE_DIR


def test_cache_dir_reads_env_override(monkeypatch):
    monkeypatch.setenv("EMBED_CACHE_DIR", "/tmp/some-cache")
    assert _cache_dir() == "/tmp/some-cache"


def test_model_name_empty_string_env_falls_back_to_default(monkeypatch):
    # os.getenv(...) or DEFAULT -- an empty string is falsy, so this should
    # fall back rather than silently trying to load a model named "".
    monkeypatch.setenv("EMBED_MODEL", "")
    assert _model_name() == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Real model. Needs the fastembed cache (locally warmed) or network access.
# ---------------------------------------------------------------------------


@pytest.mark.embedding
def test_embedding_dim_is_384_for_bge_small():
    assert embedding_dim() == 384


@pytest.mark.embedding
def test_embed_documents_returns_one_vector_per_text_in_order():
    texts = ["PTO accrues monthly.", "Remote work requires manager approval."]
    vectors = embed_documents(texts)
    assert len(vectors) == 2
    assert all(len(v) == embedding_dim() for v in vectors)
    # order-preserving: embedding the same two texts reversed should reverse
    # which vector is which (they are not identical texts, so not identical
    # vectors either).
    reversed_vectors = embed_documents(list(reversed(texts)))
    assert vectors[0] == reversed_vectors[1]
    assert vectors[1] == reversed_vectors[0]


@pytest.mark.embedding
def test_embed_query_returns_single_vector_of_correct_dimension():
    vector = embed_query("How much PTO do I have?")
    assert len(vector) == embedding_dim()


@pytest.mark.embedding
def test_embed_documents_is_deterministic():
    text = "Employees accrue PTO at a rate of 1.5 days per month."
    first = embed_documents([text])[0]
    second = embed_documents([text])[0]
    assert first == second


@pytest.mark.embedding
def test_embed_query_is_deterministic():
    query = "Can I work remotely from another state?"
    first = embed_query(query)
    second = embed_query(query)
    assert first == second


@pytest.mark.embedding
def test_query_and_document_embeddings_of_same_text_are_currently_identical():
    # embed.py's docstring claims fastembed applies a retrieval instruction
    # inside query_embed() that embed() does not, for BAAI/bge-small-en-v1.5.
    # That is not what actually happens in fastembed 0.8.1: OnnxTextEmbedding's
    # query_embed() for THIS model just calls self.embed() directly -- no
    # prefix is added. fastembed's own model registry documents why (its
    # description for bge-small-en-v1.5 says prefixes are "not so necessary",
    # versus "necessary" for the older bge-small-en). embed_documents() and
    # embed_query() are still worth keeping as two functions -- it protects
    # against a future model swap (e.g. to a model that DOES need the prefix)
    # silently reintroducing the query/document mix-up bug the split was meant
    # to prevent -- but as currently wired, identical text embeds identically
    # either way. This test pins that fact down: if it ever starts failing, the
    # asymmetry has become real (e.g. after a fastembed upgrade or model swap)
    # and embed.py's docstring, which currently overstates the asymmetry, would
    # then also become accurate and need no correction.
    text = "What is the policy on parental leave?"
    doc_vector = embed_documents([text])[0]
    query_vector = embed_query(text)
    assert doc_vector == query_vector


@pytest.mark.embedding
def test_embedder_is_cached_across_calls():
    # _embedder() is lru_cache(maxsize=1) -- calling it twice should return
    # the same instance, not reload the model.
    first = embed_module._embedder()
    second = embed_module._embedder()
    assert first is second
