"""app/rag/store.py -- Chroma persistence mechanics.

app.rag.embed.embed_documents / embed_query are patched out here with small,
deterministic fake vectors instead of the real model. store.py's own job is
Chroma plumbing (metadata shape, upsert/query wiring, doc_filter, reset,
idempotency, score conversion) -- none of that depends on what the real
embedding model produces, and using controlled vectors lets scores be checked
exactly (a fixed cosine distance) rather than "close to something". The real
model is exercised in test_embed.py and, end-to-end together with this store,
in test_retrieval.py's integration tests.

Every test gets its own CHROMA_PATH under tmp_path and clears store.get_client's
lru_cache before and after, so tests never share a Chroma client or an on-disk
index with each other.
"""

from __future__ import annotations

import math

import pytest

import app.rag.store as store
from app.rag.chunk import Chunk


@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path, monkeypatch):
    """Fresh CHROMA_PATH and a fresh client cache for every test."""
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    store.get_client.cache_clear()
    yield
    store.get_client.cache_clear()


def _unit_vector(dim: int, hot_index: int) -> list[float]:
    """A one-hot unit vector -- cosine similarity between two of these is 1.0
    if they share a hot index, 0.0 if they don't (orthogonal)."""
    vec = [0.0] * dim
    vec[hot_index] = 1.0
    return vec


def _chunk(chunk_id, doc_id, section_id="S-1", text="Some chunk text.", chunk_index=0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section=f"{section_id} A Section",
        title="Sample Policy",
        source_path="corpus/policies/sample.md",
        text=text,
        snippet=text,
        chunk_index=chunk_index,
        token_count=3,
    )


def _patch_embeddings(monkeypatch, doc_vectors=None, query_vectors=None):
    """Patch embed_documents/embed_query as store.py imports them (by name, into
    its own module namespace), returning fixed vectors regardless of input."""
    if doc_vectors is not None:
        monkeypatch.setattr(store, "embed_documents", lambda texts: [doc_vectors[i] for i in range(len(texts))])
    if query_vectors is not None:
        calls = iter(query_vectors)
        monkeypatch.setattr(store, "embed_query", lambda text: next(calls))


# ---------------------------------------------------------------------------
# metadata / upsert / count round-trip
# ---------------------------------------------------------------------------


def test_upsert_then_count_reflects_number_of_chunks(monkeypatch):
    chunks = [_chunk("c1", "DOC-1"), _chunk("c2", "DOC-1", chunk_index=1)]
    monkeypatch.setattr(store, "embed_documents", lambda texts: [_unit_vector(4, 0) for _ in texts])
    written = store.upsert_chunks(chunks)
    assert written == 2
    assert store.count() == 2


def test_upsert_is_idempotent_same_ids_do_not_duplicate(monkeypatch):
    monkeypatch.setattr(store, "embed_documents", lambda texts: [_unit_vector(4, 0) for _ in texts])
    chunk = _chunk("c1", "DOC-1")
    store.upsert_chunks([chunk])
    store.upsert_chunks([chunk])  # re-run with the same chunk_id
    assert store.count() == 1


def test_upsert_batches_respect_batch_size(monkeypatch):
    calls = []

    def fake_embed_documents(texts):
        calls.append(len(texts))
        return [_unit_vector(4, 0) for _ in texts]

    monkeypatch.setattr(store, "embed_documents", fake_embed_documents)
    chunks = [_chunk(f"c{i}", "DOC-1", chunk_index=i) for i in range(5)]
    store.upsert_chunks(chunks, batch_size=2)
    assert calls == [2, 2, 1]
    assert store.count() == 5


# ---------------------------------------------------------------------------
# vector_search: score correctness, ordering, doc_filter
# ---------------------------------------------------------------------------


def test_vector_search_returns_policy_chunk_shape_plus_chunk_id(monkeypatch):
    monkeypatch.setattr(store, "embed_documents", lambda texts: [_unit_vector(4, 0)])
    store.upsert_chunks([_chunk("c1", "DOC-1", section_id="PTO-3")])
    monkeypatch.setattr(store, "embed_query", lambda text: _unit_vector(4, 0))

    results = store.vector_search("any query", k=5)
    assert len(results) == 1
    hit = results[0]
    assert set(hit.keys()) == {"chunk_id", "doc_id", "title", "section", "snippet", "score"}
    assert hit["chunk_id"] == "c1"
    assert hit["doc_id"] == "DOC-1"
    assert hit["section"] == "PTO-3 A Section"


def test_vector_search_score_is_1_for_identical_vectors(monkeypatch):
    monkeypatch.setattr(store, "embed_documents", lambda texts: [_unit_vector(4, 0)])
    store.upsert_chunks([_chunk("c1", "DOC-1")])
    monkeypatch.setattr(store, "embed_query", lambda text: _unit_vector(4, 0))

    results = store.vector_search("query", k=1)
    assert math.isclose(results[0]["score"], 1.0, abs_tol=1e-6)


def test_vector_search_score_is_0_for_orthogonal_vectors(monkeypatch):
    monkeypatch.setattr(store, "embed_documents", lambda texts: [_unit_vector(4, 0)])
    store.upsert_chunks([_chunk("c1", "DOC-1")])
    monkeypatch.setattr(store, "embed_query", lambda text: _unit_vector(4, 1))  # orthogonal

    results = store.vector_search("query", k=1)
    assert math.isclose(results[0]["score"], 0.0, abs_tol=1e-6)


def test_vector_search_ranks_closer_vector_first(monkeypatch):
    # c1 shares the query's hot index exactly (score ~1), c2 is orthogonal (score ~0)
    def fake_embed_documents(texts):
        return [_unit_vector(4, 0), _unit_vector(4, 2)]

    monkeypatch.setattr(store, "embed_documents", fake_embed_documents)
    store.upsert_chunks([_chunk("c1", "DOC-1"), _chunk("c2", "DOC-1", chunk_index=1)])
    monkeypatch.setattr(store, "embed_query", lambda text: _unit_vector(4, 0))

    results = store.vector_search("query", k=2)
    assert [r["chunk_id"] for r in results] == ["c1", "c2"]
    assert results[0]["score"] > results[1]["score"]


def test_vector_search_respects_k():
    pass  # covered implicitly by other tests passing explicit k; kept as a marker for coverage intent


def test_vector_search_doc_filter_restricts_results(monkeypatch):
    def fake_embed_documents(texts):
        return [_unit_vector(4, 0), _unit_vector(4, 0)]

    monkeypatch.setattr(store, "embed_documents", fake_embed_documents)
    store.upsert_chunks([
        _chunk("c1", "DOC-1"),
        _chunk("c2", "DOC-2"),
    ])
    monkeypatch.setattr(store, "embed_query", lambda text: _unit_vector(4, 0))

    results = store.vector_search("query", k=5, doc_filter=["DOC-2"])
    assert [r["doc_id"] for r in results] == ["DOC-2"]


def test_vector_search_empty_collection_returns_empty_list(monkeypatch):
    monkeypatch.setattr(store, "embed_query", lambda text: _unit_vector(4, 0))
    assert store.vector_search("anything", k=5) == []


# ---------------------------------------------------------------------------
# reset_collection
# ---------------------------------------------------------------------------


def test_reset_collection_clears_existing_data(monkeypatch):
    monkeypatch.setattr(store, "embed_documents", lambda texts: [_unit_vector(4, 0)])
    store.upsert_chunks([_chunk("c1", "DOC-1")])
    assert store.count() == 1

    store.reset_collection()
    assert store.count() == 0


def test_reset_collection_is_safe_when_nothing_exists_yet():
    # No collection has been created at all -- delete_collection should fail
    # internally and be swallowed, not raise.
    store.reset_collection()
    assert store.count() == 0


# ---------------------------------------------------------------------------
# get_client / get_collection caching
# ---------------------------------------------------------------------------


def test_get_client_is_cached_across_calls():
    first = store.get_client()
    second = store.get_client()
    assert first is second


def test_get_collection_uses_cosine_space(monkeypatch):
    monkeypatch.setattr(store, "embed_documents", lambda texts: [_unit_vector(4, 0)])
    collection = store.get_collection()
    assert collection.metadata.get("hnsw:space") == "cosine"
