"""app/rag/retrieve.py -- hybrid BM25 + vector retrieval, fused with RRF.

Two tiers, like test_embed.py / test_store.py before it:

- Pure-logic unit tests for _tokenize, _rrf_fuse, and _cap_per_document need
  no corpus, no model, no Chroma -- they test arithmetic and list-shuffling
  against small, hand-built inputs with known expected outputs.
- Integration tests run retrieve() and get_section() against a real index:
  the actual 12-document corpus, chunked for real, embedded for real, and
  queried for real. This is deliberate -- the whole point of retrieve.py is
  how BM25 and vector search actually interact on THIS corpus, which is
  exactly the thing a mocked index cannot exercise. These are marked
  `@pytest.mark.embedding` (they need the fastembed cache/network, same as
  test_embed.py) and share one module-scoped built index, since building it
  requires embedding all 106 chunks and that cost should be paid once per
  test run, not once per test.

The Colorado regression test below is the one most worth reading: it pins
down the real ranking-quality bug found and fixed while building retrieve.py
(see retrieve.py's module docstring) so it cannot silently come back.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

import app.rag.retrieve as retrieve_module
import app.rag.store as store
from app.rag.chunk import chunk_corpus
from app.rag.ingest import load_corpus
from app.rag.retrieve import (
    MAX_CHUNKS_PER_DOC,
    RRF_K,
    _cap_per_document,
    _rrf_fuse,
    _tokenize,
    get_section,
    retrieve,
)

# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_extracts_alphanumeric():
    assert _tokenize("Remote-Work Policy 2026!") == ["remote", "work", "policy", "2026"]


def test_tokenize_drops_stopwords():
    tokens = _tokenize("What is the policy on this and that for me?")
    assert "policy" in tokens
    for stopword in ("what", "is", "the", "on", "this", "and", "that", "for", "me"):
        assert stopword not in tokens


def test_tokenize_drops_tokens_of_length_2_or_less():
    tokens = _tokenize("Is PTO ok to go up by 5 or 10 units?")
    assert "pto" in tokens
    assert "units" in tokens
    assert "ok" not in tokens  # length 2
    assert "go" not in tokens  # length 2
    assert "up" not in tokens  # length 2


def test_tokenize_empty_string_returns_empty_list():
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# _rrf_fuse
# ---------------------------------------------------------------------------


def test_rrf_fuse_matches_hand_computed_score_with_default_weights():
    # vector: a rank 1, b rank 2. bm25: b rank 1, a rank 2.
    fused = _rrf_fuse(["a", "b"], ["b", "a"])
    expected_a = 0.8 / (RRF_K + 1) + 0.2 / (RRF_K + 2)
    expected_b = 0.8 / (RRF_K + 2) + 0.2 / (RRF_K + 1)
    assert fused["a"] == pytest.approx(expected_a)
    assert fused["b"] == pytest.approx(expected_b)
    # "a" is rank 1 on the heavily-weighted vector leg -- it should win overall
    # even though it is only rank 2 on the bm25 leg.
    assert fused["a"] > fused["b"]


def test_rrf_fuse_id_present_in_only_one_leg_gets_only_that_contribution():
    fused = _rrf_fuse(["x"], [])
    assert fused["x"] == pytest.approx(0.8 / (RRF_K + 1))
    assert "y" not in fused  # never scored, not scored as zero explicitly either


def test_rrf_fuse_custom_weights_and_k():
    fused = _rrf_fuse(["x"], ["x"], vector_weight=0.5, bm25_weight=0.5, k=1)
    assert fused["x"] == pytest.approx(0.5 / 2 + 0.5 / 2)


def test_rrf_fuse_higher_rank_scores_higher_than_lower_rank_same_leg():
    fused = _rrf_fuse(["first", "second", "third"], [])
    assert fused["first"] > fused["second"] > fused["third"]


# ---------------------------------------------------------------------------
# _cap_per_document
# ---------------------------------------------------------------------------


def test_cap_per_document_enforces_cap_before_moving_to_other_docs():
    ranked = ["a1", "a2", "a3", "b1"]
    doc_id_of = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
    picked = _cap_per_document(ranked, doc_id_of, k=3, cap=2)
    # first pass: a1, a2 (A hits cap), b1 -- reaches k=3 before a3 is ever considered
    assert picked == ["a1", "a2", "b1"]


def test_cap_per_document_backfills_from_capped_doc_when_no_alternatives():
    ranked = ["a1", "a2", "a3"]
    doc_id_of = {"a1": "A", "a2": "A", "a3": "A"}
    picked = _cap_per_document(ranked, doc_id_of, k=3, cap=1)
    # only one document exists, so the cap is enforced in the first pass
    # (just a1) and the backfill pass has to pull a2, a3 back in anyway.
    assert picked == ["a1", "a2", "a3"]


def test_cap_per_document_respects_k_even_with_more_candidates_available():
    ranked = ["a1", "a2", "b1", "b2", "c1"]
    doc_id_of = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C"}
    picked = _cap_per_document(ranked, doc_id_of, k=2, cap=2)
    assert len(picked) == 2
    assert picked == ["a1", "a2"]


def test_cap_per_document_default_cap_matches_module_constant():
    ranked = [f"a{i}" for i in range(5)]
    doc_id_of = {cid: "A" for cid in ranked}
    picked = _cap_per_document(ranked, doc_id_of, k=5)
    first_pass_count = min(MAX_CHUNKS_PER_DOC, 5)
    assert picked[:first_pass_count] == ranked[:first_pass_count]
    assert len(picked) == 5  # backfill still reaches k since there's only one doc


# ---------------------------------------------------------------------------
# get_section -- needs the real corpus, not the vector index
# ---------------------------------------------------------------------------


def test_get_section_returns_full_section_for_known_id():
    entry = get_section("PTO-HOLIDAYS", "PTO-3")
    assert entry["doc_id"] == "PTO-HOLIDAYS"
    assert entry["section"] == "PTO-3 Requesting and Approving PTO"
    assert "text" in entry and len(entry["text"]) > 0


def test_get_section_unknown_section_returns_error_payload():
    entry = get_section("PTO-HOLIDAYS", "PTO-99")
    assert entry == {
        "error": "section_not_found",
        "message": "No section 'PTO-99' in document 'PTO-HOLIDAYS'.",
        "doc_id": "PTO-HOLIDAYS",
        "section_id": "PTO-99",
    }


def test_get_section_unknown_document_returns_error_payload():
    entry = get_section("NOT-A-DOC", "SEC-1")
    assert entry["error"] == "section_not_found"


# ---------------------------------------------------------------------------
# Integration: real corpus, real embeddings, real Chroma index.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    """Build the real index once for every test in this module.

    Uses a temp CHROMA_PATH so this never touches (or depends on) the
    project's own ./data/chroma/, and clears every relevant lru_cache before
    building so this module doesn't inherit a stale client, corpus, or BM25
    index left over from an earlier test file or process.
    """
    chroma_path = tmp_path_factory.mktemp("chroma_retrieval")
    previous = os.environ.get("CHROMA_PATH")
    os.environ["CHROMA_PATH"] = str(chroma_path)

    store.get_client.cache_clear()
    retrieve_module._corpus_chunks.cache_clear()
    retrieve_module._bm25_index.cache_clear()
    retrieve_module._section_lookup.cache_clear()

    store.reset_collection()
    store.upsert_chunks(chunk_corpus(load_corpus()))

    yield

    store.get_client.cache_clear()
    if previous is None:
        os.environ.pop("CHROMA_PATH", None)
    else:
        os.environ["CHROMA_PATH"] = previous


@pytest.mark.embedding
def test_retrieve_known_query_returns_known_document_in_top_k(built_index):
    result = retrieve("How much paid time off do new employees accrue?", k=5)
    assert result["retrieval_mode"] == "hybrid"
    doc_ids = [c["doc_id"] for c in result["chunks"]]
    assert "PTO-HOLIDAYS" in doc_ids


@pytest.mark.embedding
def test_retrieve_is_deterministic_across_repeated_calls(built_index):
    query = "Can I work from Colorado for six weeks?"
    first = retrieve(query, k=5)
    second = retrieve(query, k=5)
    assert first == second


@pytest.mark.embedding
def test_retrieve_colorado_regression_remote_work_and_tax_outrank_leave(built_index):
    # The exact failure this hybrid weighting was built to fix (see
    # retrieve.py's module docstring): unweighted RRF let LEAVE-3 (Parental
    # Leave) outrank REMOTE-WORK for this query, purely on generic token
    # overlap ("work", "weeks"). With the 0.8/0.2 weighting, REMOTE-WORK and
    # TAX-LOCATION -- the actually relevant documents -- should lead, and
    # LEAVE should not be among the top 3.
    result = retrieve("Can I work from Colorado for six weeks?", k=5)
    doc_ids = [c["doc_id"] for c in result["chunks"]]
    assert doc_ids[:3].count("LEAVE") == 0
    assert set(doc_ids[:3]) <= {"REMOTE-WORK", "TAX-LOCATION"}


@pytest.mark.embedding
def test_retrieve_keyword_heavy_control_still_favors_exact_term_match(built_index):
    # The control query from the same empirical test: weighting the fusion
    # toward vector search should not eliminate BM25's contribution entirely.
    # A query built from exact corpus vocabulary should still surface the
    # document that vocabulary belongs to.
    result = retrieve("nexus tax registration", k=5)
    doc_ids = [c["doc_id"] for c in result["chunks"]]
    assert doc_ids[0] == "TAX-LOCATION"


@pytest.mark.embedding
def test_retrieve_no_single_document_exceeds_cap(built_index):
    result = retrieve("Tell me everything about remote work policy", k=8)
    doc_ids = [c["doc_id"] for c in result["chunks"]]
    counts = Counter(doc_ids)
    assert all(count <= MAX_CHUNKS_PER_DOC for count in counts.values())


@pytest.mark.embedding
def test_retrieve_doc_filter_restricts_to_named_documents(built_index):
    result = retrieve("policy", k=5, doc_filter=["PTO-HOLIDAYS"])
    assert all(c["doc_id"] == "PTO-HOLIDAYS" for c in result["chunks"])


@pytest.mark.embedding
def test_retrieve_vector_only_mode_skips_bm25_and_sets_mode(built_index):
    result = retrieve("nexus tax registration", k=5, mode="vector_only")
    assert result["retrieval_mode"] == "vector_only"
    assert len(result["chunks"]) > 0


@pytest.mark.embedding
def test_retrieve_empty_query_returns_no_chunks(built_index):
    result = retrieve("   ")
    assert result["chunks"] == []


@pytest.mark.embedding
def test_retrieve_off_topic_query_returns_no_chunks(built_index):
    # Regression: retrieve() replaced the fixture-phase keyword ranker, which
    # dropped every zero-score chunk and so came back empty for a query with no
    # real vocabulary overlap. The vector leg has no floor of its own -- it
    # always returns its k nearest neighbors, however weak the match -- so a
    # genuinely off-topic query stopped coming back empty, and the
    # orchestrator's "no evidence -> refuse rather than guess" guard
    # (app/agent/orchestrator.py) could never fire on the RAG-only path. See
    # MIN_VECTOR_RELEVANCE's docstring for the measured score separation this
    # floor is based on.
    assert retrieve("unanswerable", k=5)["chunks"] == []
    assert retrieve("asdf jkl qwerty zzzz", k=5)["chunks"] == []


@pytest.mark.embedding
def test_retrieve_single_real_word_query_is_rescued_by_a_bm25_hit(built_index):
    # A short, single-word real query can score as low on the vector leg as
    # genuine nonsense does -- MIN_VECTOR_RELEVANCE alone would wrongly gate it
    # out. A real corpus word always gives BM25 a positive keyword match, which
    # is exactly the rescue the "and not bm25_ids" half of the gate exists for.
    for query in ("dental", "tax", "vacation"):
        assert len(retrieve(query, k=5)["chunks"]) > 0, query


@pytest.mark.embedding
def test_retrieve_chunks_have_policy_chunk_shape_only(built_index):
    # chunk_id is an internal fusion key (store.py, retrieve.py's
    # _to_policy_chunk) and must never leak into the tool-facing response.
    result = retrieve("remote work", k=3)
    for chunk in result["chunks"]:
        assert set(chunk.keys()) == {"doc_id", "title", "section", "snippet", "score"}


@pytest.mark.embedding
def test_retrieve_hybrid_mode_reports_the_real_fused_score_not_zero(built_index):
    # Regression: _rrf_fuse computed fused_scores and retrieve() sorted and
    # capped by them, but never wrote the fused value back onto the chunk
    # dict before _to_policy_chunk stripped it -- every hybrid-mode chunk
    # reported score=0.0 regardless of its actual rank (found by Chris while
    # tracing why answer.py's confidence heuristic, which averages the score
    # of cited chunks, always returned 0). Every score here must be a real,
    # positive RRF value, and they must already be in the order the chunks
    # were returned in (retrieve() sorts by this score before capping).
    result = retrieve("FMLA leave PTO accrual during leave", k=5)
    scores = [c["score"] for c in result["chunks"]]
    assert all(isinstance(s, float) and s > 0.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
