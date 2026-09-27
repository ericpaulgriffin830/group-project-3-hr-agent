"""Hybrid retrieval: BM25 keyword search + vector search, fused with
Reciprocal Rank Fusion (RRF). Backs Contract A's search_policy_documents.

Hybrid, not vector-only, because docs/DEMO-TASKS.md's real traces against the
fixture-phase keyword ranker show two concrete failures this corpus produces:
Task A needed 5 reformulated searches to connect "another state" with "nexus"
(same underlying question, no shared vocabulary), and Task B needed 4
near-duplicate rephrasings of "insufficient balance" for the same reason.
Vector search closes that gap. BM25 stays alongside it because exact term
matches -- a section id typed verbatim, "FMLA", a specific policy number --
are something a dense embedding can under-rank relative to lexical search.
Fusing both is why retrieval_mode is "hybrid", not "vector_only".

RRF combines the two rankings by RANK, not raw score, because BM25 scores and
cosine similarities live on different, incomparable scales -- there is no
principled way to add them directly. But unweighted RRF (equal trust in both
legs) was tested against this corpus and produced a real failure: for
"Can I work from Colorado for six weeks?", BM25 ranked LEAVE-3 (Parental
Leave) ahead of REMOTE-WORK's actually-relevant sections, purely because
"work" and "weeks" are generic tokens LEAVE-3 happens to use often (repeated
discussion of "12 weeks", "6 weeks", "2-week blocks"). The vector leg
correctly ranked REMOTE-WORK's RW-3 at position 2; equal-weight RRF still let
BM25's noise drag LEAVE-3 above it. Weighting the fusion toward the vector
leg (VECTOR_WEIGHT / BM25_WEIGHT below) fixed this and was verified against
three queries -- the Colorado case above, the PTO-insufficient-balance case
from docs/DEMO-TASKS.md, and a keyword-heavy control ("nexus tax
registration", which should and does still favor exact-term BM25 matching --
the weighting reduces BM25's influence, it does not remove it).

A per-document cap (MAX_CHUNKS_PER_DOC) keeps one document's many sections
from filling every result slot for a multi-document question. It matches
mcp_server/fixtures.py's own cap exactly, and for the same reason: without
it, REMOTE-WORK's nine sections (all containing "work" and "remote") can
fill every slot before the tax policy or infosec ever get a chance, which is
fatal to rubric item 3's required multi-document question.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.rag.chunk import Chunk, chunk_corpus
from app.rag.ingest import load_corpus
from app.rag.store import vector_search

#: From the original RRF paper (Cormack, Clarke & Buettcher, 2009) and the
#: value used almost everywhere hybrid search is described. Not tuned against
#: this corpus -- it controls how sharply rank position matters, not the
#: balance between the two legs, which is what VECTOR_WEIGHT/BM25_WEIGHT
#: below are for.
RRF_K = 60

#: Empirically chosen (see module docstring): 0.8/0.2 was the smallest shift
#: toward the vector leg that fixed the Colorado/REMOTE-WORK case without
#: giving up BM25's contribution entirely -- the "nexus" control query still
#: favors exact-term matches at this weighting, it is not vector-only in
#: disguise.
VECTOR_WEIGHT = 0.8
BM25_WEIGHT = 0.2

#: Matches mcp_server/fixtures.py's MAX_CHUNKS_PER_DOC exactly.
MAX_CHUNKS_PER_DOC = 2

#: Candidates each leg contributes before fusion and the per-doc cap are
#: applied -- wider than k so the cap has real alternatives to back-fill
#: from, rather than fusing only exactly k candidates from each side.
CANDIDATE_POOL = 20

#: Relevance floor on the vector leg's raw cosine similarity (before RRF
#: fusion -- RRF's own output is a rank-based score, not a magnitude, so it
#: cannot tell a strong match from the least-bad of a pool of noise; only the
#: leg's own pre-fusion score can). Unlike the fixture-phase keyword ranker
#: this module replaced (mcp_server/fixtures.py's rank_chunks drops any
#: chunk scoring <= 0), fastembed's vector search has no floor of its own --
#: it always returns its k nearest neighbors, however weak the actual match.
#: A query this corpus has nothing to do with therefore never came back
#: empty, and the orchestrator's "no evidence -> refuse rather than guess"
#: guard (app/agent/orchestrator.py) could never fire on the RAG-only path.
#:
#: Chosen from measured separation, not a round number: every full-sentence
#: question in both eval sets (evaluation/eval_set.*.json), including the
#: hardest multi-document ones, scores >= 0.67 on the vector leg alone.
#: Nonsense strings ("unanswerable", keyboard mashing, an off-topic sentence)
#: top out around 0.56. A short, single real-word query ("dental", "tax") can
#: fall as low as 0.56 too, inside the nonsense range -- but a real corpus
#: word always gives BM25 a positive keyword match, so retrieve() only
#: refuses a query that clears NEITHER leg (below this floor on vector AND
#: zero BM25 hits) -- see the relevance gate in retrieve() below.
#:
#: Known gap, not fixed here: the BM25 rescue is "any hit at all", and in a
#: 106-chunk corpus a globally common English word absent from _STOPWORDS
#: (e.g. "like") can be rare enough WITHIN the corpus to get a real, sizeable
#: BM25 score off a single incidental match -- "I like turtles and
#: spaceships" clears the gate this way (BM25 hit on "like" alone), even
#: though nothing about the query is HR-related. Narrowing this would mean
#: growing _STOPWORDS or requiring more than one surviving token to match,
#: which risks re-breaking the single-real-word rescue this constant exists
#: for ("dental", "tax") and needs its own tuning pass against the corpus,
#: not a quick addition here.
MIN_VECTOR_RELEVANCE = 0.6

_STOPWORDS = frozenset((
    "the", "and", "for", "that", "this", "with", "from", "have", "can", "are",
    "what", "when", "does", "do", "i", "my", "me", "a", "an", "is", "it", "to",
    "of", "in", "on", "if", "be", "as", "at", "or", "any",
))


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) > 2 and t not in _STOPWORDS]


@lru_cache(maxsize=1)
def _corpus_chunks() -> tuple[Chunk, ...]:
    """The full chunk set, built once per process and cached.

    scripts/build_index.py maintains the persistent Chroma store separately;
    this cache is only for the in-process BM25 index, which has no on-disk
    persistence of its own. Rebuilding it costs nothing at this corpus size
    (106 chunks), so it does not need a store of its own.
    """
    return tuple(chunk_corpus(load_corpus()))


@lru_cache(maxsize=1)
def _bm25_index() -> tuple[BM25Okapi, tuple[Chunk, ...]]:
    chunks = _corpus_chunks()
    return BM25Okapi([_tokenize(c.text) for c in chunks]), chunks


def _bm25_ranked_ids(query: str, doc_filter: list[str] | None,
                     pool: int) -> tuple[list[str], dict[str, Chunk]]:
    """Returns (ranked chunk_ids, chunk_id -> Chunk), best score first.

    Zero-score chunks are dropped -- BM25 gives every document *some* score
    once a query has any token overlap with the corpus vocabulary at all, and
    a zero match is not evidence, it's the absence of any.
    """
    bm25, chunks = _bm25_index()
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
    if doc_filter:
        ranked = [(c, s) for c, s in ranked if c.doc_id in doc_filter]
    kept = [(c, s) for c, s in ranked if s > 0][:pool]
    return [c.chunk_id for c, _ in kept], {c.chunk_id: c for c, _ in kept}


def _vector_ranked(query: str, doc_filter: list[str] | None,
                   pool: int) -> list[dict]:
    """Contract A PolicyChunk-shaped dicts (plus chunk_id), already ranked."""
    return vector_search(query, k=pool, doc_filter=doc_filter)


def _rrf_fuse(vector_ids: list[str], bm25_ids: list[str], *,
             vector_weight: float = VECTOR_WEIGHT,
             bm25_weight: float = BM25_WEIGHT, k: int = RRF_K) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion over the two legs.

    score(id) = vector_weight / (k + rank_in_vector_list), plus
                bm25_weight / (k + rank_in_bm25_list),
    rank counted from 1, either term omitted if the id is absent from that
    list. An id missing from one leg contributes 0 from it rather than a
    penalty: a chunk only vector search found is not punished for BM25 never
    treating it as a keyword match.
    """
    fused: dict[str, float] = defaultdict(float)
    for rank, key in enumerate(vector_ids, start=1):
        fused[key] += vector_weight / (k + rank)
    for rank, key in enumerate(bm25_ids, start=1):
        fused[key] += bm25_weight / (k + rank)
    return fused


def _cap_per_document(ranked_ids: list[str], doc_id_of: dict[str, str],
                      k: int, cap: int = MAX_CHUNKS_PER_DOC) -> list[str]:
    """Up to `cap` per document in ranked order, then back-fill from the
    remainder if fewer than k were picked this way.

    Mirrors mcp_server/fixtures.py's rank_chunks() two-pass approach exactly,
    for the same reason: a single relevant document is still allowed to
    dominate when it genuinely is the only relevant one, but the cap binds
    the moment other documents are also competing for a slot.
    """
    picked: list[str] = []
    per_doc: dict[str, int] = {}
    for cid in ranked_ids:
        doc_id = doc_id_of[cid]
        if per_doc.get(doc_id, 0) >= cap:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        picked.append(cid)
        if len(picked) == k:
            return picked
    for cid in ranked_ids:
        if cid not in picked:
            picked.append(cid)
            if len(picked) == k:
                break
    return picked[:k]


def _to_policy_chunk(entry: dict) -> dict:
    """Strip to Contract A's PolicyChunk fields -- chunk_id is an internal
    fusion key, never part of the tool-facing response."""
    return {k: entry[k] for k in ("doc_id", "title", "section", "snippet", "score")}


def retrieve(query: str, k: int = 5, doc_filter: list[str] | None = None,
            mode: str = "hybrid") -> dict:
    """Contract A's search_policy_documents, for real.

    Returns {"chunks": [...], "retrieval_mode": mode} -- the exact response
    shape mcp_server/server.py's fixture-phase tool already returns, so
    wiring this in place of fixtures.rank_chunks() changes nothing downstream.

    mode="vector_only" skips the BM25 leg and RRF fusion entirely. It exists
    for the retrieval ablation (hybrid vs. vector-only is one of the required
    comparisons, per Group_Assignment.md), not for production use.
    """
    if not query or not query.strip():
        return {"chunks": [], "retrieval_mode": mode}

    pool = max(k * 4, CANDIDATE_POOL)
    vector_hits = _vector_ranked(query, doc_filter, pool)
    by_id: dict[str, dict] = {v["chunk_id"]: v for v in vector_hits}
    doc_id_of: dict[str, str] = {cid: v["doc_id"] for cid, v in by_id.items()}
    vector_ids = [v["chunk_id"] for v in vector_hits]

    # Relevance gate (see MIN_VECTOR_RELEVANCE above): computed for BOTH modes,
    # including vector_only, because the gate is about whether the query has
    # anything to do with this corpus at all, not about which ranking method
    # is under test. Cheap either way -- _bm25_index() is process-cached.
    bm25_ids, bm25_chunks = _bm25_ranked_ids(query, doc_filter, pool)
    top_vector_score = vector_hits[0]["score"] if vector_hits else 0.0
    if top_vector_score < MIN_VECTOR_RELEVANCE and not bm25_ids:
        return {"chunks": [],
                "retrieval_mode": "vector_only" if mode == "vector_only" else "hybrid"}

    if mode == "vector_only":
        fused_order = vector_ids
    else:
        for cid, chunk in bm25_chunks.items():
            doc_id_of.setdefault(cid, chunk.doc_id)
        fused_scores = _rrf_fuse(vector_ids, bm25_ids)
        fused_order = sorted(fused_scores, key=lambda cid: fused_scores[cid],
                             reverse=True)

        # A chunk BM25 found but the vector leg didn't return within `pool`
        # has no PolicyChunk entry yet -- fetch its fields from the Chunk
        # record directly so it can still be cited if RRF ranks it into the
        # final k.
        for cid in fused_order:
            if cid not in by_id and cid in bm25_chunks:
                c = bm25_chunks[cid]
                by_id[cid] = {
                    "chunk_id": cid, "doc_id": c.doc_id, "title": c.title,
                    "section": c.section, "snippet": c.snippet, "score": 0.0,
                }

        # Write the fused RRF score back onto each chunk dict so it is the
        # number search_policy_documents actually reports, rather than
        # whatever the vector leg's raw cosine score happened to leave (or
        # the 0.0 placeholder set just above for a BM25-only backfill).
        # fused_order/_cap_per_document already rank by fused_scores; this
        # just makes the exposed "score" field agree with that ranking --
        # without it every hybrid-mode chunk reports the wrong score, and
        # answer.py's confidence heuristic (which averages cited chunks'
        # scores) reads all zeros no matter how well-supported the answer is.
        for cid, score in fused_scores.items():
            if cid in by_id:
                by_id[cid]["score"] = score

    capped_ids = _cap_per_document(fused_order, doc_id_of, k)
    chunks = [_to_policy_chunk(by_id[cid]) for cid in capped_ids if cid in by_id]
    return {"chunks": chunks, "retrieval_mode": "vector_only" if mode == "vector_only" else "hybrid"}


# ------------------------------------------------------- get_policy_section

@lru_cache(maxsize=1)
def _section_lookup() -> dict[tuple[str, str], dict]:
    """doc_id, section_id -> full (unchunked) section text and its title.

    Built from ingest.py's RawSection, not chunk.py's Chunk records, because
    get_policy_section (Contract A tool #2) is meant to return the WHOLE
    section regardless of how many chunks it was split into for retrieval --
    exactly the property that lets chunk.py skip overlap between sub-chunks
    in the first place (see chunk.py's module docstring).
    """
    lookup: dict[tuple[str, str], dict] = {}
    for doc in load_corpus():
        for section in doc.sections:
            lookup[(doc.doc_id, section.section_id)] = {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "section": f"{section.section_id} {section.heading}",
                "text": section.text,
            }
    return lookup


def get_section(doc_id: str, section_id: str) -> dict:
    """Contract A's get_policy_section, for real. Returns an error payload in
    the same shape as every other tool (never raises), matching the
    error convention mcp_server/server.py already established.
    """
    entry = _section_lookup().get((doc_id, section_id))
    if entry is None:
        return {"error": "section_not_found",
                "message": f"No section '{section_id}' in document '{doc_id}'.",
                "doc_id": doc_id, "section_id": section_id}
    return entry
