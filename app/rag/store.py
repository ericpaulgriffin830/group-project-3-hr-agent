"""Chroma persistent vector store.

One collection, one schema, one place that knows how a Chunk (chunk.py) becomes
a Chroma record and back. scripts/build_index.py calls upsert_chunks() to build
the index; retrieve.py calls vector_search() as the vector leg of its hybrid
search. Nobody else should reach into Chroma directly -- a second call site
constructing its own metadata dict is how the schema drifts from what
get_policy_section and the citation fields (Contract A's PolicyChunk) expect.

CHROMA_PATH (docs/CONTRACTS.md's env table) is the one setting that controls
where the index lives, locally and at deploy: a small on-disk directory,
rebuilt by scripts/build_index.py, never a paid hosted database (rubric section
7 explicitly allows "a small local vector store built during deployment").

Embeddings are computed via embed.py and passed to Chroma explicitly
(collection.upsert(embeddings=...)) rather than letting Chroma call its own
default embedding function. That keeps exactly one embedding call site in the
whole project, so embed.py's document/query asymmetry can't be bypassed by code
that never imports it.

Known limitation, not addressed here: Chroma's HNSW index assigns each inserted
vector's graph level using an internal random draw, so an index rebuilt from the
same chunks can order exact score ties differently across runs even though every
embedding is itself deterministic. This does not affect which documents are
retrieved, only tie-breaking among equally-scored chunks, which is not currently
guarded against.
"""

from __future__ import annotations

import os
from functools import lru_cache

import chromadb

from app.rag.chunk import Chunk
from app.rag.embed import embed_documents, embed_query

COLLECTION_NAME = "policy_chunks"
DEFAULT_CHROMA_PATH = "./data/chroma"


def _chroma_path() -> str:
    return os.getenv("CHROMA_PATH") or DEFAULT_CHROMA_PATH


def _metadata(chunk: Chunk) -> dict:
    """Chroma metadata values must be str/int/float/bool -- every field here is.

    doc_id/title/section/snippet are Contract A's PolicyChunk fields, straight
    through. section_id/source_path/chunk_index/token_count are for internal use
    (doc_filter, debugging, a future get_policy_section-style exact lookup) and
    are never surfaced as citations directly.
    """
    return {
        "doc_id": chunk.doc_id,
        "section_id": chunk.section_id,
        "section": chunk.section,
        "title": chunk.title,
        "source_path": chunk.source_path,
        "snippet": chunk.snippet,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
    }


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    """One persistent client per process, at CHROMA_PATH."""
    path = _chroma_path()
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_collection(client: chromadb.ClientAPI | None = None):
    """The one collection this project uses.

    Cosine distance, matching BGE's training objective -- bge-small-en-v1.5 is
    trained and evaluated with cosine similarity, not raw dot product or
    Euclidean distance, and Chroma's default (L2) would silently misrank results
    against this specific model.
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def reset_collection(client: chromadb.ClientAPI | None = None):
    """Drop and recreate the collection.

    scripts/build_index.py calls this before rebuilding, so a section removed or
    renamed in the corpus does not linger in the index forever -- upsert alone
    only touches ids that still exist in the current chunk set, it never deletes
    ids that no longer do.
    """
    client = client or get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # no prior collection -- fine, get_or_create below makes one
    return get_collection(client)


def upsert_chunks(chunks: list[Chunk], *, batch_size: int = 64,
                  client: chromadb.ClientAPI | None = None) -> int:
    """Embed and upsert every chunk. Returns the number written.

    Idempotent: re-running with the same chunks overwrites the same ids rather
    than duplicating them, because chunk_id is a deterministic digest of
    doc_id/section_id/chunk_index (chunk.py), not a random or insertion-order id.

    Batched because embedding, not the Chroma write, is the expensive step. The
    current 106-chunk corpus would fit in one batch, but the brief allows up to
    120 pages of corpus, and this should not need rewriting if the corpus grows.
    """
    collection = get_collection(client)
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = embed_documents([c.text for c in batch])
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],
            metadatas=[_metadata(c) for c in batch],
        )
        total += len(batch)
    return total


def vector_search(text: str, k: int = 5, doc_filter: list[str] | None = None,
                  client: chromadb.ClientAPI | None = None) -> list[dict]:
    """The vector leg of retrieval. retrieve.py fuses this with a BM25/keyword
    leg via RRF; this function alone is vector-only, matching Contract A's
    "vector_only" retrieval_mode if ever called directly.

    Returns Contract A's PolicyChunk shape: doc_id, title, section, snippet,
    score. `score` is 1 - cosine_distance (higher is better), converted here
    once so every caller -- the orchestrator's citation selection, a future
    ablation script -- can assume higher-is-better without re-deriving it from
    Chroma's raw distance.
    """
    collection = get_collection(client)
    where = {"doc_id": {"$in": doc_filter}} if doc_filter else None
    result = collection.query(
        query_embeddings=[embed_query(text)],
        n_results=k,
        where=where,
        include=["metadatas", "distances"],
    )
    ids = result["ids"][0] if result["ids"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []
    return [
        {
            # chunk_id is NOT part of Contract A's PolicyChunk shape -- it is
            # carried here so retrieve.py can fuse this leg's ranking against
            # the BM25 leg's ranking by a stable id. A caller building the
            # actual tool response should drop it before returning to the
            # agent, the same way it already would not return `score`'s raw
            # Chroma distance.
            "chunk_id": cid,
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "section": meta["section"],
            "snippet": meta["snippet"],
            "score": round(1.0 - dist, 4),
        }
        for cid, meta, dist in zip(ids, metadatas, distances)
    ]


def count(client: chromadb.ClientAPI | None = None) -> int:
    return get_collection(client).count()
