"""Shared embedding client. Every embedding call in this project goes through here.

Why one module: retrieval quality and the ablation (k values, chunk sizes) are
only comparable across the week if the model, its cache location, and how
queries versus documents get embedded are set in exactly one place -- the same
reasoning app/llm.py already states for the LLM client.

fastembed replaces sentence-transformers here (see docs/DEPLOYMENT-NOTES.md and
pyproject.toml) -- same model id, a fraction of the install footprint.
EMBED_MODEL defaults to BAAI/bge-small-en-v1.5, pinned in docs/CONTRACTS.md.

Documents and queries are NOT embedded the same way. BGE models are trained with
an asymmetric convention: a passage is embedded as-is, but a query is embedded
with an added instruction ("Represent this sentence for searching relevant
passages: ...") that measurably improves retrieval against passages embedded
without it. fastembed applies that instruction internally in `query_embed()`;
`embed()` does not add it. A query run through the wrong method is not merely
suboptimal -- it degrades silently, with no error to catch it -- so this module
exposes `embed_documents()` and `embed_query()` as two distinct functions rather
than one function serving both, so a call site cannot pick the wrong one.

Model weights download from Hugging Face on first use and cache under
EMBED_CACHE_DIR (default .fastembed_cache/, git-ignored). That download could
not be verified from inside this environment -- Hugging Face is not on this
sandbox's network allowlist, confirmed by a direct attempt failing with a proxy
403. Run `uv run python -m app.rag.embed` locally (or in CI/Render's build step,
both of which should reach huggingface.co normally) before relying on this.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # Contract D / EMBED_MODEL
DEFAULT_CACHE_DIR = ".fastembed_cache"


def _model_name() -> str:
    return os.getenv("EMBED_MODEL") or DEFAULT_MODEL


def _cache_dir() -> str:
    return os.getenv("EMBED_CACHE_DIR") or DEFAULT_CACHE_DIR


@lru_cache(maxsize=1)
def _embedder() -> TextEmbedding:
    """One model instance per process. Loading it is the expensive part --
    scripts/build_index.py and any future long-lived server process should each
    pay that cost once, not per call."""
    return TextEmbedding(model_name=_model_name(), cache_dir=_cache_dir())


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed policy chunks for storage. Order-preserving."""
    return [vec.tolist() for vec in _embedder().embed(list(texts))]


def embed_query(text: str) -> list[float]:
    """Embed one user query, with BGE's retrieval instruction applied."""
    return next(iter(_embedder().query_embed([text]))).tolist()


def embedding_dim() -> int:
    return TextEmbedding.get_embedding_size(_model_name())


if __name__ == "__main__":
    # Manual smoke test -- run locally, not inside a network-restricted sandbox.
    #   uv run python -m app.rag.embed
    docs = embed_documents(["PTO accrues monthly.", "Remote work requires approval."])
    q = embed_query("How much PTO do I have?")
    print(f"model={_model_name()} dim={embedding_dim()}")
    print(f"{len(docs)} document vectors, length {len(docs[0])}")
    print(f"query vector length {len(q)}")
