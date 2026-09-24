"""Shared embedding client. Every embedding call in this project goes through here.

Why one module: retrieval quality and the ablation (k values, chunk sizes) are
only comparable across the week if the model, its cache location, and how
queries versus documents get embedded are set in exactly one place -- the same
reasoning app/llm.py already states for the LLM client.

fastembed replaces sentence-transformers here (see docs/DEPLOYMENT-NOTES.md and
pyproject.toml) -- same model id, a fraction of the install footprint.
EMBED_MODEL defaults to BAAI/bge-small-en-v1.5, pinned in docs/CONTRACTS.md.

Documents and queries are embedded through two separate functions,
embed_documents() and embed_query(), even though -- as currently wired -- they
produce identical vectors for identical text. BGE models are generally trained
with an asymmetric convention: a passage is embedded as-is, but a query gets an
added instruction ("Represent this sentence for searching relevant passages:
...") that measurably improves retrieval against passages embedded without it.
For the larger BGE models, and for the older bge-small-en, that instruction
matters enough that fastembed applies it internally in query_embed(). It does
not for BAAI/bge-small-en-v1.5, the model actually pinned here: fastembed
0.8.1's own model registry describes prefixes as "not so necessary" for this
model (versus "necessary" for bge-small-en), and its query_embed() for this
model is implemented as a direct call to embed() -- confirmed by reading
fastembed's source and by test_embed.py's
test_query_and_document_embeddings_of_same_text_are_currently_identical.

The two functions are kept separate anyway. A query run through the wrong
method, on a model where the asymmetry IS real, degrades silently with no
error to catch it -- exactly the failure mode a model swap (a larger BGE
variant, a different embedding family) could reintroduce. Keeping
embed_documents()/embed_query() as distinct call sites means that if EMBED_MODEL
ever changes to a model that needs the instruction, only this module's
internals need to catch up -- no caller has to know or change, and no caller
can pick the wrong one. As long as bge-small-en-v1.5 is the pinned model,
though, this split is insurance against a future change, not a fix for a
present one.

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
