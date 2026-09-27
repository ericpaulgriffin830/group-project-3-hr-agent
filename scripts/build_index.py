"""Build the Chroma vector index from corpus/policies/*.

`app/rag/store.py`'s own docstring already names this file as the thing that
calls `reset_collection()` + `upsert_chunks()` -- it just hadn't been written
yet. Its absence is exactly why `/health` reports `index_ready: false` even
locally: nothing in the codebase ever populated CHROMA_PATH.

Run it:
    uv run python -m scripts.build_index

`data/chroma/` (CHROMA_PATH's default) is gitignored on purpose -- an index is
a build artifact, not source, and it would drift from the corpus the moment
either changed independently. That means it has to be (re)built:
  - once, locally, after cloning -- `/health` and the demo tasks need a
    populated index to retrieve anything.
  - on every Render deploy -- the free tier has no persistent disk, so
    whatever CHROMA_PATH held before a deploy does not carry forward into the
    next one. render.yaml's buildCommand runs this after `uv sync`.

`reset_collection()` first, not a bare upsert: upsert only touches ids that
still exist in the current chunk set, so a section renamed or removed in the
corpus would otherwise leave its old vector behind forever (store.py's own
docstring flags this). A full rebuild is cheap enough here (106 chunks) that
there is no reason to reach for anything smarter.
"""

from __future__ import annotations

import sys
import time

from app.rag.chunk import chunk_corpus
from app.rag.ingest import IngestError, load_corpus
from app.rag.store import count, reset_collection, upsert_chunks


def build() -> int:
    started = time.monotonic()
    print("Loading and parsing corpus/policies/* against corpus/manifest.json...")
    try:
        documents = load_corpus()
    except IngestError as exc:
        print(f"FAILED to parse corpus: {exc}", file=sys.stderr)
        raise

    chunks = chunk_corpus(documents)
    print(f"Parsed {len(documents)} document(s) into {len(chunks)} chunk(s).")

    print("Resetting the Chroma collection and re-embedding everything...")
    reset_collection()
    written = upsert_chunks(chunks)

    elapsed = time.monotonic() - started
    total = count()
    print(f"Wrote {written} chunk(s); collection now holds {total}. ({elapsed:.1f}s)")

    if total != len(chunks):
        print(
            f"WARNING: collection count ({total}) does not match chunks parsed "
            f"({len(chunks)}) -- investigate before relying on this index.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
