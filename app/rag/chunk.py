"""Turn each document's sections into embeddable chunks.

**One section = one chunk, always.** Chunking is heading-aware: a chunk never
crosses a section boundary, and a section is never split into more than one
chunk regardless of length. This changed from an earlier design that packed
long sections into multiple sub-chunks (see git history for chunk_section's
previous paragraph-packing logic) -- Rob, Chris and Eric agreed on 2026-09-24
that a single chunk always carrying the FULL section text is worth more than a
tighter per-chunk token budget. The concrete failure case the earlier design
risked: a query matches only the first half of a two-chunk section, the second
chunk (with the actually-relevant sentence) never surfaces in the top-k, and
`get_policy_section` is never called to pull in the rest because nothing
pointed the agent at that section id in the first place. One chunk per section
means whatever chunk a query hits already contains everything `get_policy_section`
would have returned, with no second call required.

This also simplifies the reasoning `retrieve.py`'s module docstring already gives
for skipping overlap between sub-chunks -- there are no sub-chunks to overlap
between anymore. That reasoning still applies to why a chunk never crosses a
section boundary in the first place: the corpus's cross-references ("see PTO-8",
"per TAX-5") are prose anchored to whole sections, and splitting one mid-section
would risk separating a reference from what it refers to. There is now nothing
left to split.

**Known trade-off, not addressed here.** fastembed's own model registry documents
`BAAI/bge-small-en-v1.5` as truncating input at 512 tokens. The longest section in
the current corpus is 460 tokens by this module's own (approximate, word-level)
counter -- under that limit, but not by a wide margin, and a real subword
tokenizer typically produces MORE tokens than a word-level count for the same
text, not fewer. A future section longer than the real embedding truncation point
would have its tail silently dropped from the vector representation (though not
from `text`/`snippet`, and not from what `get_policy_section` returns -- only the
embedding itself would be incomplete). `token_count` is still recorded on every
Chunk specifically so this is checkable later, but nothing here enforces a limit
or warns at build time. Worth a line in the design doc's known-limitations section
if the corpus grows.

Token counts use a small offline regex approximation (count_tokens below), not a
real tokenizer -- see the trade-off above for where that approximation's direction
of error matters. tiktoken was tried first and dropped: its cl100k_base encoding
is not vendored in the package, so get_encoding() fetches a ~1.7MB BPE file from
openaipublic.blob.core.windows.net on first use with no local fallback. That is a
runtime network dependency this project should not carry for something that only
sizes and records chunk length, particularly on a free-tier deploy where
DEPLOYMENT-NOTES.md already treats every unnecessary runtime dependency as a real
risk, not a hypothetical one.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.rag.ingest import ParsedDocument, RawSection

#: Citation snippet length, matching mcp_server/fixtures.py's fixture-phase
#: truncation so a citation looks the same whether it is backed by the fixture
#: ranker or this real index.
SNIPPET_CHARS = 320

#: Word characters and punctuation counted separately, approximating subword
#: tokenization closely enough to record chunk length -- no model, no network.
#: See the module docstring's trade-off note on why this can undercount.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit -- always exactly one whole section's worth of text.

    `section` is the citation string Contract A expects -- section_id and heading
    together, e.g. "PTO-3 Requesting and Approving PTO" -- formatted identically to
    mcp_server/fixtures.py's fixture-phase chunks, so a citation reads the same
    regardless of which ranking produced it.

    `chunk_index` is always 0 now that a section never splits. The field stays
    (rather than being removed) so `store.py`'s metadata shape and `_chunk_id`'s
    signature don't need to change alongside this -- a section splitting again in
    the future would only mean this stops always being 0, not a schema change.
    """

    chunk_id: str
    doc_id: str
    section_id: str
    section: str
    title: str
    source_path: str
    text: str
    snippet: str
    chunk_index: int
    token_count: int


def _chunk_id(doc_id: str, section_id: str, chunk_index: int) -> str:
    """Deterministic id from position, not Python's per-process-salted hash()."""
    raw = f"{doc_id}|{section_id}|{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Truncate at a word boundary near `limit` rather than mid-word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip() + "…"


def chunk_section(section: RawSection, doc: ParsedDocument) -> list[Chunk]:
    """One section -> exactly one chunk, containing the section's full text.

    Returns a single-element list (not a bare Chunk) so chunk_document's
    `chunks.extend(...)` and any existing caller iterating the result keep
    working unchanged -- the one-chunk-per-section invariant lives in the
    contents, not the shape of what this returns.
    """
    citation = f"{section.section_id} {section.heading}"
    text = section.text
    return [
        Chunk(
            chunk_id=_chunk_id(doc.doc_id, section.section_id, 0),
            doc_id=doc.doc_id,
            section_id=section.section_id,
            section=citation,
            title=doc.title,
            source_path=doc.source_path,
            text=text,
            snippet=_snippet(text),
            chunk_index=0,
            token_count=count_tokens(text),
        )
    ]


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        chunks.extend(chunk_section(section, doc))
    return chunks


def chunk_corpus(documents: list[ParsedDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc))
    return chunks
