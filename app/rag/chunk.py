"""Turn each document's sections into embeddable chunks.

Chunking is heading-aware first: a chunk never crosses a section boundary. Two
reasons, both concrete rather than stylistic. First, `get_policy_section` (Contract
A, tool #2) already gives full-section access when a snippet is not enough, which
is what lets us skip overlap entirely -- see below. Second, the corpus's
cross-references ("see PTO-8", "per TAX-5") are prose anchored to whole sections;
splitting one mid-section risks separating a reference from what it refers to.

Most sections fit in one chunk untouched. A few run long enough (PTO-3, PTO-9,
RW-4, RW-6, and a handful of others) that a single chunk would be an unwieldy
retrieval unit, so those are split further -- by paragraph, never mid-sentence,
packed greedily up to TARGET_CHUNK_TOKENS.

No overlap between the resulting sub-chunks, unlike the sliding-window-with-overlap
example in the project brief. That pattern earns its overlap when chunk boundaries
fall in the middle of undifferentiated text and neighboring context might be lost.
Here, get_policy_section already covers "I need the neighboring context" by
returning the whole section -- so overlap would only duplicate content in the
index for no retrieval benefit. Every section-and-sub-chunk-count decision is
therefore driven by section structure, not a token window imposed on top of it.

Token counts use a small offline regex approximation (count_tokens below), not a
real tokenizer. This only has to be a consistent, deterministic sizing rule for
chunk-packing decisions -- not byte-exact to fastembed's own tokenizer, which
would mean loading the embedding model just to size chunks. tiktoken was tried
first and dropped: its cl100k_base encoding is not vendored in the package, so
get_encoding() fetches a ~1.7MB BPE file from openaipublic.blob.core.windows.net
on first use with no local fallback. That is a runtime network dependency this
project should not carry for something that only sizes chunks, particularly on
a free-tier deploy where DEPLOYMENT-NOTES.md already treats every unnecessary
runtime dependency as a real risk, not a hypothetical one.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.rag.ingest import ParsedDocument, RawSection

#: Sections at or under this many tokens become exactly one chunk.
TARGET_CHUNK_TOKENS = 600

#: Citation snippet length, matching mcp_server/fixtures.py's fixture-phase
#: truncation so a citation looks the same whether it is backed by the fixture
#: ranker or this real index.
SNIPPET_CHARS = 320

#: Word characters and punctuation counted separately, approximating subword
#: tokenization closely enough for a chunk-size budget -- no model, no network.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit.

    `section` is the citation string Contract A expects -- section_id and heading
    together, e.g. "PTO-3 Requesting and Approving PTO" -- formatted identically to
    mcp_server/fixtures.py's fixture-phase chunks, so a citation reads the same
    regardless of which ranking produced it.
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


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines. Never splits inside a paragraph or a list block."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pack_paragraphs(paragraphs: list[str], target_tokens: int) -> list[str]:
    """Greedily pack paragraphs into groups, each near or under target_tokens.

    Deterministic single left-to-right pass: a paragraph joins the current group
    unless that would push it over budget AND the group already has content --
    in which case it starts a new group instead. A single paragraph longer than
    the target still gets its own group rather than being cut mid-sentence; that
    is an accepted oversized chunk, not a bug, and it does not occur anywhere in
    the current corpus.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current and current_tokens + para_tokens > target_tokens:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += para_tokens
    if current:
        groups.append(current)
    return ["\n\n".join(g) for g in groups]


def chunk_section(section: RawSection, doc: ParsedDocument) -> list[Chunk]:
    if count_tokens(section.text) <= TARGET_CHUNK_TOKENS:
        pieces = [section.text]
    else:
        pieces = _pack_paragraphs(_paragraphs(section.text), TARGET_CHUNK_TOKENS)

    citation = f"{section.section_id} {section.heading}"
    return [
        Chunk(
            chunk_id=_chunk_id(doc.doc_id, section.section_id, i),
            doc_id=doc.doc_id,
            section_id=section.section_id,
            section=citation,
            title=doc.title,
            source_path=doc.source_path,
            text=piece,
            snippet=_snippet(piece),
            chunk_index=i,
            token_count=count_tokens(piece),
        )
        for i, piece in enumerate(pieces)
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
