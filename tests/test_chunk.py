"""app/rag/chunk.py -- section -> chunk logic.

These are pure unit tests: no corpus, no network, no model. Sections and
documents are built directly as RawSection/ParsedDocument fixtures so each
test isolates one piece of the id/snippet/chunking logic. The corpus-level
integration coverage (does the real corpus produce sane, unique chunks) lives
in test_ingest.py / test_retrieval.py instead.

As of 2026-09-24, chunking is one-chunk-per-section always -- a section is
never split, however long. See chunk.py's module docstring for why (Rob,
Chris and Eric agreed a chunk always carrying full section context beats a
tighter token budget). There is deliberately no test here for "a long section
gets split into multiple chunks" -- that behavior no longer exists.
"""

from __future__ import annotations

import pytest

from app.rag.chunk import (
    SNIPPET_CHARS,
    _chunk_id,
    _snippet,
    chunk_corpus,
    chunk_document,
    chunk_section,
    count_tokens,
)
from app.rag.ingest import ParsedDocument, RawSection


def _doc(doc_id="DOC-1", sections=None) -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        title="Sample Policy",
        filename="sample.md",
        format="markdown",
        version="1.0",
        effective_date="2026-01-01",
        owner="HR",
        applies_to="All employees",
        source_path="corpus/policies/sample.md",
        sections=sections or [],
    )


def _section(doc_id="DOC-1", section_id="DOC-1", heading="A Section", text="Some text.", order=0) -> RawSection:
    return RawSection(doc_id=doc_id, section_id=section_id, heading=heading, text=text, order=order)


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_counts_words_and_punctuation_separately():
    assert count_tokens("Hello, world!") == 4  # Hello , world !


def test_count_tokens_empty_string_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_is_deterministic():
    text = "Employees accrue PTO at a rate of 1.5 days per month."
    assert count_tokens(text) == count_tokens(text)


# ---------------------------------------------------------------------------
# _snippet
# ---------------------------------------------------------------------------


def test_snippet_leaves_short_text_unchanged():
    text = "Short paragraph, no truncation needed."
    assert _snippet(text) == text


def test_snippet_truncates_long_text_at_word_boundary_with_ellipsis():
    text = ("word " * 200).strip()
    snippet = _snippet(text, limit=50)
    assert snippet.endswith("…")
    assert len(snippet) <= 51
    # truncation happened at a space, not mid-word
    assert not snippet[:-1].endswith("wor")


def test_snippet_default_limit_matches_module_constant():
    text = "x" * (SNIPPET_CHARS + 50)
    snippet = _snippet(text)
    assert len(snippet) <= SNIPPET_CHARS + 1  # +1 for the ellipsis char


# ---------------------------------------------------------------------------
# _chunk_id
# ---------------------------------------------------------------------------


def test_chunk_id_is_deterministic_for_same_inputs():
    assert _chunk_id("PTO-HOLIDAYS", "PTO-3", 0) == _chunk_id("PTO-HOLIDAYS", "PTO-3", 0)


def test_chunk_id_differs_by_chunk_index():
    # chunk_section itself always passes 0 now -- this is still testing the
    # underlying function, since a future re-introduction of splitting should
    # not require _chunk_id to change.
    assert _chunk_id("PTO-HOLIDAYS", "PTO-3", 0) != _chunk_id("PTO-HOLIDAYS", "PTO-3", 1)


def test_chunk_id_differs_by_doc_or_section():
    base = _chunk_id("PTO-HOLIDAYS", "PTO-3", 0)
    assert base != _chunk_id("PTO-HOLIDAYS", "PTO-4", 0)
    assert base != _chunk_id("REMOTE-WORK", "PTO-3", 0)


def test_chunk_id_is_16_hex_characters():
    cid = _chunk_id("DOC", "SEC-1", 0)
    assert len(cid) == 16
    int(cid, 16)  # raises if not valid hex


# ---------------------------------------------------------------------------
# chunk_section -- one chunk per section, always, whatever the length
# ---------------------------------------------------------------------------


def test_chunk_section_produces_exactly_one_chunk():
    doc = _doc()
    section = _section(text="This section is short and unremarkable.")
    chunks = chunk_section(section, doc)
    assert len(chunks) == 1


def test_chunk_section_chunk_carries_the_full_section_text_unchanged():
    doc = _doc()
    section = _section(text="This section is short and unremarkable.")
    chunk = chunk_section(section, doc)[0]
    assert chunk.text == section.text
    assert chunk.chunk_index == 0
    assert chunk.doc_id == doc.doc_id
    assert chunk.section_id == section.section_id
    assert chunk.title == doc.title
    assert chunk.source_path == doc.source_path


def test_chunk_section_citation_combines_section_id_and_heading():
    doc = _doc()
    section = _section(section_id="PTO-3", heading="Requesting and Approving PTO", text="Text.")
    chunk = chunk_section(section, doc)[0]
    assert chunk.section == "PTO-3 Requesting and Approving PTO"


def test_chunk_section_a_very_long_section_is_still_exactly_one_chunk():
    doc = _doc()
    # ~1000 tokens -- well past what the old TARGET_CHUNK_TOKENS budget (600)
    # would have split. The whole point of the current design is that this no
    # longer matters.
    para = "word " * 200
    text = "\n\n".join([para.strip()] * 5)
    section = _section(text=text)
    chunks = chunk_section(section, doc)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].token_count == count_tokens(text)


def test_chunk_section_long_section_keeps_its_own_section_id_and_citation():
    doc = _doc()
    para = "word " * 200
    text = "\n\n".join([para.strip()] * 5)
    section = _section(section_id="PTO-9", heading="Long Section", text=text)
    chunk = chunk_section(section, doc)[0]
    assert chunk.section_id == "PTO-9"
    assert chunk.section == "PTO-9 Long Section"


def test_chunk_section_token_count_matches_count_tokens_of_chunk_text():
    doc = _doc()
    section = _section(text="Some words here, and a comma.")
    chunk = chunk_section(section, doc)[0]
    assert chunk.token_count == count_tokens(chunk.text)


def test_chunk_section_snippet_is_truncated_even_when_chunk_text_is_not():
    doc = _doc()
    long_text = ("word " * 200).strip()
    section = _section(text=long_text)
    chunk = chunk_section(section, doc)[0]
    # the chunk's full text is never truncated -- only its citation snippet is.
    assert chunk.text == long_text
    assert len(chunk.snippet) < len(chunk.text)


# ---------------------------------------------------------------------------
# chunk_document / chunk_corpus
# ---------------------------------------------------------------------------


def test_chunk_document_produces_exactly_one_chunk_per_section():
    doc = _doc(sections=[
        _section(section_id="A-1", heading="First", text="First section text."),
        _section(section_id="A-2", heading="Second", text="Second section text."),
    ])
    chunks = chunk_document(doc)
    assert [c.section_id for c in chunks] == ["A-1", "A-2"]
    assert len(chunks) == len(doc.sections)


def test_chunk_corpus_chunk_ids_are_globally_unique():
    doc1 = _doc(doc_id="DOC-1", sections=[
        _section(doc_id="DOC-1", section_id="S-1", text="Text one."),
        _section(doc_id="DOC-1", section_id="S-2", text="Text two."),
    ])
    doc2 = _doc(doc_id="DOC-2", sections=[
        _section(doc_id="DOC-2", section_id="S-1", text="Different text one."),
    ])
    chunks = chunk_corpus([doc1, doc2])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_corpus_over_real_corpus_produces_exactly_one_chunk_per_section():
    from app.rag.ingest import load_corpus

    documents = load_corpus()
    chunks = chunk_corpus(documents)
    total_sections = sum(len(doc.sections) for doc in documents)
    assert len(chunks) == total_sections


def test_chunk_corpus_over_real_corpus_produces_globally_unique_ids():
    from app.rag.ingest import load_corpus

    chunks = chunk_corpus(load_corpus())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) > 0
    assert len(ids) == len(set(ids))


def test_chunk_corpus_over_real_corpus_every_chunk_matches_its_source_section_text():
    from app.rag.ingest import load_corpus

    documents = load_corpus()
    by_key = {(doc.doc_id, s.section_id): s.text for doc in documents for s in doc.sections}
    for chunk in chunk_corpus(documents):
        assert chunk.text == by_key[(chunk.doc_id, chunk.section_id)]
