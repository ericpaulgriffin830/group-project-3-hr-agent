"""app/rag/chunk.py -- section -> chunk packing logic.

These are pure unit tests: no corpus, no network, no model. Sections and
documents are built directly as RawSection/ParsedDocument fixtures so each
test isolates one piece of the packing/id/snippet logic. The corpus-level
integration coverage (does the real corpus produce sane, unique chunks) lives
in test_ingest.py / test_retrieval.py instead.
"""

from __future__ import annotations

import pytest

from app.rag.chunk import (
    SNIPPET_CHARS,
    TARGET_CHUNK_TOKENS,
    _chunk_id,
    _pack_paragraphs,
    _paragraphs,
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
# _paragraphs
# ---------------------------------------------------------------------------


def test_paragraphs_splits_on_blank_lines():
    text = "First para.\n\nSecond para.\n\n\nThird para."
    assert _paragraphs(text) == ["First para.", "Second para.", "Third para."]


def test_paragraphs_strips_each_paragraph():
    text = "  leading and trailing  \n\n  spaces  "
    assert _paragraphs(text) == ["leading and trailing", "spaces"]


def test_paragraphs_single_block_with_no_blank_lines():
    text = "One paragraph\nacross two lines."
    assert _paragraphs(text) == ["One paragraph\nacross two lines."]


def test_paragraphs_ignores_empty_blocks():
    text = "First.\n\n\n\nSecond."
    assert _paragraphs(text) == ["First.", "Second."]


# ---------------------------------------------------------------------------
# _pack_paragraphs
# ---------------------------------------------------------------------------


def test_pack_paragraphs_single_group_when_under_budget():
    paras = ["Short one.", "Another short one."]
    groups = _pack_paragraphs(paras, target_tokens=600)
    assert len(groups) == 1
    assert groups[0] == "Short one.\n\nAnother short one."


def test_pack_paragraphs_splits_into_multiple_groups_over_budget():
    # each paragraph ~5 tokens; budget of 10 should force a new group every 2 paras
    paras = ["one two three four five"] * 4
    groups = _pack_paragraphs(paras, target_tokens=10)
    assert len(groups) > 1
    # no group should exceed budget once it already has content
    for group in groups:
        assert count_tokens(group) <= 10 or group.count("\n\n") == 0


def test_pack_paragraphs_never_splits_a_single_oversized_paragraph():
    huge = "word " * 1000
    paras = ["short para", huge.strip()]
    groups = _pack_paragraphs(paras, target_tokens=50)
    # the huge paragraph must appear whole in exactly one group
    assert any(huge.strip() in g for g in groups)


def test_pack_paragraphs_preserves_paragraph_order():
    paras = ["alpha", "beta", "gamma", "delta"]
    groups = _pack_paragraphs(paras, target_tokens=2)
    rejoined = "\n\n".join(groups)
    assert rejoined.split("\n\n") == paras


# ---------------------------------------------------------------------------
# chunk_section
# ---------------------------------------------------------------------------


def test_chunk_section_under_target_produces_single_chunk_with_exact_text():
    doc = _doc()
    section = _section(text="This section is short and well under the token target.")
    chunks = chunk_section(section, doc)
    assert len(chunks) == 1
    chunk = chunks[0]
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


def test_chunk_section_over_target_produces_multiple_chunks():
    doc = _doc()
    # Build paragraphs that sum well over TARGET_CHUNK_TOKENS, forcing a split.
    para = "word " * 200  # ~200 tokens each
    text = "\n\n".join([para.strip()] * 5)  # ~1000 tokens total
    section = _section(text=text)
    chunks = chunk_section(section, doc)
    assert len(chunks) > 1


def test_chunk_section_multi_chunk_has_increasing_index_and_unique_ids():
    doc = _doc()
    para = "word " * 200
    text = "\n\n".join([para.strip()] * 5)
    section = _section(text=text)
    chunks = chunk_section(section, doc)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_section_multi_chunk_all_share_section_id_and_citation():
    doc = _doc()
    para = "word " * 200
    text = "\n\n".join([para.strip()] * 5)
    section = _section(section_id="PTO-9", heading="Long Section", text=text)
    chunks = chunk_section(section, doc)
    assert all(c.section_id == "PTO-9" for c in chunks)
    assert all(c.section == "PTO-9 Long Section" for c in chunks)


def test_chunk_section_token_count_matches_count_tokens_of_chunk_text():
    doc = _doc()
    section = _section(text="Some words here, and a comma.")
    chunk = chunk_section(section, doc)[0]
    assert chunk.token_count == count_tokens(chunk.text)


# ---------------------------------------------------------------------------
# chunk_document / chunk_corpus
# ---------------------------------------------------------------------------


def test_chunk_document_concatenates_chunks_from_all_sections():
    doc = _doc(sections=[
        _section(section_id="A-1", heading="First", text="First section text."),
        _section(section_id="A-2", heading="Second", text="Second section text."),
    ])
    chunks = chunk_document(doc)
    assert [c.section_id for c in chunks] == ["A-1", "A-2"]


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


def test_chunk_corpus_over_real_corpus_produces_globally_unique_ids():
    from app.rag.ingest import load_corpus

    chunks = chunk_corpus(load_corpus())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) > 0
    assert len(ids) == len(set(ids))
