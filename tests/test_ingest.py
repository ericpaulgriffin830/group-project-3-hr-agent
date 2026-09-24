"""app/rag/ingest.py -- format-specific corpus parsing.

These tests run against the real corpus/ directory, not a fixture corpus,
because the whole point of ingest.py is correctly handling the three real
source formats (markdown, html, txt) as they actually exist on disk. A
synthetic mini-corpus would risk testing a simplified version of the parsing
problem rather than the real one.
"""

from __future__ import annotations

import copy

import pytest

from app.rag.ingest import IngestError, load_corpus, load_manifest, parse_document


@pytest.fixture(scope="module")
def documents():
    return load_corpus()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_all_twelve_documents_parse(documents):
    assert len(documents) == 12


def test_every_document_section_list_matches_manifest_exactly(documents, manifest):
    """Redundant with parse_document's own internal check, on purpose -- this
    is the explicit, readable version of the invariant load_corpus() already
    enforces by raising IngestError, so a future change to that internal
    check still has a test independent of it."""
    by_id = {d["doc_id"]: d for d in manifest["documents"]}
    for doc in documents:
        expected = [s["section_id"] for s in by_id[doc.doc_id]["sections"]]
        got = [s.section_id for s in doc.sections]
        assert got == expected, doc.doc_id


def test_markdown_parsing_preserves_inline_emphasis():
    """PTO-2 lists accrual rates with **bold** employment-type labels in the
    source markdown -- this should survive ingestion unchanged, since answer.py
    puts chunk text directly into the LLM's prompt."""
    doc = next(d for d in load_corpus() if d.doc_id == "PTO-HOLIDAYS")
    pto2 = next(s for s in doc.sections if s.section_id == "PTO-2")
    assert "**Full-time exempt employees**" in pto2.text


def test_html_parsing_converts_strong_tags_to_markdown_emphasis():
    """RW-4's source HTML wraps this phrase in <strong>; markdownify should
    turn that into **...** rather than dropping the emphasis or leaking the
    raw tag into chunk text."""
    doc = next(d for d in load_corpus() if d.doc_id == "REMOTE-WORK")
    rw4 = next(s for s in doc.sections if s.section_id == "RW-4")
    assert "**International Remote Work Exception**" in rw4.text
    assert "<strong>" not in rw4.text and "</strong>" not in rw4.text


def test_txt_parsing_extracts_all_leave_sections():
    doc = next(d for d in load_corpus() if d.doc_id == "LEAVE")
    ids = [s.section_id for s in doc.sections]
    assert ids == [f"LEAVE-{i}" for i in range(1, 10)]
    leave2 = next(s for s in doc.sections if s.section_id == "LEAVE-2")
    assert leave2.heading == "Family and Medical Leave (FMLA-aligned)"
    assert "FMLA" in leave2.text


def test_source_path_is_repo_relative_not_absolute():
    doc = next(d for d in load_corpus() if d.doc_id == "PTO-HOLIDAYS")
    assert doc.source_path == "corpus/policies/02-pto-and-holidays-policy.md"


def test_metadata_comes_from_manifest_not_document_body():
    """corpus/README.md is explicit that the manifest is authoritative --
    this checks a value that would differ if ingest.py ever started scraping
    it from frontmatter/meta-tags/header-blocks instead."""
    doc = next(d for d in load_corpus() if d.doc_id == "REMOTE-WORK")
    assert doc.title == "Work Arrangement Categories"  # manifest title, not
                                                        # the HTML <title> tag


def test_extra_manifest_section_raises_ingest_error(manifest):
    """A document whose manifest promises a section the file doesn't have
    must fail loudly at parse time, not produce a citation nothing backs."""
    broken = copy.deepcopy(next(
        d for d in manifest["documents"] if d["doc_id"] == "PTO-HOLIDAYS"
    ))
    broken["sections"].append({"section_id": "PTO-99", "heading": "Does Not Exist"})
    with pytest.raises(IngestError, match="do not match"):
        parse_document(broken)


def test_missing_manifest_section_raises_ingest_error(manifest):
    """The reverse case: the file has a section the manifest doesn't list."""
    broken = copy.deepcopy(next(
        d for d in manifest["documents"] if d["doc_id"] == "PTO-HOLIDAYS"
    ))
    broken["sections"] = broken["sections"][:-1]  # drop PTO-10
    with pytest.raises(IngestError, match="do not match"):
        parse_document(broken)


def test_unsupported_format_raises_ingest_error(manifest):
    broken = copy.deepcopy(next(
        d for d in manifest["documents"] if d["doc_id"] == "PTO-HOLIDAYS"
    ))
    broken["format"] = "pdf"
    with pytest.raises(IngestError, match="unsupported corpus format"):
        parse_document(broken)
