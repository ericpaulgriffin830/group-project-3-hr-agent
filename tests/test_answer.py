"""app/rag/answer.py -- Contract C's synthesize() seam.

No real LLM or corpus dependency anywhere in this file: `complete()` (imported
into answer.py's own namespace from app.llm) and `get_section()` (imported from
app.rag.retrieve) are monkeypatched at every call site, so these tests are fast,
deterministic, and exercise the actual grounding/parsing/scoring logic rather
than anything Groq or the real corpus would return on a given day.

Three tiers, matching the module's own structure:
  - pure-function unit tests for each private helper (_as_dict, _enrich,
    _parse_completion, _basis, _confidence, ...)
  - synthesize()-level tests with `complete` mocked, covering the branches that
    only make sense end-to-end (grounding a hallucinated citation out, the
    MAX_PROMPT_CHUNKS window, mode="refusal", no-evidence refusal)
  - the LLM-failure fallback, since app.llm.py's own stated rule ("never ship a
    fabricated answer on failure") is a real behavior this module has to uphold
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import app.rag.answer as answer_module
from app.llm import LLMError, LLMNotConfigured
from app.rag.answer import (
    MAX_PROMPT_CHUNKS,
    NEUTRAL_SCORE,
    _as_dict,
    _basis,
    _build_prompt,
    _clean_answer,
    _confidence,
    _enrich,
    _format_evidence,
    _format_tool_results,
    _llm_unavailable_fallback,
    _parse_completion,
    _to_citation,
    _truncate,
    synthesize,
)
from app.rag.chunk import Chunk


def _completion(text: str):
    """A stand-in for app.llm.Completion -- only .text is ever read."""
    class _Fake:
        pass
    fake = _Fake()
    fake.text = text
    return fake


def _rich_chunk(doc_id="PTO-HOLIDAYS", section="PTO-2 Accrual Rates by Employment Type",
                title="PTO and Holidays Policy", snippet="Accrues at 1.5 days/month.",
                score=0.8) -> dict:
    return {"doc_id": doc_id, "title": title, "section": section, "snippet": snippet, "score": score}


def _sparse_ref(doc_id="PTO-HOLIDAYS", section="PTO-3") -> dict:
    """check_policy_compliance's policy_refs shape -- doc_id and section only."""
    return {"doc_id": doc_id, "section": section}


def _tool_result(tool="check_pto_balance", ok=True, payload=None) -> dict:
    return {"tool": tool, "ok": ok, "payload": payload or {}, "error": None}


META_OK = '''The answer text goes here.

META:
{"cited_doc_ids": ["PTO-HOLIDAYS"], "unsupported_flags": []}'''


# ---------------------------------------------------------------------------
# _as_dict
# ---------------------------------------------------------------------------


def test_as_dict_passes_through_a_plain_dict():
    d = {"doc_id": "X"}
    assert _as_dict(d) is d


def test_as_dict_converts_a_chunk_dataclass_instance():
    chunk = Chunk(chunk_id="abc123", doc_id="PTO-HOLIDAYS", section_id="PTO-3",
                 section="PTO-3 Requesting and Approving PTO", title="PTO and Holidays Policy",
                 source_path="corpus/policies/pto.md", text="Full text.", snippet="Full text.",
                 chunk_index=0, token_count=2)
    d = _as_dict(chunk)
    assert d["doc_id"] == "PTO-HOLIDAYS"
    assert d["section"] == "PTO-3 Requesting and Approving PTO"
    assert d["snippet"] == "Full text."


def test_as_dict_falls_back_to_dunder_dict_for_arbitrary_objects():
    class Thing:
        def __init__(self):
            self.doc_id = "X"
    assert _as_dict(Thing()) == {"doc_id": "X"}


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


def test_truncate_leaves_short_text_unchanged():
    assert _truncate("short text") == "short text"


def test_truncate_cuts_long_text_at_a_word_boundary_with_ellipsis():
    text = ("word " * 200).strip()
    out = _truncate(text, limit=50)
    assert out.endswith("…")
    assert len(out) <= 51


# ---------------------------------------------------------------------------
# _enrich
# ---------------------------------------------------------------------------


def test_enrich_passes_through_an_already_rich_chunk_untouched(monkeypatch):
    called = []
    monkeypatch.setattr(answer_module, "get_section", lambda *a: called.append(a) or {})
    chunk = _rich_chunk()
    assert _enrich(chunk) is chunk
    assert called == []  # never looked up -- title/snippet already present


def test_enrich_resolves_a_sparse_chunk_via_get_section(monkeypatch):
    monkeypatch.setattr(answer_module, "get_section", lambda doc_id, section_id: {
        "doc_id": doc_id, "title": "PTO and Holidays Policy",
        "section": f"{section_id} Requesting and Approving PTO",
        "text": "Full section text goes here.",
    })
    result = _enrich(_sparse_ref(doc_id="PTO-HOLIDAYS", section="PTO-3"))
    assert result["doc_id"] == "PTO-HOLIDAYS"
    assert result["title"] == "PTO and Holidays Policy"
    assert result["snippet"] == "Full section text goes here."


def test_enrich_returns_none_when_get_section_reports_an_error(monkeypatch):
    monkeypatch.setattr(answer_module, "get_section", lambda *a: {
        "error": "section_not_found", "message": "nope"})
    assert _enrich(_sparse_ref(doc_id="pto-and-leave", section="1.3")) is None


def test_enrich_returns_none_when_doc_id_or_section_missing():
    assert _enrich({"section": "PTO-3"}) is None
    assert _enrich({"doc_id": "PTO-HOLIDAYS"}) is None


def test_enrich_prefers_section_id_over_section_when_both_present(monkeypatch):
    seen = {}
    def fake_get_section(doc_id, section_id):
        seen["section_id"] = section_id
        return {"doc_id": doc_id, "title": "T", "section": "S", "text": "text"}
    monkeypatch.setattr(answer_module, "get_section", fake_get_section)
    _enrich({"doc_id": "PTO-HOLIDAYS", "section_id": "PTO-3", "section": "not this"})
    assert seen["section_id"] == "PTO-3"


# ---------------------------------------------------------------------------
# _to_citation / _format_evidence / _format_tool_results / _build_prompt
# ---------------------------------------------------------------------------


def test_to_citation_strips_to_contract_c_shape():
    entry = _rich_chunk()
    citation = _to_citation(entry)
    assert set(citation.keys()) == {"doc_id", "title", "section", "snippet"}
    assert "score" not in citation


def test_format_evidence_empty_list_uses_placeholder():
    assert "no policy passages" in _format_evidence([])


def test_format_evidence_renders_each_chunk():
    text = _format_evidence([_rich_chunk(doc_id="PTO-HOLIDAYS"), _rich_chunk(doc_id="LEAVE")])
    assert "PTO-HOLIDAYS" in text and "LEAVE" in text


def test_format_tool_results_empty_list_uses_placeholder():
    assert "no employee-specific data" in _format_tool_results([])


def test_format_tool_results_drops_error_and_message_keys():
    result = [_tool_result(payload={"employee_id": "E1008", "error": "x", "message": "y"})]
    text = _format_tool_results(result)
    assert "employee_id" in text
    assert '"error"' not in text and '"message"' not in text


def test_build_prompt_workflow_mode_includes_employee_data_section():
    prompt = _build_prompt("q", "workflow", "EVIDENCE", "TOOLDATA")
    assert "Employee-specific data:\nTOOLDATA" in prompt
    assert "EVIDENCE" in prompt


def test_build_prompt_policy_mode_omits_employee_data_label():
    prompt = _build_prompt("q", "policy", "EVIDENCE", "TOOLDATA")
    assert "Employee-specific data" not in prompt


# ---------------------------------------------------------------------------
# _clean_answer / _parse_completion
# ---------------------------------------------------------------------------


def test_clean_answer_strips_answer_label_case_insensitively():
    assert _clean_answer("ANSWER: Hello there.") == "Hello there."
    assert _clean_answer("answer:   Hello there.") == "Hello there."


def test_clean_answer_leaves_unlabeled_text_untouched():
    assert _clean_answer("  Hello there.  ") == "Hello there."


def test_parse_completion_well_formed_meta_block():
    answer, cited, flags = _parse_completion(META_OK, fallback_ids=["FALLBACK"])
    assert answer == "The answer text goes here."
    assert cited == ["PTO-HOLIDAYS"]
    assert flags == []


def test_parse_completion_missing_meta_block_falls_back():
    answer, cited, flags = _parse_completion("Just an answer, no trailer.", fallback_ids=["A", "B"])
    assert answer == "Just an answer, no trailer."
    assert cited == ["A", "B"]
    assert len(flags) == 1 and "META block" in flags[0]


def test_parse_completion_malformed_json_falls_back():
    raw = "Answer text.\n\nMETA:\n{not valid json}"
    answer, cited, flags = _parse_completion(raw, fallback_ids=["A"])
    assert cited == ["A"]
    assert any("not valid JSON" in f for f in flags)


def test_parse_completion_missing_cited_doc_ids_key_falls_back():
    raw = 'Answer.\n\nMETA:\n{"unsupported_flags": ["something"]}'
    answer, cited, flags = _parse_completion(raw, fallback_ids=["A"])
    assert cited == ["A"]
    assert "something" in flags  # original flag preserved
    assert any("cited_doc_ids missing" in f for f in flags)


def test_parse_completion_non_string_cited_doc_ids_falls_back():
    raw = 'Answer.\n\nMETA:\n{"cited_doc_ids": [1, 2], "unsupported_flags": []}'
    _, cited, flags = _parse_completion(raw, fallback_ids=["A"])
    assert cited == ["A"]
    assert any("cited_doc_ids missing" in f for f in flags)


# ---------------------------------------------------------------------------
# _basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("citations,tools,mode,expected", [
    ([{"doc_id": "X"}], [{"ok": True}], "policy", "both"),
    ([{"doc_id": "X"}], [], "policy", "policy_rag"),
    ([], [{"ok": True}], "workflow", "tool_data"),
    ([], [], "policy", "refused"),
    ([{"doc_id": "X"}], [{"ok": True}], "refusal", "refused"),
])
def test_basis_matrix(citations, tools, mode, expected):
    assert _basis(citations, tools, mode) == expected


# ---------------------------------------------------------------------------
# _confidence
# ---------------------------------------------------------------------------


def test_confidence_is_zero_when_refused():
    assert _confidence([], [], "refused", []) == 0.0


def test_confidence_tool_data_baseline_with_no_flags():
    assert _confidence([], [{"ok": True}], "tool_data", []) == 0.75


def test_confidence_policy_rag_averages_scores():
    resolved = [{"score": 0.8}, {"score": 0.6}]
    assert _confidence(resolved, [], "policy_rag", []) == pytest.approx(0.70)


def test_confidence_policy_rag_uses_neutral_score_when_missing():
    assert _confidence([{"doc_id": "X"}], [], "policy_rag", []) == NEUTRAL_SCORE


def test_confidence_clips_scores_above_one():
    assert _confidence([{"score": 1.5}], [], "policy_rag", []) == 1.0


def test_confidence_both_averages_with_tool_baseline():
    resolved = [{"score": 0.9}]
    # base = 0.9, then (0.9 + 0.75) / 2 = 0.825, rounded to 0.82
    assert _confidence(resolved, [{"ok": True}], "both", []) == pytest.approx(0.82, abs=0.005)


def test_confidence_penalized_when_unsupported_flags_present():
    base = _confidence([{"score": 0.8}], [], "policy_rag", [])
    penalized = _confidence([{"score": 0.8}], [], "policy_rag", ["something is unsupported"])
    assert penalized == pytest.approx(base * 0.6, abs=0.01)


# ---------------------------------------------------------------------------
# _llm_unavailable_fallback
# ---------------------------------------------------------------------------


def test_llm_unavailable_fallback_confidence_is_always_zero():
    out = _llm_unavailable_fallback([_rich_chunk()], [{"ok": True}], [], "workflow")
    assert out["confidence"] == 0.0


def test_llm_unavailable_fallback_carries_real_citations_and_a_flag():
    out = _llm_unavailable_fallback([_rich_chunk(doc_id="PTO-HOLIDAYS")], [], [], "policy")
    assert out["citations"][0]["doc_id"] == "PTO-HOLIDAYS"
    assert any("llm_unavailable" in f for f in out["unsupported_flags"])


# ---------------------------------------------------------------------------
# synthesize() -- mode="refusal" and no-evidence short circuits
# ---------------------------------------------------------------------------


def test_synthesize_refusal_mode_never_calls_the_model():
    with patch.object(answer_module, "complete") as mock_complete:
        out = synthesize("anything", mode="refusal")
        mock_complete.assert_not_called()
    assert out["basis"] == "refused"
    assert out["citations"] == []
    assert out["confidence"] == 0.0


def test_synthesize_no_evidence_and_no_tool_results_refuses_without_calling_model():
    with patch.object(answer_module, "complete") as mock_complete:
        out = synthesize("anything", chunks=[], tool_results=[], mode="policy")
        mock_complete.assert_not_called()
    assert out["basis"] == "refused"
    assert "no_evidence" in out["unsupported_flags"]


def test_synthesize_all_sparse_refs_unresolved_refuses_with_flag(monkeypatch):
    monkeypatch.setattr(answer_module, "get_section", lambda *a: {"error": "section_not_found"})
    out = synthesize("q", chunks=[_sparse_ref(doc_id="pto-and-leave", section="1.3")],
                     tool_results=[], mode="policy")
    assert out["basis"] == "refused"
    assert any("could not be resolved" in f for f in out["unsupported_flags"])


# ---------------------------------------------------------------------------
# synthesize() -- happy paths, with `complete` mocked
# ---------------------------------------------------------------------------


def test_synthesize_happy_path_policy_mode():
    with patch.object(answer_module, "complete", return_value=_completion(META_OK)):
        out = synthesize("How much PTO do I accrue?", chunks=[_rich_chunk()],
                         tool_results=[], mode="policy")
    assert out["answer"] == "The answer text goes here."
    assert out["citations"] == [{
        "doc_id": "PTO-HOLIDAYS", "title": "PTO and Holidays Policy",
        "section": "PTO-2 Accrual Rates by Employment Type", "snippet": "Accrues at 1.5 days/month.",
    }]
    assert out["basis"] == "policy_rag"
    assert out["unsupported_flags"] == []
    assert out["confidence"] == pytest.approx(0.8)


def test_synthesize_workflow_mode_with_tool_results_is_basis_both():
    completion = _completion(
        'Answer.\n\nMETA:\n{"cited_doc_ids": ["PTO-HOLIDAYS"], "unsupported_flags": []}')
    with patch.object(answer_module, "complete", return_value=completion):
        out = synthesize("Can I take 3 days?", chunks=[_rich_chunk()],
                         tool_results=[_tool_result(payload={"available_days": 1.5})],
                         mode="workflow")
    assert out["basis"] == "both"


def test_synthesize_workflow_mode_without_tool_results_flags_it():
    completion = _completion(
        'Answer.\n\nMETA:\n{"cited_doc_ids": ["PTO-HOLIDAYS"], "unsupported_flags": []}')
    with patch.object(answer_module, "complete", return_value=completion):
        out = synthesize("Can I take 3 days?", chunks=[_rich_chunk()],
                         tool_results=[], mode="workflow")
    assert any("no employee-specific data" in f for f in out["unsupported_flags"])


def test_synthesize_drops_a_hallucinated_citation_not_in_the_evidence():
    completion = _completion(
        'Answer.\n\nMETA:\n{"cited_doc_ids": ["NOT-A-REAL-DOC"], "unsupported_flags": []}')
    with patch.object(answer_module, "complete", return_value=completion):
        out = synthesize("q", chunks=[_rich_chunk(doc_id="PTO-HOLIDAYS")], mode="policy")
    # the hallucinated id resolves to nothing; falls back to top evidence instead
    # of returning zero citations for a question that DID have real evidence.
    assert out["citations"][0]["doc_id"] == "PTO-HOLIDAYS"
    assert any("did not name a resolvable citation" in f for f in out["unsupported_flags"])


def test_synthesize_ignores_evidence_beyond_max_prompt_chunks_window():
    chunks = [_rich_chunk(doc_id=f"DOC-{i}", section=f"S-{i}", score=0.5) for i in range(MAX_PROMPT_CHUNKS + 3)]
    # the model "cites" a doc_id that exists in the full chunk list but falls
    # outside the window actually shown to it -- it must not resolve.
    out_of_window_id = chunks[-1]["doc_id"]
    completion = _completion(
        f'Answer.\n\nMETA:\n{{"cited_doc_ids": ["{out_of_window_id}"], "unsupported_flags": []}}')
    with patch.object(answer_module, "complete", return_value=completion):
        out = synthesize("q", chunks=chunks, mode="policy")
    cited = {c["doc_id"] for c in out["citations"]}
    assert out_of_window_id not in cited


def test_synthesize_deduplicates_repeated_doc_and_section():
    completion = _completion(META_OK)
    duplicate = [_rich_chunk(), _rich_chunk()]  # identical doc_id + section
    with patch.object(answer_module, "complete", return_value=completion) as mock_complete:
        out = synthesize("q", chunks=duplicate, mode="policy")
    assert len(out["citations"]) == 1
    prompt_arg = mock_complete.call_args[0][0]
    assert prompt_arg.count("PTO-HOLIDAYS") == 1


def test_synthesize_accepts_real_chunk_dataclass_instances():
    chunk = Chunk(chunk_id="abc", doc_id="LEAVE", section_id="LEAVE-3",
                 section="LEAVE-3 Parental Leave", title="Types of Leave",
                 source_path="corpus/policies/leave.txt", text="Full text.",
                 snippet="Full text.", chunk_index=0, token_count=2)
    completion = _completion(
        'Answer.\n\nMETA:\n{"cited_doc_ids": ["LEAVE"], "unsupported_flags": []}')
    with patch.object(answer_module, "complete", return_value=completion):
        out = synthesize("q", chunks=[chunk], mode="policy")
    assert out["citations"][0]["doc_id"] == "LEAVE"


# ---------------------------------------------------------------------------
# synthesize() -- LLM failure fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [LLMNotConfigured("no key"), LLMError("exhausted")])
def test_synthesize_falls_back_gracefully_when_llm_unavailable(exc):
    with patch.object(answer_module, "complete", side_effect=exc):
        out = synthesize("q", chunks=[_rich_chunk(doc_id="PTO-HOLIDAYS")], mode="policy")
    assert out["confidence"] == 0.0
    assert out["citations"][0]["doc_id"] == "PTO-HOLIDAYS"
    assert any("llm_unavailable" in f for f in out["unsupported_flags"])


def test_synthesize_never_raises_when_llm_fails_with_only_tool_results():
    with patch.object(answer_module, "complete", side_effect=LLMError("exhausted")):
        out = synthesize("q", chunks=[], tool_results=[_tool_result()], mode="workflow")
    assert out["basis"] == "tool_data"
    assert out["confidence"] == 0.0
