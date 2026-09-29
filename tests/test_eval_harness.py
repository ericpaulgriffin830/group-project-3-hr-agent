"""The parts of the evaluation harness that produce reported numbers.

The harnesses are scripts, and most of what they do is only meaningful against a
live model. These are the pieces that are pure functions and that a graded table
in `design-and-evaluation.md` rests on -- a percentile off by one row, or a
scorer counting an un-asserted expectation as a pass, is a wrong number in a
submitted document with nothing to catch it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str):
    """Import a script from evaluation/ by path.

    `evaluation/` is not a package -- the scripts put the repo root on sys.path
    themselves and are run directly. Importing by path keeps it that way rather
    than adding an __init__.py to make the test convenient.
    """
    spec = importlib.util.spec_from_file_location(name, _ROOT / "evaluation" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load("run_eval")


# --------------------------------------------------------------- percentiles

@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [
        # Nearest-rank, 1-indexed: p50 of ten values is the 5th.
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50, 5),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 10),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100, 10),
        # p95 of 27 items -- the actual eval set size -- is the 26th, NOT the max.
        (list(range(1, 28)), 95, 26),
        ([42], 50, 42),
        ([42], 95, 42),
        ([], 50, 0.0),
    ],
)
def test_percentile_is_nearest_rank(values, pct, expected):
    assert run_eval._percentile(list(values), pct) == expected


def test_percentile_does_not_interpolate():
    """`statistics.quantiles` would return 5.5 here. A p95 that invents a latency
    no request actually had is a worse number to report than a real one."""
    assert run_eval._percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) in (5, 5.0)


def test_percentile_ignores_input_order():
    import random

    values = list(range(1, 51))
    shuffled = values[:]
    random.shuffle(shuffled)
    assert run_eval._percentile(shuffled, 95) == run_eval._percentile(values, 95)


# -------------------------------------------------------------- the scorer

def _envelope(**kw):
    base = {"answer": "", "citations": [], "trace": [],
            "requires_confirmation": False, "escalation": None}
    return {**base, **kw}


def test_an_unasserted_check_is_None_not_a_pass():
    """None keeps the item out of that check's denominator entirely.

    Counting it as a pass inflates every rate by however many items were silent
    on that check; counting it as a failure deflates them. Both are wrong, and
    with 27 items and 6 checks most cells are legitimately silent.
    """
    checks = run_eval.score({"expected": {}}, _envelope())
    assert set(checks) == set(run_eval.CHECK_NAMES)
    assert all(v is None for v in checks.values())


def test_citation_check_requires_every_named_document():
    """A multi-document answer that finds one of two is not half right."""
    item = {"expected": {"must_cite_doc_ids": ["REMOTE-WORK", "TAX-LOCATION"]}}

    both = _envelope(citations=[{"doc_id": "REMOTE-WORK"}, {"doc_id": "TAX-LOCATION"}])
    assert run_eval.score(item, both)["citations"] is True

    one = _envelope(citations=[{"doc_id": "REMOTE-WORK"}])
    assert run_eval.score(item, one)["citations"] is False

    # Extra documents are not penalised -- citing a third relevant policy is
    # better, not worse, and the brief asks for breadth.
    extra = _envelope(citations=[{"doc_id": "REMOTE-WORK"}, {"doc_id": "TAX-LOCATION"},
                                 {"doc_id": "INFOSEC"}])
    assert run_eval.score(item, extra)["citations"] is True


def test_must_not_call_is_scored_as_tool_selection():
    """Action safety rests on this: a forbidden tool must fail the item even when
    every required tool was also called."""
    item = {"expected": {"must_call": ["check_pto_balance"],
                         "must_not_call": ["create_mock_hr_ticket"]}}
    trace = [{"type": "tool_call", "tool": "check_pto_balance"},
             {"type": "tool_call", "tool": "create_mock_hr_ticket"}]
    assert run_eval.score(item, _envelope(trace=trace))["tool_selection"] is False


def test_throttle_detection_reads_the_guardrail_step():
    """A throttled turn degrades quietly -- it answers with less evidence rather
    than erroring. Without this signal a rate limit is scored as a wrong answer."""
    throttled = _envelope(trace=[
        {"type": "guardrail", "status": "error",
         "result_summary": "model call failed: RateLimitError"}])
    assert run_eval._throttled(throttled) is True

    ordinary_error = _envelope(trace=[
        {"type": "guardrail", "status": "error",
         "result_summary": "refused to act: employee mismatch"}])
    assert run_eval._throttled(ordinary_error) is False


# --------------------------------------------------------------- the ablation

def test_ablation_refuses_an_empty_vector_index():
    """The guard that stops a BM25-only run being reported as hybrid.

    `retrieve()` returns retrieval_mode "hybrid" whether or not the vector leg
    contributed anything, so an unbuilt `data/chroma` produces a full, plausible,
    completely wrong ablation. It did, once.
    """
    ablation = _load("ablation")

    import app.rag.store as store

    real_count = store.count
    store.count = lambda *a, **k: 0
    try:
        with pytest.raises(SystemExit, match="build_index"):
            ablation.run("evaluation/eval_set.*.json")
    finally:
        store.count = real_count


def test_ablation_degraded_mode_silences_the_vector_leg():
    """The context manager must restore the real function, or every later cell in
    the same process silently measures BM25 only."""
    ablation = _load("ablation")
    from app.rag import retrieve as retrieve_module

    original = retrieve_module._vector_ranked
    with ablation._vector_leg_unavailable():
        assert retrieve_module._vector_ranked("anything", None, 5) == []
    assert retrieve_module._vector_ranked is original
