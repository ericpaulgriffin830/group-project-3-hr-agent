"""The operational trace.

The brief: "Do not expose hidden chain-of-thought; provide concise operational
traces instead." That is the rule these tests exist to keep true as other people
add nodes.
"""

from __future__ import annotations

import pytest

from app.agent.trace import (SUMMARY_LIMIT, ChainOfThoughtLeak, Trace, _scrub_args)


def test_steps_number_from_one_in_order():
    t = Trace()
    t.add("intent", result_summary="workflow")
    t.add("tool_call", tool="check_pto_balance", result_summary="ok")
    assert [s["step"] for s in t.as_list()] == [1, 2]


def test_unknown_step_type_is_rejected():
    with pytest.raises(ValueError, match="not a trace step type"):
        Trace().add("vibes", result_summary="hmm")


@pytest.mark.parametrize(
    "field",
    ["reasoning", "chain_of_thought", "thought", "scratchpad", "thinking",
     "reasoning_content", "deliberation", "analysis"],
)
def test_reasoning_shaped_fields_are_refused_loudly(field):
    """The leak arrives as a forwarded field, not a decision.

    gpt-oss returns `reasoning` next to `content`; a node passing a raw provider
    response into a step would ship literal chain-of-thought to the UI with nothing
    looking wrong. Dropping it silently would let the mistake recur.
    """
    with pytest.raises(ChainOfThoughtLeak):
        Trace().add("synthesis", result_summary="done", **{field: "first I..."})


def test_result_summary_is_truncated():
    t = Trace()
    step = t.add("tool_call", tool="x", result_summary="y" * 400)
    assert len(step["result_summary"]) <= SUMMARY_LIMIT


def test_confirm_token_is_redacted_from_recorded_args():
    """A trace panel renders in a browser and gets pasted into bug reports.

    It should show that a token was present, not what it was -- the token
    authorises a write.
    """
    step = Trace().add("tool_call", tool="create_mock_hr_ticket",
                       args={"employee_id": "E-1043", "confirm_token": "cf_secret"})
    assert step["args"]["confirm_token"] == "<redacted>"
    assert step["args"]["employee_id"] == "E-1043"
    assert "cf_secret" not in str(step)


def test_timings_split_retrieval_from_tools():
    t = Trace()
    t.add("retrieval", tool="search_policy_documents", latency_ms=40)
    t.add("tool_call", tool="check_pto_balance", latency_ms=10)
    t.add("tool_call", tool="lookup_employee_profile", latency_ms=5)
    assert t.timings() == {"retrieval_ms": 40, "tools_ms": 15, "total_ms": 55}


def test_tools_used_is_ordered_and_feeds_the_metric():
    t = Trace()
    t.add("intent", result_summary="workflow")
    t.add("tool_call", tool="lookup_employee_profile")
    t.add("tool_call", tool="check_pto_balance")
    assert t.tools_used() == ["lookup_employee_profile", "check_pto_balance"]


def test_had_error_detects_a_failed_step():
    t = Trace()
    t.add("tool_call", tool="x", status="ok")
    assert not t.had_error()
    t.add("tool_call", tool="y", status="error")
    assert t.had_error()


def test_render_is_human_readable():
    t = Trace()
    t.add("tool_call", tool="check_pto_balance", result_summary="6.5 days",
          latency_ms=12)
    assert "check_pto_balance" in t.render()
    assert "12ms" in t.render()


def test_scrubber_leaves_ordinary_args_alone():
    assert _scrub_args({"query": "pto", "k": 5}) == {"query": "pto", "k": 5}
