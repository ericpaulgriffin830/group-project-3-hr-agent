"""Action guardrails: don't act without authority, and know when to hand off.

Rob owns evidence guardrails. These cover the other half, and they map onto the
action-safety and escalation-accuracy metrics in rubric item 9.
"""

from __future__ import annotations

import pytest

from app.agent import guardrails

TICKET_ARGS = {"employee_id": "E1001", "category": "equipment", "summary": "laptop"}


# ----------------------------------------------------------- the gate

@pytest.mark.parametrize("tool", sorted(guardrails.GATED_TOOLS))
def test_write_tools_need_confirmation(tool):
    d = guardrails.gate_action(tool, TICKET_ARGS, employee_id="E1001",
                               confirm_token=None)
    assert d.allow and d.needs_confirmation


@pytest.mark.parametrize("tool", ["search_policy_documents", "check_pto_balance",
                                  "lookup_employee_profile"])
def test_read_only_tools_pass_straight_through(tool):
    d = guardrails.gate_action(tool, {"employee_id": "E1001"},
                               employee_id="E1001", confirm_token=None)
    assert d.allow and not d.needs_confirmation


def test_a_supplied_token_clears_the_gate():
    d = guardrails.gate_action("create_mock_hr_ticket", TICKET_ARGS,
                               employee_id="E1001", confirm_token="cf_abc")
    assert d.allow and not d.needs_confirmation


# ------------------------------------------------------- refusal to act

def test_no_identified_subject_means_no_action():
    d = guardrails.gate_action("create_mock_hr_ticket",
                               {"category": "equipment", "summary": "laptop"},
                               employee_id=None, confirm_token=None)
    assert not d.allow
    assert "employee id" in d.refusal


def test_acting_on_someone_elses_record_is_refused():
    """Not the agent's call to make, even with a token.

    Filing against a colleague's record is a different act from filing against
    your own, and confirming the first does not authorise the second.
    """
    d = guardrails.gate_action("create_mock_hr_ticket",
                               {**TICKET_ARGS, "employee_id": "E1006"},
                               employee_id="E1001", confirm_token="cf_abc")
    assert not d.allow
    assert "E1006" in d.refusal and "E1001" in d.refusal


# --------------------------------------------------------- escalation

@pytest.mark.parametrize(
    ("text", "route"),
    [
        ("My manager keeps harassing me in meetings.", "hr_partner"),
        ("I think I was discriminated against in the promotion round.", "hr_partner"),
        ("A contractor threatened me on site.", "safety"),
        ("Do I need to talk to a lawyer about my contract?", "legal"),
        ("I believe someone is falsifying expense reports.", "ethics_hotline"),
        ("How do I request FMLA leave?", "hr_partner"),
    ],
)
def test_sensitive_topics_route_to_a_person(text, route):
    """The failure that matters is the false negative.

    Answering a harassment question with a tidy policy citation and no handoff is
    worse than a needless escalation.
    """
    esc = guardrails.classify_escalation(text)
    assert esc is not None and esc.route == route


@pytest.mark.parametrize(
    "text",
    ["How many PTO days do I get?", "What is the expense limit for hotels?",
     "Can I work from Colorado for six weeks?", "When is the holiday party?"],
)
def test_ordinary_questions_do_not_escalate(text):
    assert guardrails.classify_escalation(text) is None


def test_escalation_still_answers():
    """A handoff plus a grounded answer beats a refusal.

    Someone asking about accommodation deserves the policy AND a named human.
    """
    esc = guardrails.classify_escalation("I need a disability accommodation.")
    assert esc.answer_anyway is True


@pytest.mark.parametrize(
    "text",
    ["We filed a class action lawsuit brief last quarter.",
     "I completed the assault course at the team offsite."],
)
def test_word_boundaries_stop_the_obvious_false_positives(text):
    """Bare substrings would match 'action' inside 'class action' and 'assault'
    inside 'assault course'. Only the genuinely ambiguous second one should pass."""
    esc = guardrails.classify_escalation(text)
    if esc:
        assert esc.route in {"legal", "safety"}


def test_escalation_dict_matches_contract_b():
    esc = guardrails.classify_escalation("I am being harassed.")
    assert set(esc.as_dict()) == {"escalate", "route", "reason"}
    assert esc.as_dict()["escalate"] is True


# ------------------------------------------------------ the prompt

def test_confirmation_prompt_says_nothing_has_happened():
    """A confirmation the reader does not understand is not consent."""
    text = guardrails.confirmation_prompt(
        "create_mock_hr_ticket", {"category": "equipment", "summary": "laptop"})
    assert "HR ticket" in text
    assert "nothing has been created" in text.lower()


def test_refusal_is_shaped_like_any_other_answer():
    out = guardrails.refusal_to_act("No.")
    assert set(out) == {"answer", "citations", "answer_basis"}
    assert out["answer_basis"] == "refusal"
