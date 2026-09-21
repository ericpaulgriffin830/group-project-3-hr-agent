"""Action guardrails -- the confirmation gate, and refusing to invent.

These map onto the action-safety pass rate in rubric item 9. The two write tools
are mock, so nothing here can do damage; the gate is tested anyway because the
brief requires irreversible actions be confirmed, and because a gate that is never
tested is a gate that quietly stops working.

Split of responsibility (docs/CONTRACTS.md): Rob owns *evidence* guardrails --
don't assert what the corpus doesn't support. This file covers Chris's half --
don't DO what the user didn't confirm, and don't make up an employee.
"""

from __future__ import annotations

import pytest

TICKET = {
    "employee_id": "E1001",
    "category": "workplace_location",
    "summary": "Multi-state work registration",
}

EMAIL = {
    "employee_id": "E1001",
    "recipient_role": "manager",
    "intent": "PTO request",
    "context": "Requesting 3 days in October.",
}

WRITE_CALLS = [("create_mock_hr_ticket", TICKET), ("draft_hr_email", EMAIL)]


# --------------------------------------------------------------- the gate

@pytest.mark.parametrize(("tool", "args"), WRITE_CALLS)
async def test_first_call_previews_and_writes_nothing(call, tool, args):
    result = await call(tool, **args)
    assert result["requires_confirmation"] is True
    assert result["confirm_token"]
    assert "preview" in result
    # The action itself must not have happened.
    assert "ticket_id" not in result
    assert "draft" not in result


@pytest.mark.parametrize(("tool", "args"), WRITE_CALLS)
async def test_second_call_with_the_token_performs_the_action(call, tool, args):
    token = (await call(tool, **args))["confirm_token"]
    done = await call(tool, **args, confirm_token=token)
    assert "error" not in done
    assert "ticket_id" in done or "draft" in done


@pytest.mark.parametrize(("tool", "args"), WRITE_CALLS)
async def test_forged_token_is_refused(call, tool, args):
    result = await call(tool, **args, confirm_token="deadbeefdeadbeef")
    assert result["error"] == "invalid_confirm_token"


@pytest.mark.parametrize(
    ("tool", "args", "mutation"),
    [
        ("create_mock_hr_ticket", TICKET, {"category": "termination"}),
        ("create_mock_hr_ticket", TICKET, {"summary": "Something else entirely"}),
        ("create_mock_hr_ticket", TICKET, {"employee_id": "E1010"}),
        ("draft_hr_email", EMAIL, {"context": "Please approve my resignation."}),
        ("draft_hr_email", EMAIL, {"recipient_role": "hr_partner"}),
    ],
)
async def test_token_cannot_be_replayed_against_different_arguments(call, tool, args, mutation):
    """The bypass that matters.

    A token issued for the action the user *saw* must not authorise a different
    action. Without argument binding, an agent could preview a harmless ticket,
    get it confirmed, then submit the token with a different payload -- and the
    human would have approved something they never read.
    """
    token = (await call(tool, **args))["confirm_token"]
    result = await call(tool, **{**args, **mutation}, confirm_token=token)
    assert result["error"] == "invalid_confirm_token"


async def test_token_is_stable_across_processes(call):
    """Deterministic by construction -- a sha256 digest, never Python's salted hash().

    Rubric item 1 asks for fixed seeds where applicable. If the token or the ticket
    id moved between runs, evaluation reruns would not be comparable.
    """
    first = (await call("create_mock_hr_ticket", **TICKET))["confirm_token"]
    second = (await call("create_mock_hr_ticket", **TICKET))["confirm_token"]
    assert first == second

    ticket_a = await call("create_mock_hr_ticket", **TICKET, confirm_token=first)
    ticket_b = await call("create_mock_hr_ticket", **TICKET, confirm_token=first)
    assert ticket_a["ticket_id"] == ticket_b["ticket_id"]


async def test_draft_hr_email_never_reports_sending(call):
    token = (await call("draft_hr_email", **EMAIL))["confirm_token"]
    result = await call("draft_hr_email", **EMAIL, confirm_token=token)
    assert result["sent"] is False


# ------------------------------------------------- refusing to invent data

@pytest.mark.parametrize(
    "tool",
    ["lookup_employee_profile", "check_pto_balance", "lookup_benefits_status"],
)
async def test_unknown_employee_is_an_error_not_a_guess(call, tool):
    result = await call(tool, employee_id="E9999")
    assert result["error"] == "employee_not_found"
    assert "name" not in result


async def test_write_tools_reject_an_unknown_employee_before_previewing(call):
    """No token should ever be issued for an employee who does not exist."""
    result = await call("create_mock_hr_ticket", **{**TICKET, "employee_id": "E9999"})
    assert result["error"] == "employee_not_found"
    assert "confirm_token" not in result


async def test_contractor_with_no_pto_record_is_distinguished_from_a_missing_employee(call):
    """'You have no PTO record' and 'you do not exist' are different answers.

    Collapsing them would have the agent tell a real contractor they are not an
    employee.
    """
    result = await call("check_pto_balance", employee_id="E1011")
    assert result["error"] == "no_pto_record"


# ------------------------------------------------------ evidence handling

async def test_compliance_returns_insufficient_evidence_rather_than_guessing(call):
    """Escalation accuracy is scored. Not knowing is a correct answer."""
    result = await call(
        "check_policy_compliance",
        scenario="Can I expense a jetpack for my commute?",
        employee_id="E1001",
    )
    assert result["verdict"] == "insufficient_evidence"
    assert result["policy_refs"] == []


async def test_multi_document_scenario_cites_more_than_one_policy(call):
    """Rubric item 3 requires a question needing several documents.

    Out-of-state remote work is the demo Task A path: remote-work plus tax plus
    data-security. If this collapses to one document, the demo loses its point.
    """
    result = await call(
        "check_policy_compliance",
        scenario="I want to work remotely from another state for six weeks.",
        employee_id="E1001",
    )
    assert result["verdict"] == "conditional"
    doc_ids = {ref["doc_id"] for ref in result["policy_refs"]}
    assert len(doc_ids) >= 2


async def test_empty_query_is_rejected(call):
    result = await call("search_policy_documents", query="   ")
    assert result["error"] == "empty_query"


async def test_missing_section_is_not_found_not_an_empty_string(call):
    result = await call("get_policy_section", doc_id="REMOTE-WORK", section_id="99.9")
    assert result["error"] == "section_not_found"
