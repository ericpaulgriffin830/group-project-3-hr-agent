"""Contract A in prose vs Contract A in code.

`docs/CONTRACTS.md` describes the eight tools, `mcp_server/schemas.py` models them,
and `mcp_server/server.py` implements them. Three copies of one contract drift
unless something checks. This is that something.

Every test here calls the tool for real, through the MCP layer, and validates the
actual payload. A test that validated a hand-written dict would only prove the
models parse themselves.
"""

from __future__ import annotations

import pytest

from mcp_server import schemas
from mcp_server.server import mcp

TICKET = {"employee_id": "E1001", "category": "workplace_location",
          "summary": "Multi-state work registration"}
EMAIL = {"employee_id": "E1001", "recipient_role": "manager",
         "intent": "PTO request", "context": "Requesting 3 days in October."}


# ------------------------------------------------- registry covers the surface

async def test_registry_matches_the_discovered_tools_exactly():
    """schemas.py must model every tool the server exposes, and no ghosts.

    A tool added to the server without a model here would go unvalidated; a model
    for a tool that no longer exists is a stale contract nobody notices.
    """
    discovered = {t.name for t in await mcp.list_tools()}
    assert set(schemas.TOOL_SCHEMAS) == discovered


@pytest.mark.parametrize("tool", sorted(schemas.TOOL_SCHEMAS))
async def test_input_model_agrees_with_the_published_mcp_schema(tool):
    """The required arguments in the model and in the wire schema must match.

    These are what Rob and Eric code against. If they disagree, one of them is
    building against a contract the server does not honour.
    """
    published = {t.name: t for t in await mcp.list_tools()}[tool].input_schema
    in_model, _ = schemas.TOOL_SCHEMAS[tool]

    model_required = {
        name for name, f in in_model.model_fields.items() if f.is_required()
    }
    assert model_required == set(published.get("required", []))
    assert set(in_model.model_fields) == set(published.get("properties", {}))


# ------------------------------------------------ real payloads, real models

@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("search_policy_documents", {"query": "remote work out of state"}),
        ("get_policy_section", {"doc_id": "REMOTE-WORK", "section_id": "RW-2"}),
        ("lookup_employee_profile", {"employee_id": "E1001"}),
        ("check_pto_balance", {"employee_id": "E1001"}),
        ("lookup_benefits_status", {"employee_id": "E1001"}),
        ("check_policy_compliance",
         {"scenario": "work remotely from another state", "employee_id": "E1001"}),
        ("create_mock_hr_ticket", TICKET),
        ("draft_hr_email", EMAIL),
    ],
)
async def test_every_tool_returns_a_shape_the_contract_declares(call, tool, args):
    payload = await call(tool, **args)
    validated = schemas.validate_output(tool, payload)
    assert not isinstance(validated, schemas.ToolError)


async def test_write_tools_validate_in_both_phases(call):
    """Preview and result are different shapes, and both are in the contract."""
    for tool, args in (("create_mock_hr_ticket", TICKET), ("draft_hr_email", EMAIL)):
        preview = await call(tool, **args)
        assert isinstance(schemas.validate_output(tool, preview),
                          schemas.ConfirmationRequired)

        done = await call(tool, **args, confirm_token=preview["confirm_token"])
        assert isinstance(schemas.validate_output(tool, done),
                          schemas.ConfirmationRequired) is False
        schemas.validate_output(tool, done)


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("lookup_employee_profile", {"employee_id": "E9999"}),
        ("check_pto_balance", {"employee_id": "E1011"}),
        ("get_policy_section", {"doc_id": "REMOTE-WORK", "section_id": "99.9"}),
        ("search_policy_documents", {"query": "   "}),
    ],
)
async def test_error_payloads_validate_as_ToolError(call, tool, args):
    """Every declined call is still a contract-shaped response, never a raise."""
    payload = await call(tool, **args)
    validated = schemas.validate_output(tool, payload)
    assert isinstance(validated, schemas.ToolError)
    assert validated.error in schemas.ErrorCode.__args__


# ------------------------------------------------------ drift is caught

def test_an_extra_field_fails_validation():
    """Outputs are extra="forbid" on purpose.

    A tool quietly growing a field is a contract change, and change control says
    those go through a PR. Tolerating it here would make that rule unenforceable.
    """
    good = {"doc_id": "REMOTE-WORK", "title": "Remote Work Policy",
            "section": "RW-2", "text": "..."}
    schemas.GetPolicySectionOut.model_validate(good)

    with pytest.raises(Exception):
        schemas.GetPolicySectionOut.model_validate({**good, "confidence": 0.9})


def test_a_value_outside_a_closed_set_fails_validation():
    with pytest.raises(Exception):
        schemas.PolicyComplianceOut.model_validate(
            {"verdict": "probably_fine", "conditions": [],
             "policy_refs": [], "rationale": "..."})


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="Contract A"):
        schemas.validate_output("summon_hr_demon", {"ok": True})


def test_draft_hr_email_cannot_report_itself_as_sent():
    """`sent` is Literal[False]. There is no send capability and no path to one."""
    draft = {"to": "Priya Raman", "subject": "s", "body": "b"}
    schemas.DraftHrEmailOut.model_validate({"draft": draft, "sent": False})

    with pytest.raises(Exception):
        schemas.DraftHrEmailOut.model_validate({"draft": draft, "sent": True})


# ------------------------------------- section ids as search results hand them back

async def test_get_policy_section_accepts_the_section_string_from_a_chunk(call):
    """A chunk's `section` is "RW-3 Temporary Remote Work"; the tool keys on "RW-3".

    Advertising a value in one shape and demanding another is the tool's fault.
    The agent was reading the section off a search result, getting
    section_not_found, and spending tool steps guessing at how to split it.
    """
    found = await call("search_policy_documents", query="remote work another state", k=3)
    chunk = found["chunks"][0]

    verbatim = await call("get_policy_section", doc_id=chunk["doc_id"],
                          section_id=chunk["section"])
    assert "error" not in verbatim, verbatim

    bare = await call("get_policy_section", doc_id=chunk["doc_id"],
                      section_id=chunk["section"].split()[0])
    assert "error" not in bare
    assert verbatim["text"] == bare["text"]


async def test_a_genuinely_missing_section_still_errors(call):
    """Tolerance must not turn a real miss into a silent wrong answer."""
    result = await call("get_policy_section", doc_id="REMOTE-WORK",
                        section_id="RW-999 Nonexistent Section")
    assert result["error"] == "section_not_found"
