"""MCP tool discovery.

Rubric item 8 requires CI to run "at least one test or script that verifies MCP
tool discovery or a simple MCP tool call". This file is that test.

It asserts against the tool *surface* -- names, required arguments, descriptions --
because that surface is Contract A in docs/CONTRACTS.md. Rob and Eric build against
these names. If one is renamed, this fails before their code does.
"""

from __future__ import annotations

import pytest

from mcp_server.server import mcp

# Contract A. Order does not matter; membership does.
EXPECTED_TOOLS = {
    "search_policy_documents",
    "get_policy_section",
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "check_policy_compliance",
    "create_mock_hr_ticket",
    "draft_hr_email",
}

RAG_TOOLS = {"search_policy_documents", "get_policy_section"}
MOCK_DATA_TOOLS = {"lookup_employee_profile", "check_pto_balance", "lookup_benefits_status"}
WRITE_TOOLS = {"create_mock_hr_ticket", "draft_hr_email"}


async def _tools_by_name() -> dict:
    return {t.name: t for t in await mcp.list_tools()}


async def test_discovery_returns_the_contract_a_tools():
    names = set(await _tools_by_name())
    assert names == EXPECTED_TOOLS


async def test_brief_minimums_are_met():
    """>=5 tools, >=1 backed by RAG, >=1 using mock data or a mock operation."""
    names = set(await _tools_by_name())
    assert len(names) >= 5
    assert names & RAG_TOOLS
    assert names & (MOCK_DATA_TOOLS | WRITE_TOOLS)


@pytest.mark.parametrize(
    ("tool", "required"),
    [
        ("search_policy_documents", {"query"}),
        ("get_policy_section", {"doc_id", "section_id"}),
        ("lookup_employee_profile", {"employee_id"}),
        ("check_pto_balance", {"employee_id"}),
        ("lookup_benefits_status", {"employee_id"}),
        ("check_policy_compliance", {"scenario", "employee_id"}),
        ("create_mock_hr_ticket", {"employee_id", "category", "summary"}),
        ("draft_hr_email", {"employee_id", "recipient_role", "intent", "context"}),
    ],
)
async def test_required_arguments_match_the_contract(tool, required):
    schema = (await _tools_by_name())[tool].input_schema
    assert set(schema.get("required", [])) == required


async def test_confirm_token_is_optional_on_write_tools():
    """The gate works by *omitting* the token on the first call.

    If confirm_token were required, there would be no way to request a preview and
    the confirmation flow could not exist.
    """
    tools = await _tools_by_name()
    for name in WRITE_TOOLS:
        schema = tools[name].input_schema
        assert "confirm_token" in schema["properties"]
        assert "confirm_token" not in schema.get("required", [])


async def test_every_tool_describes_itself():
    """The agent selects tools from these descriptions, so an empty one is a bug.

    Tool-selection accuracy is a scored metric; a tool the model cannot tell apart
    from its neighbours is the usual cause of losing it.
    """
    for name, tool in (await _tools_by_name()).items():
        assert tool.description and tool.description.strip(), f"{name} has no description"


async def test_a_simple_tool_call_round_trips(call):
    """Discovery is not enough -- prove a tool actually executes through the MCP layer."""
    profile = await call("lookup_employee_profile", employee_id="E1001")
    assert profile["employee_id"] == "E1001"
    assert "error" not in profile
