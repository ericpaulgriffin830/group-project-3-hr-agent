"""The transport adapter the agent reaches tools through.

Most tests here connect in-process for speed -- still a real `Client` and a real
session, just no subprocess. One test deliberately spawns the server over stdio,
because that is the path the deployed app uses and an in-process-only suite would
not prove it works.
"""

from __future__ import annotations

import ast
import pathlib
from contextlib import asynccontextmanager

import pytest

from app.agent.mcp_client import MCPClient, MCPUnavailable, ToolCall, _server_target
from mcp_server.server import mcp

@asynccontextmanager
async def connected():
    """Connect in-process -- a real Client and a real session, just no subprocess.

    Deliberately NOT a pytest fixture. An async-generator fixture is entered and
    exited in different tasks, and MCP's anyio task groups reject that with
    "Attempted to exit cancel scope in a different task than it was entered in".
    Entering inside the test keeps both ends on one task.
    """
    async with MCPClient(server=mcp) as c:
        await c.discover()
        yield c


# ------------------------------------------------------------- discovery

async def test_discovery_finds_the_eight_tools():
    async with connected() as client:
        assert len(client.tools) == 8
        assert "check_pto_balance" in client.tools


async def test_discovery_carries_schemas_and_descriptions():
    """The model selects tools from these. Empty descriptions lose selection accuracy."""
    async with connected() as client:
        for name, spec in client.tools.items():
            assert spec.description, f"{name} has no description"
            assert spec.input_schema.get("type") == "object"

        assert client.tools["check_pto_balance"].required == {"employee_id"}
        assert client.tools["get_policy_section"].required == {"doc_id", "section_id"}


async def test_nothing_hardcodes_the_tool_list():
    """Names come from the server, never from a list the client keeps.

    The antipattern this guards against is a literal roster --
    `TOOLS = ["search_policy_documents", ...]` -- which would make discovery
    decorative and the brief's "the agent must actually call MCP-exposed tools"
    untrue. Checked via AST for a collection holding two or more tool names, so a
    single name in a docstring example does not trip it.
    """
    async with connected() as client:
        known = set(client.tool_names())

    tree = ast.parse(pathlib.Path("app/agent/mcp_client.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            literals = {e.value for e in node.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            overlap = literals & known
            assert len(overlap) < 2, f"tool roster hardcoded at line {node.lineno}: {overlap}"


# --------------------------------------------- the rubric boundary itself

def test_agent_layer_never_imports_the_server_directly():
    """The brief: hard-coded direct function calls do not count.

    Everything under app/agent must reach tools through the MCP layer. This walks
    the AST rather than grepping, so a `from mcp_server.server import ...` cannot
    slip in disguised by formatting.
    """
    offenders = []
    for path in pathlib.Path("app/agent").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for n in names:
                if n.split(".")[0] == "mcp_server":
                    offenders.append(f"{path}:{node.lineno} imports {n}")
    assert not offenders, "agent layer bypasses the MCP layer:\n" + "\n".join(offenders)


# ------------------------------------------------------------- calling

async def test_successful_call_returns_payload_and_latency():
    async with connected() as client:
        call = await client.call("lookup_employee_profile", employee_id="E-1043")
        assert call.ok
        # Assert on the id, not the display name -- renaming a fixture employee is
        # cosmetic and should not break the transport test.
        assert call.payload["employee_id"] == "E-1043"
        assert call.payload["name"]
        assert call.latency_ms >= 0


async def test_tool_refusal_is_not_a_transport_failure():
    """A tool declining is a contract-shaped answer with its own code.

    Collapsing it into `tool_unavailable` would have the orchestrator degrade to
    RAG-only when the real answer is "that employee does not exist" -- a different
    response to the user entirely.
    """
    async with connected() as client:
        call = await client.call("lookup_employee_profile", employee_id="E-9999")
        assert call.ok is False
        assert call.error == "employee_not_found"


async def test_unknown_tool_is_refused_before_reaching_the_wire():
    """A hallucinated tool name should fail legibly, not as a transport error."""
    async with connected() as client:
        call = await client.call("summon_hr_demon", x=1)
        assert call.ok is False
        assert call.error == "tool_unavailable"
        assert "summon_hr_demon" in call.message


async def test_calling_before_connecting_returns_rather_than_raises():
    """Graceful failure is a rubric item; an exception into the graph is not."""
    call = await MCPClient(server=mcp).call("check_pto_balance", employee_id="E-1043")
    assert isinstance(call, ToolCall)
    assert call.ok is False
    assert call.error == "tool_unavailable"


async def test_discover_before_connecting_raises():
    """The one place we DO raise: never having discovered is misconfiguration.

    Returning an empty tool list would let the app start looking healthy while the
    agent silently has nothing to call.
    """
    with pytest.raises(MCPUnavailable):
        await MCPClient(server=mcp).discover()


# ------------------------------------------------------------- the trace

async def test_trace_step_matches_contract_b_and_omits_reasoning():
    async with connected() as client:
        call = await client.call("check_pto_balance", employee_id="E-1043")
        step = call.as_trace_step(2)

        assert set(step) == {"step", "type", "tool", "args", "result_summary",
                         "status", "latency_ms"}
        assert step["type"] == "tool_call"
        assert step["status"] == "ok"
        # Operational only. No rationale, and not the whole payload.
        assert "reasoning" not in step
        assert len(step["result_summary"]) <= 120


async def test_failed_call_traces_as_error_with_its_code():
    async with connected() as client:
        step = (await client.call("lookup_employee_profile",
                              employee_id="E-9999")).as_trace_step(1)
        assert step["status"] == "error"
        assert "employee_not_found" in step["result_summary"]


# ------------------------------------------------------------- transport

def test_transport_selection(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    assert getattr(_server_target(), "command", None) is not None

    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8765/mcp")
    assert _server_target() == "http://localhost:8765/mcp"


def test_streamable_http_without_a_url_is_rejected(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    with pytest.raises(MCPUnavailable, match="MCP_SERVER_URL"):
        _server_target()


def test_unknown_transport_is_rejected(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "carrier-pigeon")
    with pytest.raises(MCPUnavailable, match="carrier-pigeon"):
        _server_target()


async def test_health_reports_connection_state():
    async with connected() as client:
        health = client.health()
        assert health["mcp_connected"] is True
        assert health["tools_discovered"] == 8


# ------------------------------------------- the real deployed path

@pytest.mark.slow
async def test_stdio_subprocess_end_to_end(monkeypatch):
    """Spawn the server as a real subprocess and talk to it over stdio.

    This is the transport the deployed app uses. Everything above runs in-process,
    so without this the suite could stay green while the shipped path was broken.
    """
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    async with MCPClient() as c:
        tools = await c.discover()
        assert len(tools) == 8
        call = await c.call("lookup_employee_profile", employee_id="E-1043")
        assert call.ok and call.payload["employee_id"] == "E-1043"
