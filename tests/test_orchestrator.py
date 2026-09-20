"""The LangGraph orchestrator.

The chat model is scripted, not live. CI has no API key, and a test whose outcome
depends on what a model decides today is not a regression test -- it is a coin
flip that fails on the morning of the demo. The MCP side is real throughout: every
tool call here crosses an actual MCP session.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from app.agent import tools as tool_bridge
from app.agent.mcp_client import MCPClient
from app.agent.orchestrator import MAX_TOOL_STEPS, answer, build_graph
from mcp_server.server import mcp


@asynccontextmanager
async def connected():
    async with MCPClient(server=mcp) as c:
        await c.discover()
        yield c


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChat:
    """A chat model that replays a script.

    `script` is a list of FakeMessage. Each ainvoke pops the next one; the last
    repeats, so a graph that loops does not run off the end of the script.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]


def tool_call(name, args, call_id="c1"):
    return {"id": call_id, "name": name, "args": args}


# ------------------------------------------------------------- routing

@pytest.mark.parametrize(
    ("intent", "expect_first_after_classify"),
    [("policy_qa", "retrieval"), ("refuse", "guardrail"), ("clarify", "guardrail")],
)
async def test_classified_intent_picks_the_path(intent, expect_first_after_classify):
    async with connected() as client:
        out = await answer("anything", client=client,
                           chat_model=FakeChat([FakeMessage(intent)]))
    assert out["trace"][0]["result_summary"] == intent
    assert out["trace"][1]["type"] == expect_first_after_classify


async def test_unparseable_intent_falls_back_to_policy_qa():
    """A model that answers off-script must not wedge the graph."""
    async with connected() as client:
        out = await answer("x", client=client,
                           chat_model=FakeChat([FakeMessage("¯\\_(ツ)_/¯")]))
    assert out["trace"][0]["result_summary"] == "policy_qa"


# --------------------------------------------------------- the tool loop

async def test_workflow_path_executes_tools_through_mcp():
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("check_pto_balance", {"employee_id": "E-1043"})]),
        FakeMessage("You have 6.5 days available."),
    ]
    async with connected() as client:
        out = await answer("how much pto", client=client, chat_model=FakeChat(script),
                           employee_id="E-1043")

    tool_steps = [s for s in out["trace"] if s["type"] == "tool_call"]
    assert [s["tool"] for s in tool_steps] == ["check_pto_balance"]
    assert tool_steps[0]["status"] == "ok"


async def test_tool_loop_is_capped():
    """A model that keeps calling the same tool must not loop until timeout."""
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("check_pto_balance", {"employee_id": "E-1043"})]),
    ]
    async with connected() as client:
        out = await answer("loop forever", client=client, chat_model=FakeChat(script))

    assert len([s for s in out["trace"] if s["type"] == "tool_call"]) == MAX_TOOL_STEPS


async def test_a_failing_tool_does_not_stop_the_graph():
    """Graceful failure is a rubric item. The error is traced and the turn finishes."""
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("lookup_employee_profile",
                                          {"employee_id": "E-9999"})]),
        FakeMessage("I could not find that employee."),
    ]
    async with connected() as client:
        out = await answer("who is E-9999", client=client, chat_model=FakeChat(script))

    failed = [s for s in out["trace"] if s.get("status") == "error"]
    assert failed and "employee_not_found" in failed[0]["result_summary"]
    assert out["answer"]


# ------------------------------------------------- the confirmation gate

async def test_write_tool_short_circuits_and_creates_nothing():
    """The safety property.

    A pending write must end the turn and wait for a human. If the graph looped
    back to the model instead, the model would be the thing 'confirming' -- which
    is exactly what the gate exists to prevent.
    """
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("create_mock_hr_ticket", {
            "employee_id": "E-1043", "category": "equipment",
            "summary": "Laptop replacement"})]),
        FakeMessage("should never be reached"),
    ]
    async with connected() as client:
        out = await answer("file a ticket", client=client, chat_model=FakeChat(script))

    assert out["requires_confirmation"] is True
    assert out["basis"] == "awaiting_confirmation"
    assert out["pending_action"]["tool"] == "create_mock_hr_ticket"
    assert out["pending_action"]["confirm_token"]
    assert "ticket_id" not in str(out["pending_action"])
    assert out["trace"][-1]["type"] == "guardrail"


async def test_the_model_is_never_offered_confirm_token():
    """The token is the human's, supplied by the UI -- not an argument to invent."""
    async with connected() as client:
        for definition in tool_bridge.tool_definitions(client):
            params = definition["function"]["parameters"]
            assert "confirm_token" not in params.get("properties", {})
            assert "confirm_token" not in params.get("required", [])


async def test_bound_tools_carry_no_executable_callable():
    """Descriptions only.

    If a LangChain tool held a local Python function, the framework would execute
    it directly and the brief's "must actually call MCP-exposed tools" would be
    false while everything still worked.
    """
    async with connected() as client:
        for definition in tool_bridge.tool_definitions(client):
            assert set(definition) == {"type", "function"}
            assert not any(callable(v) for v in definition["function"].values())


# --------------------------------------------------- grounding & the trace

async def test_no_evidence_produces_a_refusal_not_a_guess():
    """With nothing retrieved and no tool data, answering would be model priors."""
    script = [FakeMessage("workflow"), FakeMessage("I know this already!")]
    async with connected() as client:
        out = await answer("unanswerable", client=client, chat_model=FakeChat(script))
    assert out["basis"] == "refusal"


async def test_trace_is_operational_and_carries_no_reasoning():
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("check_pto_balance", {"employee_id": "E-1043"})]),
        FakeMessage("done"),
    ]
    async with connected() as client:
        out = await answer("pto", client=client, chat_model=FakeChat(script))

    assert [s["step"] for s in out["trace"]] == list(range(1, len(out["trace"]) + 1))
    for step in out["trace"]:
        assert step["type"] in {"intent", "retrieval", "tool_call", "synthesis",
                                "guardrail", "escalation"}
        assert "reasoning" not in step
        assert "chain_of_thought" not in step


async def test_envelope_matches_contract_b():
    async with connected() as client:
        out = await answer("x", client=client, chat_model=FakeChat([FakeMessage("refuse")]))
    assert set(out) == {"answer", "citations", "trace", "requires_confirmation",
                        "pending_action", "escalation", "basis"}


async def test_graph_compiles_without_a_live_model():
    async with connected() as client:
        assert build_graph(client, FakeChat([FakeMessage("policy_qa")])) is not None
