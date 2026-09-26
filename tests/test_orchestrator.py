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


#: Tests that script "workflow" must pass an employee_id. A workflow with no
#: subject is downgraded to policy_qa on purpose -- see
#: test_workflow_without_an_employee_id_falls_back_to_policy_qa -- so omitting it
#: silently routes the test down the RAG-only path and no tool loop runs.
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
        FakeMessage(tool_calls=[tool_call("check_pto_balance", {"employee_id": "E1001"})]),
        FakeMessage("You have 6.5 days available."),
    ]
    async with connected() as client:
        out = await answer("how much pto", client=client, chat_model=FakeChat(script),
                           employee_id="E1001")

    tool_steps = [s for s in out["trace"] if s["type"] == "tool_call"]
    assert [s["tool"] for s in tool_steps] == ["check_pto_balance"]
    assert tool_steps[0]["status"] == "ok"


async def test_tool_loop_is_capped():
    """A model that keeps calling the same tool must not loop until timeout."""
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("check_pto_balance", {"employee_id": "E1001"})]),
    ]
    async with connected() as client:
        out = await answer("loop forever", client=client, chat_model=FakeChat(script),
                           employee_id="E1001")

    assert len([s for s in out["trace"] if s["type"] == "tool_call"]) == MAX_TOOL_STEPS


async def test_a_failing_tool_does_not_stop_the_graph():
    """Graceful failure is a rubric item. The error is traced and the turn finishes."""
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("lookup_employee_profile",
                                          {"employee_id": "E9999"})]),
        FakeMessage("I could not find that employee."),
    ]
    async with connected() as client:
        out = await answer("who is E9999", client=client, chat_model=FakeChat(script),
                           employee_id="E1001")

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
            "employee_id": "E1001", "category": "equipment",
            "summary": "Laptop replacement"})]),
        FakeMessage("should never be reached"),
    ]
    async with connected() as client:
        out = await answer("file a ticket", client=client, chat_model=FakeChat(script),
                           employee_id="E1001")

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
        FakeMessage(tool_calls=[tool_call("check_pto_balance", {"employee_id": "E1001"})]),
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


async def test_a_model_failure_does_not_take_the_turn_down():
    """Observed live: Groq 400 output_parse_failed on gpt-oss.

    The model emits reasoning prose where a tool call belongs and the provider
    rejects it. Intermittent, and on 10/1 an unhandled raise is a dead demo.
    """
    class Exploding(FakeChat):
        async def ainvoke(self, messages):
            if any(m.get("role") == "system" and "route HR questions" in m["content"]
                   for m in messages):
                return FakeMessage("workflow")
            raise RuntimeError("Error code: 400 - output_parse_failed")

    async with connected() as client:
        out = await answer("anything", client=client,
                           chat_model=Exploding([FakeMessage("workflow")]),
                           employee_id="E1001")

    assert out["answer"]
    assert any(s["type"] == "guardrail" and s.get("status") == "error"
               for s in out["trace"])


# ------------------------------------------------------------- escalation

async def test_escalation_reaches_the_answer_not_just_the_envelope():
    """An escalation the user never reads is metadata, not behaviour."""
    async with connected() as client:
        out = await answer("My manager has been harassing me.", client=client,
                           chat_model=FakeChat([FakeMessage("policy_qa")]),
                           employee_id="E1001")

    assert out["escalation"]["route"] == "hr_partner"
    assert "HR business partner" in out["answer"]
    assert any(s["type"] == "escalation" for s in out["trace"])


async def test_ordinary_question_carries_no_handoff():
    async with connected() as client:
        out = await answer("How many vacation days do I get?", client=client,
                           chat_model=FakeChat([FakeMessage("policy_qa")]))
    assert out["escalation"] is None
    assert "HR business partner" not in out["answer"]


async def test_acting_on_another_employee_is_refused_in_the_graph():
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("create_mock_hr_ticket", {
            "employee_id": "E1006", "category": "equipment", "summary": "laptop"})]),
        FakeMessage("done"),
    ]
    async with connected() as client:
        out = await answer("file a ticket for E1006", client=client,
                           chat_model=FakeChat(script), employee_id="E1001")

    assert out["basis"] == "refusal"
    assert "E1006" in out["answer"]
    assert out["requires_confirmation"] is False


async def test_confirm_token_never_appears_in_the_rendered_trace():
    """The trace is rendered in a browser and pasted into bug reports."""
    script = [
        FakeMessage("workflow"),
        FakeMessage(tool_calls=[tool_call("create_mock_hr_ticket", {
            "employee_id": "E1001", "category": "equipment", "summary": "laptop"})]),
    ]
    async with connected() as client:
        out = await answer("file it", client=client, chat_model=FakeChat(script),
                           employee_id="E1001", confirm_token="cf_supersecret")
    assert "cf_supersecret" not in str(out["trace"])


# --------------------------------------------------- citation selection

from app.agent.orchestrator import _diverse_citations  # noqa: E402


def _chunk(doc, score, section="S-1"):
    return {"doc_id": doc, "title": doc, "section": section,
            "snippet": "...", "score": score}


def test_citations_are_chosen_by_score_first():
    evidence = [_chunk("A", 0.9), _chunk("B", 0.8), _chunk("A", 0.7)]
    got = _diverse_citations(evidence, limit=2)
    assert [c["score"] for c in got] == [0.9, 0.8]


def test_a_second_document_is_admitted_when_it_earns_it():
    """Rubric item 3 needs a multi-document answer to actually cite two documents.

    All five top chunks coming from one document would cite one source even when
    a second policy genuinely bears on the answer.
    """
    evidence = [_chunk("REMOTE-WORK", 0.9 - i / 100) for i in range(5)]
    evidence.append(_chunk("TAX-LOCATION", 0.6))
    got = _diverse_citations(evidence, limit=5)
    assert {c["doc_id"] for c in got} == {"REMOTE-WORK", "TAX-LOCATION"}


def test_a_weak_second_document_is_not_admitted():
    """Precision beats spread.

    An earlier version round-robined across every document present, which made a
    PTO question cite REMOTE-WORK and TAX-LOCATION. Citation accuracy scores
    whether the citation is right, not how varied it is.
    """
    evidence = [_chunk("PTO-HOLIDAYS", 0.9 - i / 100) for i in range(5)]
    evidence.append(_chunk("REMOTE-WORK", 0.05))
    got = _diverse_citations(evidence, limit=5)
    assert {c["doc_id"] for c in got} == {"PTO-HOLIDAYS"}


def test_single_document_evidence_stays_single_document():
    evidence = [_chunk("PTO-HOLIDAYS", 0.9), _chunk("PTO-HOLIDAYS", 0.8)]
    assert {c["doc_id"] for c in _diverse_citations(evidence, limit=5)} == {"PTO-HOLIDAYS"}


def test_no_evidence_yields_no_citations():
    assert _diverse_citations([], limit=5) == []


async def test_a_model_failure_in_classify_does_not_take_the_turn_down():
    """Found live: Groq 429 when the free tier's daily token limit ran out.

    The agent node already survived model failures; classify did not, so the
    whole turn raised. On 10/1 that is a dead demo, and a rate limit is exactly
    the thing most likely to happen during a recording.
    """
    class Exploding:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")

    async with connected() as client:
        out = await answer("How many vacation days do I get?", client=client,
                           chat_model=Exploding())
    assert out["answer"]
    assert out["trace"][0]["result_summary"] == "policy_qa"


async def test_workflow_without_an_employee_id_falls_back_to_policy_qa():
    """A workflow needs someone to run it for.

    First-person phrasing ("am I eligible for parental leave?") routes to workflow
    on wording alone. With no employee_id there is no record to consult, so the
    agent spends its budget on lookups it cannot make and retrieves less policy
    than the RAG-only path would have. Three of Eric's policy items lost their
    expected citations exactly this way.
    """
    async with connected() as client:
        out = await answer("Am I eligible for parental leave?", client=client,
                           chat_model=FakeChat([FakeMessage("workflow")]))
    assert out["trace"][0]["result_summary"] == "policy_qa"


async def test_workflow_with_an_employee_id_is_left_alone():
    async with connected() as client:
        out = await answer("How much PTO do I have?", client=client,
                           chat_model=FakeChat([FakeMessage("workflow")]),
                           employee_id="E1001")
    assert out["trace"][0]["result_summary"] == "workflow"
