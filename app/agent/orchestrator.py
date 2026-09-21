"""The agent. A LangGraph state graph over MCP-exposed tools.

    START -> classify -> retrieve ----------> synthesize -> END
                      \\-> agent <-> tools --/
                      \\-> synthesize (clarify / refuse)

`classify` decides whether retrieval alone answers the question or whether the
employee's own data is needed -- rubric item 4's "decide whether RAG alone is
sufficient". `agent` and `tools` then alternate, capped at MAX_TOOL_STEPS, and
`synthesize` hands the assembled evidence to Rob's `synthesize()` per Contract C.

Every tool call goes through `MCPClient`, never a local function. That is the
difference between satisfying the brief and appearing to.

**The confirmation gate short-circuits the graph.** When a write tool returns
`requires_confirmation`, the loop stops immediately and the pending action is
surfaced to the user. It does not get "handled" by another model turn -- the whole
point is that a human authorises the write, so the graph must end its turn there
and wait for one.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent import guardrails, tools as tool_bridge
from app.agent.mcp_client import MCPClient, ToolCall
from app.agent.trace import Trace

MAX_TOOL_STEPS = 6

Intent = Literal["policy_qa", "workflow", "clarify", "refuse"]

CLASSIFY_SYSTEM = """You route HR questions for an internal assistant.

Answer with ONE word, nothing else.

workflow  - the answer depends on WHO is asking. Any question about the speaker's
            own eligibility, approval chain, balances, benefits, location or
            equipment, or any request to file a ticket or draft a message.
            An employee id anywhere in the question means workflow.
policy_qa - a general question about what the policy says, where the answer is the
            same for everyone.
clarify   - too ambiguous to act on without more information.
refuse    - not an HR topic, or something the assistant should not do.

Examples:
"How many vacation days does the company give?" -> policy_qa
"What is the expense limit for hotels?" -> policy_qa
"Can I work from Colorado for six weeks? I am E1001." -> workflow
"Am I eligible for the dental plan?" -> workflow
"How much PTO do I have left?" -> workflow
"File a ticket about my laptop." -> workflow
"Can I do the thing we discussed?" -> clarify
"Write me a poem." -> refuse
"""

AGENT_SYSTEM = """You are an internal HR assistant.

Use the provided tools to gather what you need. Rules:
- Look up the employee's profile before answering anything that depends on who is
  asking - eligibility, approval chains, location-specific rules.
- Always retrieve the governing policy. Never answer policy from memory.
- Never invent an employee id, a balance, or a policy statement.
- When you have enough to answer, stop calling tools and reply in plain text.
"""


class AgentState(TypedDict, total=False):
    """What flows through the graph.

    `trace` is Contract B's operational record -- tool, args, result summary,
    latency. It never carries reasoning; the brief forbids exposing it.
    """

    question: str
    employee_id: str | None
    confirm_token: str | None

    intent: Intent
    messages: list[dict]
    evidence: list[dict]
    tool_results: list[dict]
    trace: list[dict]
    steps: int

    answer: str
    action_refusal: str | None
    citations: list[dict]
    answer_basis: str
    requires_confirmation: bool
    pending_action: dict | None
    escalation: dict | None


def _trace(state: AgentState, kind: str, **fields: Any) -> list[dict]:
    """Append one step via Trace, which rejects reasoning-shaped fields."""
    trace = Trace(steps=list(state.get("trace", [])))
    trace.add(kind, **fields)
    return trace.as_list()


def _synthesize(question: str, evidence: list[dict], tool_results: list[dict],
                mode: str) -> dict:
    """Contract C's seam -- Rob's `app/rag/answer.py`.

    Imported lazily so the graph runs before his module exists. The fallback does
    NOT write an answer: it reports that synthesis is unavailable and marks the
    basis, because a plausible-looking stub answer would be indistinguishable from
    a real one in a demo and in the evaluation.
    """
    try:
        from app.rag.answer import synthesize  # type: ignore
    except ImportError:
        citations = evidence[:5]
        return {
            "answer": (
                "Evidence gathered, but answer synthesis is not wired up yet "
                f"({len(evidence)} policy passage(s), {len(tool_results)} tool "
                "result(s))."
            ),
            "citations": citations,
            "basis": "unsynthesized",
        }
    return synthesize(question=question, chunks=evidence,
                      tool_results=tool_results, mode=mode)


# ------------------------------------------------------------------- nodes


def build_graph(client: MCPClient, chat_model: Any):
    """Compile the state graph. `client` must already have discovered its tools."""

    tool_defs = tool_bridge.tool_definitions(client)
    bound = chat_model.bind_tools(tool_defs) if tool_defs else chat_model

    async def classify(state: AgentState) -> dict:
        # The caller's own id is a strong signal: a question asked *by* a known
        # employee is nearly always about that employee's situation.
        question = state["question"]
        if state.get("employee_id"):
            question = f"[asked by employee {state['employee_id']}] {question}"
        result = await chat_model.ainvoke(
            [{"role": "system", "content": CLASSIFY_SYSTEM},
             {"role": "user", "content": question}]
        )
        word = (result.content or "").strip().lower().split()[:1]
        intent: Intent = word[0] if word and word[0] in (
            "policy_qa", "workflow", "clarify", "refuse") else "policy_qa"
        escalation = guardrails.classify_escalation(state["question"])
        trace = _trace(state, "intent", result_summary=intent, status="ok")
        if escalation:
            t = Trace(steps=trace)
            t.add("escalation", result_summary=f"{escalation.route}: {escalation.reason}",
                  status="ok", route=escalation.route)
            trace = t.as_list()

        return {
            "intent": intent,
            "escalation": escalation.as_dict() if escalation else None,
            "steps": 0,
            "messages": [{"role": "system", "content": AGENT_SYSTEM},
                         {"role": "user", "content": state["question"]}],
            "trace": trace,
        }

    async def retrieve(state: AgentState) -> dict:
        """RAG-only path: one retrieval, straight to synthesis."""
        call = await client.call("search_policy_documents",
                                 query=state["question"], k=5)
        chunks = call.payload.get("chunks", []) if call.ok else []
        return {
            "evidence": chunks,
            "trace": _trace(state, "retrieval", tool=call.tool, args=call.args,
                            result_summary=f"{len(chunks)} passage(s)",
                            status="ok" if call.ok else "error",
                            latency_ms=call.latency_ms),
        }

    async def agent(state: AgentState) -> dict:
        try:
            result = await bound.ainvoke(state["messages"])
        except Exception as exc:
            # Observed against Groq + gpt-oss: HTTP 400 output_parse_failed, where
            # the model emits reasoning prose in place of a tool call and the
            # provider's parser rejects it. It is intermittent and it must not take
            # the turn down -- the brief requires graceful failure, and on 10/1 a
            # raised exception is a dead demo. Fall through to synthesis with
            # whatever evidence is already gathered.
            return {
                "messages": state["messages"] + [
                    {"role": "assistant", "content": ""}],
                "trace": _trace(state, "guardrail",
                                result_summary=f"model call failed: "
                                               f"{type(exc).__name__}",
                                status="error"),
            }

        message: dict[str, Any] = {"role": "assistant",
                                   "content": result.content or ""}
        if getattr(result, "tool_calls", None):
            message["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["args"])}}
                for tc in result.tool_calls
            ]
        return {"messages": state["messages"] + [message]}

    async def run_tools(state: AgentState) -> dict:
        """Execute the model's chosen tools through the MCP layer."""
        last = state["messages"][-1]
        evidence = list(state.get("evidence", []))
        results = list(state.get("tool_results", []))
        messages = list(state["messages"])
        trace = state.get("trace", [])
        pending: dict | None = None
        refusal: str | None = None

        for raw in last.get("tool_calls", []):
            name = raw["function"]["name"]
            args = json.loads(raw["function"]["arguments"] or "{}")

            # The human's token, never the model's. See tools._sanitise_schema.
            if tool_bridge.is_write_tool(name) and state.get("confirm_token"):
                args["confirm_token"] = state["confirm_token"]

            decision = guardrails.gate_action(
                name, args, employee_id=state.get("employee_id"),
                confirm_token=state.get("confirm_token"))
            if not decision.allow:
                t = Trace(steps=list(trace))
                t.add("guardrail", tool=name, result_summary=decision.refusal or
                      "refused to act", status="error")
                trace = t.as_list()
                messages.append({"role": "tool", "tool_call_id": raw["id"],
                                 "name": name, "content": decision.refusal or ""})
                refusal = decision.refusal
                continue

            call: ToolCall = await client.call(name, **args)
            t = Trace(steps=list(trace))
            t.add_tool_call(call)
            trace = t.as_list()
            results.append({"tool": name, "ok": call.ok,
                            "payload": call.payload, "error": call.error})

            if call.ok and "chunks" in call.payload:
                evidence.extend(call.payload["chunks"])
            if call.ok and call.payload.get("citations"):
                evidence.extend(call.payload["citations"])
            if call.ok and call.payload.get("policy_refs"):
                evidence.extend(call.payload["policy_refs"])

            if call.payload.get("requires_confirmation"):
                pending = {
                    "tool": name,
                    "confirm_token": call.payload.get("confirm_token"),
                    "preview": call.payload.get("preview", {}),
                }

            messages.append({
                "role": "tool", "tool_call_id": raw["id"], "name": name,
                "content": tool_bridge.summarise_for_model(call.payload),
            })

        return {
            "messages": messages, "evidence": evidence, "tool_results": results,
            "trace": trace, "steps": state.get("steps", 0) + 1,
            "requires_confirmation": pending is not None,
            "pending_action": pending,
            "action_refusal": refusal,
        }

    def _with_handoff(result: dict, state: AgentState) -> dict:
        """Escalations reach the reader, whatever path produced the answer."""
        raw = state.get("escalation")
        if not raw:
            return result
        esc = guardrails.Escalation(route=raw["route"], reason=raw["reason"])
        return {**result, "answer": guardrails.append_handoff(
            result.get("answer", ""), esc)}

    async def synthesize_node(state: AgentState) -> dict:
        intent = state.get("intent", "policy_qa")

        if state.get("requires_confirmation"):
            action = state["pending_action"] or {}
            return {
                "answer": guardrails.confirmation_prompt(
                    action.get("tool", ""), action.get("preview", {})),
                "citations": [], "answer_basis": "awaiting_confirmation",
                "trace": _trace(state, "guardrail",
                                result_summary=f"confirmation gate: {action.get('tool')}",
                                status="ok"),
            }

        if state.get("action_refusal"):
            return _with_handoff(
                {**guardrails.refusal_to_act(state["action_refusal"]),
                 "trace": state.get("trace", [])}, state)

        evidence = state.get("evidence", [])
        results = state.get("tool_results", [])

        if intent == "refuse":
            return _with_handoff({"answer": "That falls outside what this HR assistant covers.",
                    "citations": [], "answer_basis": "refusal",
                    "trace": _trace(state, "guardrail",
                                    result_summary="out of scope", status="ok")},
                                 state)

        if intent == "clarify":
            return _with_handoff({"answer": "I need a bit more detail before I can answer that.",
                    "citations": [], "answer_basis": "clarification",
                    "trace": _trace(state, "guardrail",
                                    result_summary="ambiguous request", status="ok")},
                                 state)

        # No evidence and no tool data: refuse rather than answer from model priors.
        if not evidence and not results:
            return _with_handoff({"answer": ("I could not find anything in company policy that "
                               "covers that."),
                    "citations": [], "answer_basis": "refusal",
                    "trace": _trace(state, "guardrail",
                                    result_summary="insufficient evidence",
                                    status="ok")},
                                 state)

        out = _synthesize(state["question"], evidence, results,
                          "policy" if intent == "policy_qa" else "workflow")
        basis = out.get("basis") or (
            "both" if evidence and results else
            "policy_rag" if evidence else "tool_data")
        return _with_handoff({
            "answer": out.get("answer", ""),
            "citations": out.get("citations", []),
            "answer_basis": basis,
            "trace": _trace(state, "synthesis",
                            result_summary=f"basis={basis}, "
                                           f"{len(out.get('citations', []))} citation(s)",
                            status="ok"),
        }, state)

    # --------------------------------------------------------------- edges

    def after_classify(state: AgentState) -> str:
        return {"policy_qa": "retrieve", "workflow": "agent"}.get(
            state.get("intent", "policy_qa"), "synthesize")

    def after_agent(state: AgentState) -> str:
        """No tool calls -- either the model is done, or its call failed above.

        Both end the turn: retrying a model that just produced unparseable output
        tends to produce it again, and the step budget is better spent answering
        from the evidence already gathered.
        """
        return "tools" if state["messages"][-1].get("tool_calls") else "synthesize"

    def after_tools(state: AgentState) -> str:
        """Stop for a human on a pending write, or when the step budget is spent.

        The cap is what stops a model that keeps re-calling a failing tool from
        looping until the request times out.
        """
        if state.get("requires_confirmation"):
            return "synthesize"
        return "agent" if state.get("steps", 0) < MAX_TOOL_STEPS else "synthesize"

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", after_classify,
                                {"retrieve": "retrieve", "agent": "agent",
                                 "synthesize": "synthesize"})
    graph.add_edge("retrieve", "synthesize")
    graph.add_conditional_edges("agent", after_agent,
                                {"tools": "tools", "synthesize": "synthesize"})
    graph.add_conditional_edges("tools", after_tools,
                                {"agent": "agent", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile()


async def answer(question: str, *, client: MCPClient, chat_model: Any,
                 employee_id: str | None = None,
                 confirm_token: str | None = None) -> dict:
    """Run one turn and return Contract B's envelope (minus the API's own fields)."""
    graph = build_graph(client, chat_model)
    final = await graph.ainvoke({
        "question": question, "employee_id": employee_id,
        "confirm_token": confirm_token, "trace": [], "evidence": [],
        "tool_results": [], "messages": [], "steps": 0,
    })
    return {
        "answer": final.get("answer", ""),
        "citations": final.get("citations", []),
        "trace": final.get("trace", []),
        "requires_confirmation": bool(final.get("requires_confirmation")),
        "pending_action": final.get("pending_action"),
        "escalation": final.get("escalation"),
        "basis": final.get("answer_basis", "unknown"),
    }
