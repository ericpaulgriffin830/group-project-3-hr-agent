"""MCP tools, described to the model in LangChain's shape.

`langchain-mcp-adapters` exists for exactly this and does not work here -- it
imports `RequestContext` from `mcp.shared.context`, which MCP 2.x removed. So this
is the bridge, hand-written, and it is the only part of the orchestration we do not
get from the framework.

Two things it deliberately does NOT do:

It does not execute anything. These are *descriptions* -- name, docstring, argument
schema -- handed to `bind_tools()` so the model can choose. Execution goes through
`MCPClient.call()` in the orchestrator's tool node, so every call crosses the real
MCP layer. A LangChain tool holding a local Python callable would run the function
directly and make the brief's "must actually call MCP-exposed tools" untrue while
looking entirely correct.

It does not hardcode a roster. Names, descriptions and schemas all come from
`list_tools()` at runtime.
"""

from __future__ import annotations

from typing import Any

from app.agent.mcp_client import MCPClient, ToolSpec


def _sanitise_schema(schema: dict) -> dict:
    """Make an MCP input schema safe to hand to bind_tools.

    MCP schemas carry a `title` that providers sometimes echo oddly, and we drop
    `confirm_token` from what the MODEL sees: the token is issued by the server
    after a preview and supplied by the human through the UI. A model that believes
    it can pass one would try to invent it, and the gate exists precisely so a
    human -- not the model -- authorises the write.
    """
    cleaned = {k: v for k, v in schema.items() if k != "title"}
    props = {k: v for k, v in cleaned.get("properties", {}).items()
             if k != "confirm_token"}
    cleaned["properties"] = props
    cleaned["required"] = [r for r in cleaned.get("required", []) if r in props]
    return cleaned


def spec_to_openai_tool(spec: ToolSpec) -> dict:
    """One tool in the function-calling shape bind_tools accepts."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": _sanitise_schema(spec.input_schema),
        },
    }


def tool_definitions(client: MCPClient) -> list[dict]:
    """Every discovered tool, described for the model.

    Ordered by name so the prompt is byte-stable across runs -- dict ordering drift
    would change the prompt, and a changed prompt makes two evaluation runs
    incomparable for reasons nothing in the results would explain.
    """
    return [spec_to_openai_tool(client.tools[name]) for name in client.tool_names()]


WRITE_TOOLS = frozenset({"create_mock_hr_ticket", "draft_hr_email"})


def is_write_tool(name: str) -> bool:
    """Tools that perform an action and therefore need human confirmation.

    Kept here rather than inferred from the name so adding a write tool is a
    deliberate edit, not something that depends on what someone called it.
    """
    return name in WRITE_TOOLS


def summarise_for_model(payload: dict[str, Any], limit: int = 900) -> str:
    """Render a tool result back into the conversation.

    Truncated: a full policy chunk list can run to thousands of tokens and the
    model needs the facts, not the transcript. The untruncated payload stays in the
    trace and in the evidence the answer is built from.
    """
    import json

    text = json.dumps(payload, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."
