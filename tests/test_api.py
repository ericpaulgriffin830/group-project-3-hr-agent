"""Tests for app/api.py -- Contract B's HTTP layer.

Same principle as test_orchestrator.py: the MCP side is real (an in-process
session against mcp_server.server.mcp, no subprocess), the chat model is a
scripted fake. CI has no Groq key, and a test whose outcome depends on what a
live model decides today is not a regression test.

`get_mcp_client_optional` / `get_chat_model` are FastAPI dependencies precisely
so these tests can override them instead of reaching into `app.state` or
triggering the real `lifespan` (which would spawn the MCP server as a
subprocess and try to build a chat model from whatever .env happens to be
present).

Two test styles on purpose:

- The two tests with NO live MCP session (`test_health_reports_mcp_and_index_
  without_lifespan`, `test_chat_without_mcp_connected_returns_503`) use plain
  `TestClient` -- a single request with nothing held open across it.
- Every test that holds an open `MCPClient` session across the request opens
  it itself, inline, with `async with _api_client(...) as client:` -- not via
  a pytest fixture. `httpx.AsyncClient` against the app directly (`ASGITransport`)
  keeps everything in one async task, avoiding the thread-hop `TestClient`
  would introduce (a `TestClient` request runs on a background thread with its
  own event loop, and handing that thread an MCP session opened elsewhere
  breaks anyio's cancel-scope accounting for it). A `@pytest.fixture async def`
  wrapping the same context manager was tried first and passed every
  assertion, but still raised a teardown-only `ExceptionGroup` -- pytest-asyncio
  appears to resume a fixture's teardown in a different task than its setup
  ran in, which is exactly what anyio's TaskGroup forbids for the session
  opened inside it. Opening and closing the session within a single test
  body -- one task, start to finish -- sidesteps that entirely.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi.testclient import TestClient
from httpx import ASGITransport

from app.agent.mcp_client import MCPClient
from app.api import app, get_chat_model, get_mcp_client_optional
from mcp_server.server import mcp


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChat:
    """Replays a script of FakeMessage; the last one repeats. See
    tests/test_orchestrator.py for the identical pattern used there.

    classify() and agent() share this same instance (bind_tools() returns
    self), so a script exercising the agentic path needs the classify-step
    response FIRST -- a bare intent word like "workflow" -- and the tool-call
    message SECOND, or classify never routes to "agent" at all and the
    tool-call message is never consumed.
    """

    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]


@asynccontextmanager
async def _connected_client():
    async with MCPClient(server=mcp) as client:
        await client.discover()
        yield client


@asynccontextmanager
async def _api_client(chat_model):
    """Real in-process MCP session + a scripted chat model, wired in via
    dependency_overrides so `lifespan` never runs, exercised through an async
    client in this same task."""
    async with _connected_client() as mcp_client:
        app.dependency_overrides[get_mcp_client_optional] = lambda: mcp_client
        app.dependency_overrides[get_chat_model] = lambda: chat_model
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                yield client
        finally:
            app.dependency_overrides.clear()


def _basic_chat_model():
    return FakeChat([FakeMessage(content="PTO accrues per PTO-2.")])


def test_health_reports_mcp_and_index_without_lifespan():
    """No dependency override, no lifespan run: /health must still answer
    (degraded, not a 500) rather than assuming app.state was populated."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"status", "mcp_connected", "tools_discovered", "index_ready"}
    assert body["status"] == "degraded"
    assert body["mcp_connected"] is False


async def test_health_ok_when_mcp_connected():
    async with _api_client(_basic_chat_model()) as client:
        resp = await client.get("/health")
    body = resp.json()
    assert body["mcp_connected"] is True
    assert body["tools_discovered"] >= 8
    assert body["status"] == "ok"


async def test_chat_envelope_matches_contract_b():
    async with _api_client(_basic_chat_model()) as client:
        resp = await client.post("/chat", json={"question": "How many holidays do we get?"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "answer", "citations", "trace", "requires_confirmation",
        "pending_action", "basis", "escalate",
    }
    assert isinstance(body["citations"], list)
    assert isinstance(body["trace"], list)
    assert isinstance(body["escalate"], bool)


def test_chat_without_mcp_connected_returns_503():
    client = TestClient(app)
    resp = client.post("/chat", json={"question": "How many holidays do we get?"})
    assert resp.status_code == 503


async def test_chat_rejects_empty_question():
    async with _api_client(_basic_chat_model()) as client:
        resp = await client.post("/chat", json={"question": ""})
    assert resp.status_code == 422


async def test_chat_confirmation_gate_short_circuits():
    """A write tool the model calls without a confirm_token must come back as
    requires_confirmation=True, per the orchestrator's confirmation gate --
    not silently executed and not a normal-looking answer.

    Script order matters here (see FakeChat's docstring): "workflow" routes
    classify() to the agent/tools path at all; only then does the tool-call
    message get a chance to run.
    """
    chat_model = FakeChat([
        FakeMessage(content="workflow"),
        FakeMessage(tool_calls=[{
            "id": "c1", "name": "create_mock_hr_ticket",
            "args": {"employee_id": "E1005", "category": "equipment",
                     "summary": "broken laptop"},
        }]),
    ])
    async with _api_client(chat_model) as client:
        resp = await client.post(
            "/chat",
            json={"question": "File a ticket about my laptop.", "employee_id": "E1005"},
        )
    body = resp.json()
    assert body["requires_confirmation"] is True
    assert body["pending_action"]["tool"] == "create_mock_hr_ticket"
