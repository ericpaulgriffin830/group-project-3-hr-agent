"""The web application. Contract B: `POST /chat`, `GET /health`.

Owner: Eric. `docs/CONTRACTS.md` freezes the response envelope; this module's
job is to be a thin HTTP skin over Chris's orchestrator (`app/agent/orchestrator.py`)
and nothing else -- no agent logic lives here.

The MCP client and chat model are constructed ONCE at startup and held for the
life of the process (`app.state`), not per request. `MCP_TRANSPORT=stdio`
launches the MCP server as a subprocess; doing that on every request would be
slow, and it would defeat `/health`'s `tools_discovered` count, which is only
meaningful for a client that has already run `discover()`.

The app still starts without a configured Groq key -- CI, `uv run python -c
"import app.api"`, and `/health` all have to work before a key exists. Only
`POST /chat` needs one, and it fails with a clear 503 rather than the whole
process refusing to boot over a missing `.env`. This mirrors `app/llm.py`'s own
`LLMNotConfigured` split between "not configured" and "failed".
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app import llm
from app.agent.mcp_client import MCPClient, MCPUnavailable
from app.agent.orchestrator import answer as run_agent

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("app.api")


def _index_ready() -> bool:
    """Best-effort check that Rob's vector index has been built.

    Imported lazily, exactly like `mcp_server/server.py`'s own retrieval
    import: building or opening the index costs real time and, on a cold
    Chroma path, may not exist yet. An unbuilt index is a normal pre-deploy
    state, not a crash, so any failure here just reports "not ready".
    """
    try:
        from app.rag.store import count
        return count() > 0
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mcp_client = None
    app.state.chat_model = None
    app.state.startup_error = None

    client = MCPClient()
    try:
        await client.__aenter__()
        await client.discover()
        app.state.mcp_client = client
        logger.info("MCP client connected: %d tool(s) discovered",
                    len(client.tools))
    except MCPUnavailable as exc:
        # Logged, not raised -- a Render cold start where the MCP subprocess is
        # briefly unavailable should show up as a degraded /health, not a dead
        # process that never accepts a request at all.
        logger.error("MCP server unavailable at startup: %s", exc)
        app.state.startup_error = str(exc)

    try:
        app.state.chat_model = llm.chat_model()
    except llm.LLMNotConfigured as exc:
        logger.warning("Starting without a configured Groq key: %s", exc)

    yield

    if app.state.mcp_client is not None:
        await app.state.mcp_client.__aexit__(None, None, None)


app = FastAPI(title="HR Agent API", lifespan=lifespan)


# --------------------------------------------------------------- dependencies


def get_mcp_client_optional(request: Request) -> MCPClient | None:
    """The raw lookup, never raising. Backs /health, which has to report a
    disconnected server rather than 503 on it -- and backs get_mcp_client
    below, so overriding this ONE dependency in a test affects both routes.
    """
    return getattr(request.app.state, "mcp_client", None)


def get_mcp_client(
    request: Request,
    client: MCPClient | None = Depends(get_mcp_client_optional),
) -> MCPClient:
    """Depends() hook so tests can override this instead of poking app.state."""
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(request.app.state, "startup_error", None)
            or "MCP server is not connected.",
        )
    return client


def get_chat_model(request: Request) -> Any:
    """Built at startup when a key is present; retried here otherwise so a key
    added to a running process (e.g. Render env var updated without a redeploy)
    is picked up on the next request rather than requiring a restart.
    """
    model = getattr(request.app.state, "chat_model", None)
    if model is not None:
        return model
    try:
        model = llm.chat_model()
    except llm.LLMNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    request.app.state.chat_model = model
    return model


# --------------------------------------------------------------------- schema


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    employee_id: str | None = None
    confirm_token: str | None = None


class ChatResponse(BaseModel):
    """Contract B's envelope, exactly as `docs/CONTRACTS.md` specifies it."""

    answer: str
    citations: list[dict]
    trace: list[dict]
    requires_confirmation: bool
    pending_action: dict | None
    basis: str
    escalate: bool


# -------------------------------------------------------------------- routes


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    client: MCPClient = Depends(get_mcp_client),
    chat_model: Any = Depends(get_chat_model),
) -> ChatResponse:
    result = await run_agent(
        req.question,
        client=client,
        chat_model=chat_model,
        employee_id=req.employee_id,
        confirm_token=req.confirm_token,
    )
    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        trace=result["trace"],
        requires_confirmation=result["requires_confirmation"],
        pending_action=result["pending_action"],
        basis=result["basis"],
        # Contract B's envelope carries a boolean flag; the orchestrator's
        # {"route", "reason"} detail is already folded into `answer` by
        # `guardrails.append_handoff`, so nothing is lost by flattening it here.
        escalate=bool(result.get("escalation")),
    )


@app.get("/health")
async def health(
    client: MCPClient | None = Depends(get_mcp_client_optional),
) -> dict:
    mcp_health = client.health() if client is not None else {
        "mcp_connected": False,
        "tools_discovered": 0,
    }
    ready = _index_ready()
    return {
        "status": "ok" if mcp_health["mcp_connected"] else "degraded",
        "mcp_connected": mcp_health["mcp_connected"],
        "tools_discovered": mcp_health["tools_discovered"],
        "index_ready": ready,
    }
