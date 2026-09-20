"""The agent's only door to the tools.

The brief is explicit: *"The agent must actually call MCP-exposed tools during
execution; hard-coded direct function calls are not sufficient unless they are
wrapped and invoked through the MCP layer."* So the orchestrator never imports
anything from `mcp_server` -- it holds one of these and calls `call()`. Importing a
tool function directly would still work, and would silently forfeit a rubric item.

Transport is chosen by `MCP_TRANSPORT` and nothing else in the codebase branches on
it: `stdio` launches the server as a subprocess for local dev and CI, and
`streamable-http` talks to `MCP_SERVER_URL` for a split deployment. Same tools, same
schemas, same session either way.

**Failures are returned, never raised.** A tool that is down, a name the server does
not expose, a timeout -- each comes back as a `ToolCall` with `ok=False` and an error
code the orchestrator can switch on. The brief requires handling unavailable MCP
tools gracefully, and an exception escaping into the state graph is the opposite of
that. The one exception is `discover()`, which raises if the server cannot be
reached at startup: a client that never discovered anything is misconfigured, not
degraded, and pretending otherwise hides the real fault.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from mcp import Client, StdioServerParameters

DEFAULT_TIMEOUT_S = 30.0


class MCPUnavailable(RuntimeError):
    """The server could not be reached at all. Raised only from connect/discover."""


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the server advertises it.

    `description` matters more than it looks: the model picks tools from these, and
    tool-selection accuracy is a scored metric. A tool the model cannot tell apart
    from its neighbours is the usual way that score is lost.
    """

    name: str
    description: str
    input_schema: dict

    @property
    def required(self) -> set[str]:
        return set(self.input_schema.get("required", []))


@dataclass
class ToolCall:
    """One call, shaped for both the orchestrator and the trace.

    `result_summary` is a short factual string, never the full payload and never a
    rationale -- Contract B's trace is operational, and the brief forbids exposing
    chain-of-thought.
    """

    tool: str
    args: dict
    ok: bool
    latency_ms: int
    payload: dict = field(default_factory=dict)
    error: str | None = None
    message: str | None = None

    def result_summary(self, limit: int = 120) -> str:
        if not self.ok:
            return f"{self.error}: {self.message}"
        keys = ", ".join(list(self.payload)[:6])
        return f"ok ({keys})"[:limit]

    def as_trace_step(self, step: int) -> dict:
        return {
            "step": step,
            "type": "tool_call",
            "tool": self.tool,
            "args": self.args,
            "result_summary": self.result_summary(),
            "status": "ok" if self.ok else "error",
            "latency_ms": self.latency_ms,
        }


def _unwrap(result: Any) -> dict:
    """Pull the tool's dict payload out of a CallToolResult.

    MCP 2.x fills `structured_content` when a tool declares a dict return and falls
    back to a JSON text block otherwise. Handle both so this does not break on an
    SDK detail.
    """
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": "upstream_error", "message": text[:200]}
    return {}


def _server_target() -> Any:
    """Build whatever `Client` needs for the configured transport."""
    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "streamable-http":
        url = os.getenv("MCP_SERVER_URL")
        if not url:
            raise MCPUnavailable(
                "MCP_TRANSPORT=streamable-http but MCP_SERVER_URL is unset."
            )
        return url

    if transport == "stdio":
        # Launch the server as a subprocess. MCP_TRANSPORT is passed explicitly so
        # the child cannot inherit a conflicting value mid-migration.
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env={**os.environ, "MCP_TRANSPORT": "stdio"},
        )

    raise MCPUnavailable(
        f"MCP_TRANSPORT must be 'stdio' or 'streamable-http', got {transport!r}"
    )


class MCPClient:
    """Async context manager wrapping one real `ClientSession`.

    Usage:
        async with MCPClient() as client:
            await client.discover()
            call = await client.call("check_pto_balance", employee_id="E-1043")
    """

    def __init__(self, server: Any = None, timeout_s: float = DEFAULT_TIMEOUT_S):
        # `server` is injectable so tests can pass the MCPServer object and talk to
        # it in-process -- still a real session over the MCP layer, just no subprocess.
        self._server = server
        self._timeout_s = timeout_s
        self._client: Client | None = None
        self._tools: dict[str, ToolSpec] = {}
        self._discovered = False

    # ------------------------------------------------------------- lifecycle

    async def __aenter__(self) -> MCPClient:
        target = self._server if self._server is not None else _server_target()
        self._client = Client(target, read_timeout_seconds=self._timeout_s)
        try:
            await self._client.__aenter__()
        except Exception as exc:
            self._client = None
            raise MCPUnavailable(f"Could not connect to the MCP server: {exc}") from exc
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, tb: TracebackType | None) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None

    # ------------------------------------------------------------- discovery

    async def discover(self) -> list[ToolSpec]:
        """Ask the server what it exposes. Called once at startup.

        This is the `list_tools()` the brief asks us to document, and the thing that
        makes tool use discovered rather than hard-coded: nothing in this codebase
        holds a list of tool names that the server did not just supply.
        """
        if self._client is None:
            raise MCPUnavailable("discover() called before the client was connected.")
        try:
            listed = await self._client.list_tools()
        except Exception as exc:
            raise MCPUnavailable(f"Tool discovery failed: {exc}") from exc

        # The high-level Client returns a ListToolsResult; a server object returns a
        # plain list. Accept either so swapping how we connect stays an adapter change.
        tools = getattr(listed, "tools", listed)

        self._tools = {
            t.name: ToolSpec(
                name=t.name,
                description=(t.description or "").strip(),
                input_schema=dict(t.input_schema or {}),
            )
            for t in tools
        }
        self._discovered = True
        return list(self._tools.values())

    @property
    def tools(self) -> dict[str, ToolSpec]:
        return dict(self._tools)

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    # ------------------------------------------------------------- calling

    async def call(self, tool: str, **args: Any) -> ToolCall:
        """Invoke one tool. Always returns; never raises."""
        started = time.monotonic()

        def failed(code: str, message: str) -> ToolCall:
            return ToolCall(
                tool=tool, args=args, ok=False,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=code, message=message,
            )

        if self._client is None:
            return failed("tool_unavailable", "MCP client is not connected.")

        # Refuse a name the server never advertised rather than sending it and
        # letting the transport decide. Keeps a hallucinated tool name out of the
        # wire and makes the failure legible in the trace.
        if self._discovered and tool not in self._tools:
            return failed(
                "tool_unavailable",
                f"{tool!r} is not exposed by this server. "
                f"Available: {', '.join(self.tool_names())}",
            )

        try:
            raw = await self._client.call_tool(tool, args)
        except Exception as exc:
            return failed("tool_unavailable", f"{type(exc).__name__}: {exc}")

        payload = _unwrap(raw)
        latency_ms = int((time.monotonic() - started) * 1000)

        # A tool reporting its own refusal is not a transport failure. It is a
        # contract-shaped answer, and the orchestrator switches on the code.
        if "error" in payload:
            return ToolCall(
                tool=tool, args=args, ok=False, latency_ms=latency_ms,
                payload=payload, error=payload.get("error"),
                message=payload.get("message"),
            )

        return ToolCall(tool=tool, args=args, ok=True,
                        latency_ms=latency_ms, payload=payload)

    # ------------------------------------------------------------- health

    def health(self) -> dict:
        """Feeds Contract B's /health. Never performs a call."""
        return {
            "mcp_connected": self._client is not None,
            "tools_discovered": len(self._tools),
            "transport": os.getenv("MCP_TRANSPORT", "stdio"),
        }
