"""Standalone CI smoke test: MCP tool discovery and a real tool call.

Kept as its own script rather than an inline `run: >` block in ci.yml, because a
YAML folded scalar collapses this function's newlines into spaces and would
silently destroy the indentation Python needs to parse it.

Runs entirely in-process against the MCP server object (same pattern as
tests/conftest.py's `call` fixture) -- no subprocess, no network, no API keys,
so it cannot flake in CI.
"""

from __future__ import annotations

import asyncio
import json

from mcp_server.server import mcp


async def main() -> None:
    tools = await mcp.list_tools()
    names = sorted(t.name for t in tools)
    assert len(names) == 8, f"expected 8 tools, got {len(names)}: {names}"

    result = await mcp.call_tool("lookup_employee_profile", {"employee_id": "E1007"})
    structured = getattr(result, "structured_content", None)
    payload = structured if structured else json.loads(result.content[0].text)
    assert "error" not in payload, payload

    print(f"MCP: discovered {len(names)} tools, lookup_employee_profile call succeeded for E1007")


if __name__ == "__main__":
    asyncio.run(main())
