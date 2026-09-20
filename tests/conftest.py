"""Shared helpers for the MCP tool tests.

The agent only ever reaches tools through the MCP layer, so the tests do too --
they call `mcp.call_tool(name, args)` rather than importing the Python functions.
A test that imported the function directly would pass even if the tool were never
registered, which is exactly the failure the brief cares about.
"""

from __future__ import annotations

import json

import pytest

from mcp_server.server import mcp


def unwrap(result) -> dict:
    """Pull the tool's dict payload out of a CallToolResult.

    MCP 2.x returns structured_content when the tool declares a dict return, and
    falls back to a JSON text block otherwise. Handle both so these tests do not
    break on an SDK detail.
    """
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured
    return json.loads(result.content[0].text)


@pytest.fixture
def call():
    async def _call(name: str, **args):
        return unwrap(await mcp.call_tool(name, args))

    return _call
