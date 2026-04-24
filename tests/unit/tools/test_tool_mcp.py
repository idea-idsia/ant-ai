import os
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from ant_ai.tools.tool import Tool, mcp_tools_from_url

_PORT = int(os.environ.get("MCP_TEST_PORT", "43654"))

mcp: FastMCP[Any] = FastMCP("remote", host="127.0.0.1", port=_PORT)


@mcp.tool(name="greet", description="Greet someone")
def greet(name: str) -> str:
    return f"Hello, {name}!"


@mcp.tool(name="add", description="Add two numbers")
def add(a: int, b: int) -> str:
    s = a + b
    return f"Sum: {s}"


@pytest.mark.unit
async def test_hello_tool(mcp_server):
    tools: list[Tool] = await mcp_tools_from_url(mcp_server)
    hello = next(t for t in tools if t.name == "greet")
    result = await hello(name="Alice")
    assert "Hello, Alice!" in str(result)


@pytest.mark.unit
async def test_add_tool(mcp_server):
    tools: list[Tool] = await mcp_tools_from_url(mcp_server)
    add = next(t for t in tools if t.name == "add")
    result = await add(a=5, b=7)
    assert "Sum: 12" in str(result)


@pytest.mark.unit
async def test_discovers_all_tools(mcp_server):
    """Verify the correct number of tools is returned."""
    tools: list[Tool] = await mcp_tools_from_url(mcp_server)
    names = {t.name for t in tools}
    assert names == {"greet", "add"}


@pytest.mark.unit
async def test_namespace_prefixes_tool_names(mcp_server):
    """Verify namespace is prepended to every tool name."""
    tools: list[Tool] = await mcp_tools_from_url(mcp_server, namespace="remote")
    for tool in tools:
        assert tool.name.startswith("remote"), f"{tool.name!r} missing namespace"


@pytest.mark.unit
async def test_invalid_url_raises(mcp_server):
    """Connecting to a non-existent server should raise."""
    with pytest.raises(ExceptionGroup):
        await mcp_tools_from_url("http://127.0.0.1:1/mcp")
