from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer

from ant_ai.tools.tool import Tool, ToolError, mcp_tools_from_url

mcp: MCPServer[Any] = MCPServer("remote")


@mcp.tool(name="greet", description="Greet someone")
def greet(name: str) -> str:
    return f"Hello, {name}!"


@mcp.tool(name="add", description="Add two numbers")
def add(a: int, b: int) -> str:
    s = a + b
    return f"Sum: {s}"


@mcp.tool(name="fail", description="Always fails")
def fail() -> str:
    raise ValueError("nothing to see here")


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
    assert names == {"greet", "add", "fail"}


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


@pytest.mark.unit
async def test_mcp_is_error_result_raises_tool_error(mcp_server):
    """A server-side failure comes back as `CallToolResult(is_error=True)`; it
    must surface as a ToolError rather than be unwrapped as a normal result.
    (The server masks the original exception text by default.)"""
    tools: list[Tool] = await mcp_tools_from_url(mcp_server)
    fail = next(t for t in tools if t.name == "fail")
    with pytest.raises(ToolError, match="Error executing tool fail"):
        await fail()
