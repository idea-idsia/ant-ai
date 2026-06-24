from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ant_ai.tools.tool import mcp_tools_from_url


def _make_mcp_session(tool_names: list[str]):
    """Build a mock MCPClientSession that returns the given tool names on list_tools."""
    import mcp.types as mt

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.initialize = AsyncMock()

    tools = [
        mt.Tool(name=n, description=f"{n} description", inputSchema={"type": "object"})
        for n in tool_names
    ]
    session.list_tools = AsyncMock(return_value=MagicMock(tools=tools))
    return session


def _make_transport_ctx(read=None, write=None):
    """Return an async context manager yielding (read, write, ...)."""
    _read = read if read is not None else MagicMock()
    _write = write if write is not None else MagicMock()

    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        yield (_read, _write, MagicMock())

    return _ctx


@pytest.mark.asyncio
async def test_http_transport_uses_streamable_http_client():
    fake_session = _make_mcp_session(["ping"])

    with (
        patch(
            "ant_ai.tools.tool.streamable_http_client",
            side_effect=_make_transport_ctx(),
        ) as mock_http,
        patch("ant_ai.tools.tool.sse_client") as mock_sse,
        patch("ant_ai.tools.tool.MCPClientSession", return_value=fake_session),
    ):
        tools = await mcp_tools_from_url("http://localhost:8000/mcp")

    mock_http.assert_called_once_with("http://localhost:8000/mcp")
    mock_sse.assert_not_called()
    assert len(tools) == 1
    assert tools[0].name == "ping"


@pytest.mark.asyncio
async def test_sse_transport_uses_sse_client():
    fake_session = _make_mcp_session(["search"])

    with (
        patch(
            "ant_ai.tools.tool.sse_client", side_effect=_make_transport_ctx()
        ) as mock_sse,
        patch("ant_ai.tools.tool.streamable_http_client") as mock_http,
        patch("ant_ai.tools.tool.MCPClientSession", return_value=fake_session),
    ):
        tools = await mcp_tools_from_url(
            "http://localhost:8000/sse",
            transport="sse",
            headers={"Authorization": "Bearer token"},
        )

    mock_sse.assert_called_once_with(
        "http://localhost:8000/sse", headers={"Authorization": "Bearer token"}
    )
    mock_http.assert_not_called()
    assert len(tools) == 1
    assert tools[0].name == "search"


@pytest.mark.asyncio
async def test_default_transport_is_http_backward_compatible():
    """Calling mcp_tools_from_url without transport= still uses HTTP."""
    fake_session = _make_mcp_session([])

    with (
        patch(
            "ant_ai.tools.tool.streamable_http_client",
            side_effect=_make_transport_ctx(),
        ) as mock_http,
        patch("ant_ai.tools.tool.sse_client") as mock_sse,
        patch("ant_ai.tools.tool.MCPClientSession", return_value=fake_session),
    ):
        await mcp_tools_from_url("http://localhost:9000/mcp")

    mock_http.assert_called_once()
    mock_sse.assert_not_called()
