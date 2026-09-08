from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections.abc import Iterator
from unittest.mock import MagicMock

import acp
import pytest
import uvicorn
import websockets
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

from ant_ai.acp.adapter import ACPAdapter
from ant_ai.acp.server import ACPServer, build_acp_ws_route
from ant_ai.core.events import FinalAnswerEvent


def _make_server() -> ACPServer:
    agent = MagicMock()
    agent.name = "TestAgent"
    workflow = MagicMock()
    return ACPServer(agent=agent, workflow=workflow)


def test_build_acp_ws_route_path():
    agent = MagicMock()
    agent.name = "TestAgent"
    workflow = MagicMock()
    route: WebSocketRoute = build_acp_ws_route(agent, workflow)
    assert isinstance(route, WebSocketRoute)
    assert route.path == "/acp/ws"


def test_acp_server_starlette_app_returns_asgi():
    server = _make_server()
    app = server.starlette_app()
    # Starlette app is callable (ASGI interface)
    assert callable(app)


def test_acp_server_fastapi_app_returns_asgi():
    server = _make_server()
    app = server.fastapi_app()
    assert callable(app)


def test_acp_server_routes_contain_ws_endpoint():
    server = _make_server()
    app = server.starlette_app()
    paths = [r.path for r in app.routes]
    assert "/acp/ws" in paths


# ---------------------------------------------------------------------------
# WebSocket bridge
#
# These run against a real uvicorn server rather than starlette's TestClient:
# TestClient cancels its portal task on exit, which surfaces as a spurious
# CancelledError once the machine is slow enough.
# ---------------------------------------------------------------------------


def _ws_app() -> Starlette:
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.tools = []
    workflow = MagicMock()

    async def _stream(**_):
        yield FinalAnswerEvent(content="hi there")

    workflow.stream = _stream
    workflow.create_state = MagicMock(return_value=MagicMock())
    return Starlette(routes=[build_acp_ws_route(agent, workflow)])


@pytest.fixture
def acp_ws_url() -> Iterator[str]:
    """Serve the ACP route on an ephemeral port and yield its WebSocket URL."""
    server = uvicorn.Server(
        uvicorn.Config(_ws_app(), host="127.0.0.1", port=0, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            server.should_exit = True
            pytest.fail("uvicorn did not start within 10s")
        time.sleep(0.01)

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"ws://127.0.0.1:{port}/acp/ws"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def _rpc(ws, request_id: int, method: str, params: dict) -> None:
    await ws.send(
        json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
    )


async def _await_response(ws, request_id: int) -> tuple[dict, list[dict]]:
    """Read until the response for request_id arrives; return it plus notifications."""
    notifications: list[dict] = []
    while True:
        message = json.loads(await ws.recv())
        if message.get("id") == request_id:
            return message, notifications
        notifications.append(message)


async def _open_session(ws) -> str:
    await _rpc(ws, 1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    await _await_response(ws, 1)
    await _rpc(ws, 2, "session/new", {"cwd": "/tmp", "mcpServers": []})
    response, _ = await _await_response(ws, 2)
    return response["result"]["sessionId"]


async def test_ws_route_serves_initialize_and_new_session(acp_ws_url):
    async with websockets.connect(acp_ws_url) as ws:
        await _rpc(
            ws, 1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}}
        )
        init, _ = await _await_response(ws, 1)
        assert init["result"]["agentInfo"]["name"] == "TestAgent"

        await _rpc(ws, 2, "session/new", {"cwd": "/tmp", "mcpServers": []})
        new_session, _ = await _await_response(ws, 2)
        assert new_session["result"]["sessionId"]


async def test_ws_route_streams_prompt_updates_before_response(acp_ws_url):
    async with websockets.connect(acp_ws_url) as ws:
        session_id = await _open_session(ws)
        await _rpc(
            ws,
            3,
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]},
        )
        response, notifications = await _await_response(ws, 3)

    assert response["result"]["stopReason"] == "end_turn"
    updates = [n["params"]["update"] for n in notifications]
    assert {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": "hi there"},
    } in updates


async def test_ws_route_closes_when_the_socket_bridge_cannot_start(
    acp_ws_url, monkeypatch
):
    async def _boom(**_):
        raise OSError("no file descriptors available")

    monkeypatch.setattr(asyncio, "open_connection", _boom)

    async with websockets.connect(acp_ws_url) as ws:
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await ws.recv()


# ---------------------------------------------------------------------------
# serve / serve_stdio
# ---------------------------------------------------------------------------


def test_serve_runs_uvicorn_on_configured_host_and_port(monkeypatch):
    server = _make_server()
    server.host = "0.0.0.0"
    server.port = 9123
    calls = {}

    def _fake_run(app, host, port):
        calls.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    server.serve()

    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 9123
    assert "/acp/ws" in [r.path for r in calls["app"].routes]


def test_serve_raises_when_uvicorn_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(ImportError, match="Uvicorn is not installed"):
        _make_server().serve()


def test_serve_returns_cleanly_on_keyboard_interrupt(monkeypatch):
    def _interrupt(*_, **__):
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn, "run", _interrupt)
    _make_server().serve()  # swallowed, not re-raised


def test_serve_wraps_startup_failure_in_runtime_error(monkeypatch):
    def _boom(*_, **__):
        raise OSError("port already in use")

    monkeypatch.setattr(uvicorn, "run", _boom)
    with pytest.raises(RuntimeError, match="port already in use"):
        _make_server().serve()


def test_serve_stdio_runs_adapter_for_the_agent(monkeypatch):
    seen = {}

    async def _fake_run_agent(adapter, **kwargs):
        seen["adapter"] = adapter
        seen["kwargs"] = kwargs

    monkeypatch.setattr(acp, "run_agent", _fake_run_agent)
    _make_server().serve_stdio()

    assert isinstance(seen["adapter"], ACPAdapter)
    assert seen["kwargs"]["use_unstable_protocol"] is True
