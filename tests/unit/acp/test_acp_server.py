from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import MagicMock

import acp
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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


def _rpc(ws, request_id: int, method: str, params: dict) -> None:
    ws.send_text(
        json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
    )


def _await_response(ws, request_id: int) -> tuple[dict, list[dict]]:
    """Read until the response for request_id arrives; return it plus notifications."""
    notifications: list[dict] = []
    while True:
        message = json.loads(ws.receive_text())
        if message.get("id") == request_id:
            return message, notifications
        notifications.append(message)


def test_ws_route_serves_initialize_and_new_session():
    with TestClient(_ws_app()).websocket_connect("/acp/ws") as ws:
        _rpc(ws, 1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        init, _ = _await_response(ws, 1)
        assert init["result"]["agentInfo"]["name"] == "TestAgent"

        _rpc(ws, 2, "session/new", {"cwd": "/tmp", "mcpServers": []})
        new_session, _ = _await_response(ws, 2)
        assert new_session["result"]["sessionId"]


def test_ws_route_streams_prompt_updates_before_response():
    with TestClient(_ws_app()).websocket_connect("/acp/ws") as ws:
        _rpc(ws, 1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        _await_response(ws, 1)
        _rpc(ws, 2, "session/new", {"cwd": "/tmp", "mcpServers": []})
        session_id = _await_response(ws, 2)[0]["result"]["sessionId"]

        _rpc(
            ws,
            3,
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]},
        )
        response, notifications = _await_response(ws, 3)

    assert response["result"]["stopReason"] == "end_turn"
    updates = [n["params"]["update"] for n in notifications]
    assert {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": "hi there"},
    } in updates


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


def test_ws_route_closes_when_the_socket_bridge_cannot_start(monkeypatch):
    async def _boom(**_):
        raise OSError("no file descriptors available")

    monkeypatch.setattr(asyncio, "open_connection", _boom)

    with (
        TestClient(_ws_app()).websocket_connect("/acp/ws") as ws,
        pytest.raises(WebSocketDisconnect),
    ):
        ws.receive_text()
