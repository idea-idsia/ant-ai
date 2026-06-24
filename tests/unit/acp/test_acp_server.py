from __future__ import annotations

from unittest.mock import MagicMock

from starlette.routing import WebSocketRoute

from ant_ai.acp.server import ACPServer, build_acp_ws_route


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
