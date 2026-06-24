from __future__ import annotations

import asyncio
import socket
from typing import Annotated

from acp.schema import AvailableCommand
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, SkipValidation
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

from ant_ai.acp.adapter import ACPAdapter
from ant_ai.agent.agent import Agent
from ant_ai.workflow.workflow import Workflow


def build_acp_ws_route(
    agent: Agent,
    workflow: Workflow,
    *,
    slash_commands: list[AvailableCommand] | None = None,
) -> WebSocketRoute:
    """Return a Starlette WebSocketRoute that bridges ACP over WebSocket at ``/acp/ws``."""
    adapter = ACPAdapter(agent, workflow, slash_commands=slash_commands)

    async def _handle_ws(websocket: WebSocket) -> None:
        from acp.agent.connection import AgentSideConnection

        await websocket.accept()

        sock_agent, sock_bridge = socket.socketpair()
        try:
            reader_agent, writer_agent = await asyncio.open_connection(sock=sock_agent)
            reader_bridge, writer_bridge = await asyncio.open_connection(
                sock=sock_bridge
            )
        except Exception:
            sock_agent.close()
            sock_bridge.close()
            await websocket.close()
            return

        conn = AgentSideConnection(adapter, writer_agent, reader_agent, listening=False)

        async def _ws_to_pipe() -> None:
            try:
                async for message in websocket.iter_text():
                    writer_bridge.write(message.encode() + b"\n")
                    await writer_bridge.drain()
            finally:
                writer_bridge.close()

        async def _pipe_to_ws() -> None:
            try:
                while True:
                    line = await reader_bridge.readline()
                    if not line:
                        break
                    await websocket.send_text(line.decode().rstrip("\n"))
            except Exception:
                pass

        try:
            await asyncio.gather(
                conn.listen(),
                _ws_to_pipe(),
                _pipe_to_ws(),
                return_exceptions=True,
            )
        finally:
            await conn.close()
            writer_agent.close()
            writer_bridge.close()
            sock_agent.close()
            sock_bridge.close()

    return WebSocketRoute("/acp/ws", _handle_ws)


class ACPServer(BaseModel):
    """Serve an ant-ai agent over the Agent Client Protocol (ACP).

    Supports two modes:

    - **ASGI / WebSocket** – call :meth:`starlette_app` or :meth:`fastapi_app` to get an
      ASGI application that exposes the agent at ``/acp/ws``.
    - **stdio** – call :meth:`serve_stdio` to run the agent as a stdio ACP process that
      editors such as Zed or Gemini CLI can spawn directly.

    To serve both A2A and ACP from a single process, combine the routes from each server's
    ASGI app — they occupy disjoint paths and compose cleanly::

        from starlette.applications import Starlette

        a2a = A2AServer(agent=agent, workflow=wf, agent_card=card)
        acp = ACPServer(agent=agent, workflow=wf)

        app = Starlette(routes=[*a2a.starlette_app().routes, *acp.starlette_app().routes])
        # A2A: POST /  and GET /.well-known/agent-card.json
        # ACP: WS  /acp/ws
    """

    agent: Annotated[Agent, SkipValidation]
    workflow: Annotated[Workflow, SkipValidation]
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=9001)
    slash_commands: list[AvailableCommand] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build_routes(self) -> list:
        return [
            build_acp_ws_route(
                self.agent, self.workflow, slash_commands=self.slash_commands or None
            )
        ]

    def starlette_app(self) -> Starlette:
        """Create a Starlette application serving ACP over WebSocket."""
        return Starlette(routes=self.build_routes())

    def fastapi_app(self) -> FastAPI:
        """Create a FastAPI application serving ACP over WebSocket."""
        return FastAPI(title=self.agent.name, routes=self.build_routes())

    def serve(self) -> None:
        """Start a uvicorn server exposing ACP over WebSocket at ``/acp/ws``."""
        try:
            import uvicorn

            logger.info(
                f"Starting ACP WebSocket server for agent '{self.agent.name}' "
                f"at {self.host}:{self.port} (ws://{self.host}:{self.port}/acp/ws)..."
            )
            uvicorn.run(self.starlette_app(), host=self.host, port=self.port)
        except ImportError as e:
            raise ImportError(
                "Uvicorn is not installed. Please install it with 'uv add uvicorn'."
            ) from e
        except KeyboardInterrupt:
            logger.info("Server stopped")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise RuntimeError(f"Failed to start the server: {e}") from e

    def serve_stdio(self) -> None:
        """Run the agent as a stdio ACP process (for editors/CLIs that spawn agents)."""
        import asyncio

        from acp import run_agent

        logger.info(f"Starting ACP stdio agent '{self.agent.name}'...")
        adapter = ACPAdapter(
            self.agent, self.workflow, slash_commands=self.slash_commands or None
        )
        asyncio.run(run_agent(adapter))
