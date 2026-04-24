from __future__ import annotations

from a2a.server.events import QueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.applications import Starlette

from ant_ai.a2a.context_builder import HistoryRequestContextBuilder
from ant_ai.a2a.executor import A2AExecutor
from ant_ai.agent.agent import Agent
from ant_ai.workflow.workflow import Workflow


class A2AServer(BaseModel):
    """Class to instantiate a fully usable A2A server via uvicorn, either as a FastAPI or Starlete app."""

    agent: Agent
    workflow: Workflow
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=9000)
    agent_card: AgentCard
    task_store: TaskStore = Field(default_factory=InMemoryTaskStore)
    queue_manager: QueueManager | None = Field(default=None)
    push_config_store: PushNotificationConfigStore | None = Field(default=None)
    push_sender: PushNotificationSender | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _create_request_handler(self) -> A2AServer:
        request_context_builder = HistoryRequestContextBuilder(
            should_populate_referred_tasks=True,
            task_store=self.task_store,
        )

        self._request_handler: DefaultRequestHandler = DefaultRequestHandler(
            agent_executor=A2AExecutor(
                agent=self.agent,
                workflow=self.workflow,
            ),
            task_store=self.task_store,
            agent_card=self.agent_card,
            request_context_builder=request_context_builder,
            queue_manager=self.queue_manager,
            push_config_store=self.push_config_store,
            push_sender=self.push_sender,
        )
        return self

    def _build_routes(self) -> list:
        routes = []
        routes.extend(create_agent_card_routes(self.agent_card))
        routes.extend(create_jsonrpc_routes(self._request_handler, rpc_url="/"))
        return routes

    def starlette_app(self) -> Starlette:
        """Create a Starlette application for serving the agent"""
        return Starlette(routes=self._build_routes())

    def fastapi_app(self) -> FastAPI:
        """Create a FastAPI application for serving the agent"""
        return FastAPI(title=self.agent_card.name, routes=self._build_routes())

    def serve(self, use_fastapi: bool = True) -> None:
        """Start serving the agent using Uvicorn

        Args:
            use_fastapi: Whether to use FastAPI or Starlette as the web framework. Defaults to True (FastAPI).

        Returns:
            None
        """
        try:
            import uvicorn

            logger.info(
                f"Starting A2A server for agent '{self.agent.name}' at {self.host}:{self.port}..."
            )
            app: Starlette = self.fastapi_app() if use_fastapi else self.starlette_app()
            logger.info(
                f"Using {'FastAPI' if use_fastapi else 'Starlette'} application"
            )
            uvicorn.run(app, host=self.host, port=self.port)
        except ImportError as e:
            raise ImportError(
                "Uvicorn is not installed. Please install it with 'pip install uvicorn'."
            ) from e
        except KeyboardInterrupt:
            logger.info("Server stopped")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise RuntimeError(f"Failed to start the server: {e}") from e
