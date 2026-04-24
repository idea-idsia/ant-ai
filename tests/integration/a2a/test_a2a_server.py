import builtins
import sys
from typing import Any

import pytest
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from starlette.applications import Starlette

from ant_ai.a2a.executor import A2AExecutor
from ant_ai.a2a.server import A2AServer
from ant_ai.agent.agent import Agent
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.workflow.workflow import Workflow


@pytest.fixture
def agent() -> Agent:
    return Agent(
        name="Remote",
        system_prompt="You are a 10x Developer, you always start your answers with <DEPENDS>",
        llm=LiteLLMChat(model="hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct"),
    )


@pytest.fixture
def skills() -> list[AgentSkill]:
    return [
        AgentSkill(
            id="echo",
            name="echo",
            description="Echo skill for testing",
            input_modes=["text"],
            output_modes=["text"],
            tags=["test"],
        )
    ]


@pytest.fixture
def capabilities() -> AgentCapabilities:
    return AgentCapabilities(streaming=True)


@pytest.fixture
def agent_card(
    agent: Agent, skills: list[AgentSkill], capabilities: AgentCapabilities
) -> AgentCard:
    """
    AgentCard is explicitly injected into A2AServer, so we test that the server
    serves exactly what it's provided.
    """
    card = AgentCard(
        name=agent.name,
        description=agent.description,
        capabilities=capabilities,
        version="1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
    )
    card.supported_interfaces.append(
        AgentInterface(protocol_binding="JSONRPC", url="http://localhost:9000/")
    )
    card.skills.extend(skills)
    return card


@pytest.fixture
def task_store() -> TaskStore:
    return InMemoryTaskStore()


@pytest.fixture
def server(
    agent: Agent,
    minimal_workflow: Workflow,
    agent_card: AgentCard,
    task_store: TaskStore,
) -> A2AServer:
    return A2AServer(
        agent=agent,
        workflow=minimal_workflow,
        agent_card=agent_card,
        host="127.0.0.1",
        port=9000,
        task_store=task_store,
        queue_manager=None,
        push_config_store=None,
        push_sender=None,
    )


@pytest.fixture
def fastapi_app(server: A2AServer) -> FastAPI:
    return server.fastapi_app()


@pytest.fixture
def fastapi_client(fastapi_app: FastAPI) -> TestClient:
    return TestClient(fastapi_app)


def test_server_exposes_injected_agent_card(
    server: A2AServer, agent_card: AgentCard
) -> None:
    assert server.agent_card is agent_card
    assert isinstance(server.agent_card, AgentCard)


def test_request_handler_created_and_wired(server: A2AServer) -> None:
    """
    Validates the model_validator created a DefaultRequestHandler and that it wires
    executor/task_store correctly.
    """
    handler = getattr(server, "_request_handler", None)
    assert isinstance(handler, DefaultRequestHandler)

    assert isinstance(handler.agent_executor, A2AExecutor)
    assert handler.agent_executor.agent is server.agent
    assert handler.agent_executor.workflow is server.workflow

    assert handler.task_store is server.task_store


def test_starlette_app_builds(server: A2AServer) -> None:
    app: Starlette = server.starlette_app()
    assert isinstance(app, Starlette)


def test_fastapi_app_builds(fastapi_app: FastAPI) -> None:
    assert isinstance(fastapi_app, FastAPI)


def test_fastapi_agent_card_endpoint_returns_injected_card(
    server: A2AServer, fastapi_client: TestClient
) -> None:
    response: Response = fastapi_client.get("/.well-known/agent-card.json")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == server.agent_card.name
    assert data.get("description") == server.agent_card.description

    assert "capabilities" in data
    assert "skills" in data

    default_input_modes = data.get("default_input_modes", data.get("defaultInputModes"))
    default_output_modes = data.get(
        "default_output_modes", data.get("defaultOutputModes")
    )

    assert default_input_modes == ["text"]
    assert default_output_modes == ["text"]


@pytest.mark.parametrize(
    "use_fastapi,expected_type", [(True, FastAPI), (False, Starlette)]
)
def test_serve_selects_app_and_calls_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    server: A2AServer,
    use_fastapi: bool,
    expected_type: type,
) -> None:
    """
    Ensure serve() constructs the correct app flavor and forwards host/port to uvicorn.run.
    """
    called: dict[str, Any] = {}

    class DummyUvicorn:
        def run(self, app, host, port):
            called["app"] = app
            called["host"] = host
            called["port"] = port

    monkeypatch.setitem(sys.modules, "uvicorn", DummyUvicorn())

    server.serve(use_fastapi=use_fastapi)

    assert isinstance(called["app"], expected_type)
    assert called["host"] == server.host
    assert called["port"] == server.port


def test_serve_raises_import_error_when_uvicorn_missing(
    monkeypatch: pytest.MonkeyPatch, server: A2AServer
) -> None:
    """
    serve() should raise a friendly ImportError if uvicorn can't be imported.
    We force import failure by patching builtins.__import__.
    """
    original_import = builtins.__import__

    def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ImportError("No module named uvicorn")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    with pytest.raises(ImportError, match="Uvicorn is not installed"):
        server.serve(use_fastapi=True)


def test_serve_wraps_unexpected_errors_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch, server: A2AServer
) -> None:
    """
    If uvicorn.run throws, serve() should wrap into RuntimeError (per implementation).
    """

    class DummyUvicorn:
        def run(self, app, host, port):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "uvicorn", DummyUvicorn())

    with pytest.raises(RuntimeError, match=r"Failed to start the server: .*boom"):
        server.serve(use_fastapi=True)
