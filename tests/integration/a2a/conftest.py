from __future__ import annotations

import asyncio
import functools
import gc
import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import uuid4

import a2a.server.tasks.database_task_store as _db_store_mod
import pytest
import uvicorn
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.models import create_task_model as _create_task_model
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
    SendMessageRequest,
)
from sse_starlette.sse import AppStatus, _get_shutdown_state

import ant_ai.llm.integrations.lite_llm as _llm_mod
from ant_ai.a2a.colony import Colony
from ant_ai.agent.agent import Agent
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.workflow.workflow import END, START, NodeYield, Workflow

_A2A_DIR = Path(__file__).parent
_REPO_ROOT = _A2A_DIR.parents[2]

# Memoize create_task_model so that calling DatabaseTaskStore with the same
# table_name twice in the same process does not redefine the SQLAlchemy table
# on the shared Base.metadata, which would raise InvalidRequestError.
_db_store_mod.create_task_model = functools.lru_cache(maxsize=None)(_create_task_model)


@pytest.fixture
def minimal_workflow() -> Workflow:
    """
    Minimal valid workflow graph: START must have an outgoing edge.
    Using START -> END avoids needing any nodes and keeps tests fast.
    """
    wf = Workflow(max_steps=5)
    wf.add_edge(START, END)
    return wf


@pytest.fixture
def event_queue() -> EventQueue:
    return EventQueue()


@pytest.fixture
def request_context() -> RequestContext:
    """
    Standard RequestContext used across A2A integration tests.
    """
    msg = Message(
        role=Role.ROLE_USER,
        parts=[Part(text="ping")],
        message_id=uuid4().hex,
    )
    send_request = SendMessageRequest(message=msg)
    return RequestContext(call_context=None, request=send_request)


STICKY_AGENT_SYS = "You are a history test agent."


class _FakeUsage:
    def model_dump(self) -> dict:
        return {"in_tokens": 1, "out_tokens": 1}


class _FakeMessage:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.role = "assistant"
        self._content = content
        self.tool_calls = tool_calls or []

    def get(self, key: str, default=None):
        return self._content if key == "content" else default


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeResponse:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]
        self.usage = _FakeUsage()


def make_text_response(content: str) -> _FakeResponse:
    """LLM returns a plain-text final answer (no tool calls)."""
    return _FakeResponse(_FakeMessage(content))


def _free_port() -> int:
    """Ask the OS for an available TCP port on loopback.

    Used only when we need a plain port number (e.g. to pass via env var to a
    subprocess).  For in-process servers prefer ``_bound_socket`` so the port
    stays reserved until uvicorn takes ownership.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bound_socket() -> socket.socket:
    """Bind a socket to a free loopback port and return it **open**.

    Keeping the socket open prevents any other process or pytest-parallel
    worker from claiming the same port before uvicorn starts serving on it.
    Pass the returned socket to ``_start_server``; uvicorn takes ownership.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(128)
    return s


async def _start_server(
    app, sock: socket.socket
) -> tuple[uvicorn.Server, asyncio.Task]:
    config = uvicorn.Config(app=app, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    task: asyncio.Task = asyncio.create_task(server.serve(sockets=[sock]))
    for _ in range(40):  # wait up to 2 s
        if server.started:
            break
        await asyncio.sleep(0.05)
    else:
        task.cancel()
        raise RuntimeError(f"Server on {sock.getsockname()} did not start within 2 s")
    return server, task


async def _stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    await task

    state = _get_shutdown_state()
    for _ in range(20):  # wait up to ~1 s for the watcher task to finish
        if not state.watcher_started:
            break
        await asyncio.sleep(0.05)
    AppStatus.should_exit = False


async def _drain_loop() -> None:
    """Force GC and yield to the event loop after closing a Colony.

    Call this after ``colony.aclose()`` when using a DatabaseTaskStore.
    Asyncpg connection pools hold cyclic references; an explicit gc.collect()
    ensures finalizers run in the current event loop rather than being deferred.
    """
    gc.collect()
    await asyncio.sleep(0.1)  # flush any call_soon callbacks scheduled by finalizers


async def _run_agent_once(
    agent: Agent, state: State, ctx: InvocationContext | None
) -> AsyncGenerator[NodeYield]:
    """Workflow node: run the agent once and pass all events through."""
    async for event in agent.stream(state, ctx=ctx):
        yield event
    yield state


def build_single_node_workflow() -> Workflow:
    """START → run_agent → END."""
    wf = Workflow()
    wf.add_node("run", _run_agent_once)
    wf.add_edge(START, "run")
    wf.add_edge("run", END)
    return wf


def _make_agent_card(port: int) -> AgentCard:
    card = AgentCard(
        name="sticky",
        description="Agent for sticky history tests",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
    )
    card.supported_interfaces.append(
        AgentInterface(protocol_binding="JSONRPC", url=f"http://127.0.0.1:{port}/")
    )
    card.skills.append(AgentSkill(id="sticky", name="sticky", description="sticky"))
    return card


def _make_colony(db_url: str | None = None) -> tuple[Colony, Agent, socket.socket]:
    """Build a Colony + agent on a free port. Does NOT start the server.

    Returns an open bound socket (not a bare port number) so the port stays
    reserved until the caller passes it to ``_start_server``.
    """
    sock = _bound_socket()
    port = sock.getsockname()[1]
    agent = Agent(
        name="sticky",
        llm=LiteLLMChat("test-model"),
        system_prompt=STICKY_AGENT_SYS,
        description="Sticky history test agent",
    )
    colony = Colony(db_url=db_url)
    colony.agent(
        "sticky",
        agent=agent,
        workflow=build_single_node_workflow(),
        card=_make_agent_card(port),
    )
    return colony, agent, sock


async def _not_installed(*_, **__) -> None:
    raise AssertionError(
        "scripted_llm.install(async_fn) must be called before any LLM call."
    )


@pytest.fixture
def scripted_llm(monkeypatch: pytest.MonkeyPatch):
    """
    Monkeypatches acompletion so tests control LLM responses.

    Usage::

        async def dispatch(*, messages, **_):
            return make_text_response("hello")

        scripted_llm.install(dispatch)
    """
    monkeypatch.setattr(_llm_mod, "acompletion", _not_installed)

    class _ScriptedLLM:
        make_text_response = staticmethod(make_text_response)

        def install(self, fn) -> None:
            monkeypatch.setattr(_llm_mod, "acompletion", fn)

    return _ScriptedLLM()


@pytest.fixture
async def single_agent_hive(scripted_llm) -> AsyncGenerator[dict]:
    """
    Single-agent colony backed by InMemoryTaskStore. CI-safe, no external deps.

    Yields ``{"port": int}``.
    """
    colony, _, sock = _make_colony(db_url=None)
    port = sock.getsockname()[1]
    app = colony.asgi(agent_name="sticky", use_fastapi=True)
    server, task = await _start_server(app, sock)
    yield {"port": port}
    await _stop_server(server, task)


@pytest.fixture(scope="session")
def postgres_url():
    """Provide a live PostgreSQL URL, auto-starting docker compose if needed.

    If ``POSTGRES_TEST_URL`` is already set in the environment, uses that
    directly (CI / local with a running instance).  Otherwise starts the
    ``task-db`` service from ``compose.test.yml`` and tears it down after
    the session.  Skips if Docker is not available.
    """
    pg_url = os.getenv("POSTGRES_TEST_URL")
    started = False

    if not pg_url:
        compose_file = _REPO_ROOT / "compose.test.yml"
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d", "--wait"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            pytest.skip("docker not found — skipping postgres tests")
        if result.returncode != 0:
            pytest.skip(
                "docker compose failed to start postgres — skipping postgres tests.\n"
                + result.stderr
            )
        pg_url = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5433/tasks"
        started = True

    yield pg_url

    if started:
        subprocess.run(
            ["docker", "compose", "-f", str(_REPO_ROOT / "compose.test.yml"), "down"],
            capture_output=True,
        )


@pytest.fixture
async def pg_hive(postgres_url: str, scripted_llm) -> AsyncGenerator[dict]:
    """Single-agent colony backed by DatabaseTaskStore (PostgreSQL).

    Automatically starts postgres via docker compose if needed (see
    ``postgres_url`` fixture).  Skips if docker is unavailable.
    """
    colony, _, sock = _make_colony(db_url=postgres_url)
    port = sock.getsockname()[1]
    app = colony.asgi(agent_name="sticky", use_fastapi=True)
    server, srv_task = await _start_server(app, sock)
    yield {"port": port}
    await _stop_server(server, srv_task)
    await colony.aclose()
    await _drain_loop()


@pytest.fixture(scope="session")
def codegen_agent_server():
    """Start a minimal codegen A2A agent as a subprocess and yield its URL.

    The agent runs ``codegen_agent.py`` with a stub LLM, so no API key is
    needed.  The port is chosen dynamically to avoid conflicts.
    """
    port = _free_port()
    env = {**os.environ, "CODEGEN_AGENT_PORT": str(port)}

    proc = subprocess.Popen(
        [sys.executable, str(_A2A_DIR / "codegen_agent.py")],
        env=env,
    )

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    else:
        proc.kill()
        pytest.fail("codegen agent server did not start in time")

    yield f"http://127.0.0.1:{port}"

    proc.kill()
    proc.wait(timeout=5)
