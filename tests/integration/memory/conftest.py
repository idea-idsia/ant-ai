from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncGenerator

import pytest
import uvicorn
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from sse_starlette.sse import AppStatus, _get_shutdown_state

from ant_ai.a2a.colony import Colony
from ant_ai.agent.agent import Agent
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.memory.backends.mem0 import Mem0Memory
from ant_ai.workflow.workflow import END, START, NodeYield, Workflow

_MODEL = os.environ.get("MODEL", "gpt-5-mini")


def _bound_socket() -> socket.socket:
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
    for _ in range(40):
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
    for _ in range(20):
        if not state.watcher_started:
            break
        await asyncio.sleep(0.05)
    AppStatus.should_exit = False


async def _run_agent_once(
    agent: Agent,
    state: State,
    ctx: InvocationContext | None,
) -> AsyncGenerator[NodeYield]:
    async for event in agent.stream(state, ctx=ctx):
        yield event
    yield state


def _make_workflow() -> Workflow:
    wf = Workflow()
    wf.add_node("run", _run_agent_once)
    wf.add_edge(START, "run")
    wf.add_edge("run", END)
    return wf


def _make_card(port: int) -> AgentCard:
    card = AgentCard(
        name="memory-agent",
        description="An agent with persistent mem0 memory",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
    )
    card.supported_interfaces.append(
        AgentInterface(protocol_binding="JSONRPC", url=f"http://127.0.0.1:{port}/")
    )
    card.skills.append(
        AgentSkill(id="chat", name="chat", description="Chat with memory")
    )
    return card


@pytest.fixture
async def memory_agent_server() -> AsyncGenerator[dict]:
    """Colony server with Mem0Memory on a free port. Yields {"url", "user_id"}."""
    import uuid

    user_id = f"test-{uuid.uuid4().hex[:8]}"
    sock = _bound_socket()
    port = sock.getsockname()[1]

    agent = Agent(
        name="memory-agent",
        llm=LiteLLMChat(_MODEL),
        system_prompt=(
            "You are a helpful assistant with persistent memory, exposed via "
            "the Mem0Memory_search and Mem0Memory_add tools. Whenever the "
            "user shares a fact, preference, or personal detail, call "
            "Mem0Memory_add to save it immediately — do not wait to be "
            "asked. Before answering questions that depend on something you "
            "might already know about the user, call Mem0Memory_search "
            "first. Use what you know about the user to give personalised "
            "responses."
        ),
        description="An agent with persistent mem0 memory",
        memory=Mem0Memory(),
    )
    colony = Colony()
    colony.agent(
        "memory-agent",
        agent=agent,
        workflow=_make_workflow(),
        card=_make_card(port),
    )
    app = colony.asgi(agent_name="memory-agent", use_fastapi=True)

    server, task = await _start_server(app, sock)
    yield {"url": f"http://127.0.0.1:{port}/", "user_id": user_id}
    await _stop_server(server, task)
