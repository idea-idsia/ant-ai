from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncGenerator, AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
import uvicorn
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

import ant_ai.llm.integrations.lite_llm as _llm_mod
from ant_ai.a2a.colony import Colony
from ant_ai.agent.agent import Agent
from ant_ai.agent.base import BaseAgent
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.workflow.workflow import END, START, NodeYield, Workflow


class _FakeUsage:
    def model_dump(self) -> dict:
        return {"in_tokens": 1, "out_tokens": 1}


class _FakeMessage:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.role = "assistant"
        self._content = content
        self.tool_calls = tool_calls or []

    def get(self, key: str, default: Any = None) -> Any:
        return self._content if key == "content" else default


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeResponse:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]
        self.usage = _FakeUsage()


class _FakeToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.id = "call_test_001"
        self.function = _FakeToolFunction(name, arguments)


def make_text_response(content: str) -> _FakeResponse:
    """LLM returns a plain-text final answer (no tool calls)."""
    return _FakeResponse(_FakeMessage(content))


def make_tool_call_response(tool_name: str, message: str) -> _FakeResponse:
    """LLM returns a tool call targeting `tool_name` with `message` payload.

    The A2AAgentTool parameter schema uses "message" as the required key
    (see A2AAgentTool._init_metadata).
    """
    tc = _FakeToolCall(tool_name, json.dumps({"message": message}))
    return _FakeResponse(_FakeMessage("", tool_calls=[tc]))


def _stream_chunk_from_response(resp: _FakeResponse) -> SimpleNamespace:
    """Adapt a non-streaming `_FakeResponse` into one litellm-shaped stream chunk."""
    message = resp.choices[0].message
    tool_deltas = [
        SimpleNamespace(
            index=i,
            id=tc.id,
            function=SimpleNamespace(
                name=tc.function.name, arguments=tc.function.arguments
            ),
        )
        for i, tc in enumerate(message.tool_calls)
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=message.get("content"),
                    reasoning_content=None,
                    tool_calls=tool_deltas or None,
                )
            )
        ]
    )


def _bound_socket() -> socket.socket:
    """Bind a socket to a free loopback port and return it **open**.

    Keeping the socket open prevents any other process or VS Code test-runner
    parallel session from claiming the same port before uvicorn starts.
    Pass the returned socket to ``_start_server``; uvicorn takes ownership.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(128)
    return s


async def _start_server(
    app: Any, sock: socket.socket
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
        raise RuntimeError(
            f"Agent server on {sock.getsockname()} did not start within 2 s"
        )
    return server, task


async def _stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    await task


async def _run_agent_once(
    agent: BaseAgent, state: State, ctx: InvocationContext | None
) -> AsyncGenerator[NodeYield]:
    """Workflow node: invoke the agent's ReAct loop once and pass events through."""
    async for event in agent.stream(state, ctx=ctx):
        yield event
    yield state


def build_single_node_workflow() -> Workflow:
    """START → run_agent → END  (no validation or fix loops)."""
    wf = Workflow()
    wf.add_node("run", _run_agent_once)
    wf.add_edge(START, "run")
    wf.add_edge("run", END)
    return wf


async def _not_installed(*_: Any, **__: Any) -> None:
    raise AssertionError(
        "scripted_llm.install(async_fn) must be called before any LLM call."
    )


@pytest.fixture
def scripted_llm(monkeypatch: pytest.MonkeyPatch):
    """
    Returns a helper object with:
      .install(async_fn)  – monkeypatches acompletion with your dispatch fn
      .make_text_response(content)            – factory
      .make_tool_call_response(name, message) – factory

    The dispatch fn signature:
        async def dispatch(*, model, messages, **kwargs) -> mock_response
    """
    monkeypatch.setattr(_llm_mod, "acompletion", _not_installed)

    class _ScriptedLLM:
        make_text_response = staticmethod(make_text_response)
        make_tool_call_response = staticmethod(make_tool_call_response)

        def install(self, fn: Any) -> None:
            async def adapted(*, stream: bool = False, **kwargs: Any) -> Any:
                """Streaming is now always attempted (see ChatLLM.stream()'s default
                and Agent's hook-safety gate), but most dispatch fns here only know
                how to answer non-streaming calls. When one of those returns a plain
                `_FakeResponse` for a `stream=True` call, adapt it into a one-chunk
                stream instead of forcing every test's dispatch fn to branch on
                `stream` itself. A dispatch fn that already returns something
                iterable (e.g. it branches on `stream` itself) passes through as-is.
                """
                result: Any = await fn(stream=stream, **kwargs)
                if stream and not hasattr(result, "__aiter__"):

                    async def gen() -> AsyncIterator[SimpleNamespace]:
                        yield _stream_chunk_from_response(result)

                    return gen()
                return result

            monkeypatch.setattr(_llm_mod, "acompletion", adapted)

    return _ScriptedLLM()


@pytest_asyncio.fixture
async def two_agent_hive(scripted_llm: Any) -> AsyncGenerator[dict]:
    """
    In-process colony with two agents:
      'caller'    – has an A2AAgentTool wired to 'responder'
      'responder' – answers directly

    System-prompt substrings for dispatch:
      caller    → "You are a caller"
      responder → "You are a responder"

    Yields {"ports": {"caller": int, "responder": int}}.
    """
    caller_sock = _bound_socket()
    responder_sock = _bound_socket()
    caller_port = caller_sock.getsockname()[1]
    responder_port = responder_sock.getsockname()[1]

    caller_agent = Agent(
        name="Caller",
        llm=LiteLLMChat("test-model"),
        system_prompt="You are a caller. You may delegate to the responder agent.",
        description="A caller agent",
    )
    responder_agent = Agent(
        name="Responder",
        llm=LiteLLMChat("test-model"),
        system_prompt="You are a responder. Answer questions directly.",
        description="A responder agent",
    )

    def _card(name: str, description: str, port: int, skill_id: str) -> AgentCard:
        card = AgentCard(
            name=name,
            description=description,
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
        )
        card.supported_interfaces.append(
            AgentInterface(protocol_binding="JSONRPC", url=f"http://127.0.0.1:{port}/")
        )
        card.skills.append(
            AgentSkill(id=skill_id, name=description, description=description)
        )
        return card

    caller_card = _card("caller", "A caller agent", caller_port, "call")
    responder_card = _card("responder", "A responder agent", responder_port, "respond")

    colony = Colony()  # db_url=None → InMemoryTaskStore
    colony.agent(
        "caller",
        agent=caller_agent,
        workflow=build_single_node_workflow(),
        card=caller_card,
    )
    colony.agent(
        "responder",
        agent=responder_agent,
        workflow=build_single_node_workflow(),
        card=responder_card,
    )
    colony.collab("caller", "responder")  # auto-endpoint: responder_card.url

    servers_tasks = []
    for name, sock in [("caller", caller_sock), ("responder", responder_sock)]:
        app = colony.asgi(agent_name=name, use_fastapi=True)
        server, task = await _start_server(app, sock)
        servers_tasks.append((server, task))

    yield {"ports": {"caller": caller_port, "responder": responder_port}}

    for server, task in servers_tasks:
        await _stop_server(server, task)
