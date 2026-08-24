from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

import ant_ai.llm.integrations.lite_llm as _llm_mod
from ant_ai.agent.agent import Agent
from ant_ai.core.message import Message
from ant_ai.core.types import State
from ant_ai.hooks.builtins.history_compression import (
    _SUMMARY_PREFIX,
    HistoryCompressionHook,
)
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.workflow.workflow import END, START, Workflow


class _FakeUsage:
    def model_dump(self) -> dict:
        return {"in_tokens": 1, "out_tokens": 1}


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self._content = content
        self.tool_calls = []

    def get(self, key: str, default: Any = None) -> Any:
        return self._content if key == "content" else default


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(_FakeMessage(content))]
        self.usage = _FakeUsage()


def _adapt_stream(
    fn: Callable[..., Awaitable[_FakeResponse]],
) -> Callable[..., Awaitable[Any]]:
    """Wrap a non-streaming dispatch fn so `stream=True` calls (always attempted
    now — see ChatLLM.stream()'s default and Agent's hook-safety gate) get a
    one-chunk stream instead of the plain `_FakeResponse` these tests return.
    """

    async def wrapped(*, stream: bool = False, **kwargs: Any) -> Any:
        result = await fn(stream=stream, **kwargs)
        if not stream:
            return result

        async def gen() -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=result.choices[0].message.get("content"),
                            reasoning_content=None,
                            tool_calls=None,
                        )
                    )
                ]
            )

        return gen()

    return wrapped


class _SummaryLLM:
    """Minimal LLM used only by HistoryCompressionHook for summarisation."""

    def __init__(self, summary: str = "COMPRESSED_SUMMARY") -> None:
        self.calls: list[list[Message]] = []
        self._summary = summary

    async def ainvoke(self, messages: list[Message], **_: Any):
        self.calls.append(list(messages))

        class _R:
            pass

        r = _R()
        r.message = Message(role="assistant", content=self._summary)
        return r


def _make_agent(
    summary_llm: _SummaryLLM, *, max_messages: int, keep_last: int
) -> Agent:
    hook = HistoryCompressionHook(
        llm=summary_llm,
        max_messages=max_messages,
        keep_last=keep_last,
    )
    return Agent(
        name="test-agent",
        llm=LiteLLMChat("test-model"),
        system_prompt="You are a test agent.",
        description="Compression integration test agent",
        hooks=[hook],
    )


def _make_workflow() -> Workflow:
    from ant_ai.core.types import InvocationContext

    async def _run(agent, state, ctx: InvocationContext | None):
        async for event in agent.stream(state, ctx=ctx):
            yield event
        yield state

    wf = Workflow()
    wf.add_node("run", _run)
    wf.add_edge(START, "run")
    wf.add_edge("run", END)
    return wf


async def _run_turn_agent(agent: Agent, state: State, question: str) -> None:
    """Append a user message then drive agent.stream() to completion."""
    state.add_message(Message(role="user", content=question))
    async for _ in agent.stream(state, ctx=None):
        pass


async def _run_turn_workflow(
    agent: Agent, workflow: Workflow, state: State, question: str
) -> None:
    """Append a user message then drive workflow.stream() to completion."""
    state.add_message(Message(role="user", content=question))
    async for _ in workflow.stream(agent, ctx=None, state=state):
        pass


@pytest.mark.integration
async def test_compression_fires_in_standalone_agent(monkeypatch: pytest.MonkeyPatch):
    """HistoryCompressionHook compresses state.messages via the real Agent API.

    Drives agent.stream() directly (no workflow, no A2A).  After enough turns
    push state.messages over max_messages, the summary message appears and the
    history length stays bounded.
    """
    call_count = 0

    async def dispatch(*, messages: list, **_: Any) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        return _FakeResponse(f"Reply {call_count}.")

    monkeypatch.setattr(_llm_mod, "acompletion", _adapt_stream(dispatch))

    summary_llm = _SummaryLLM("STANDALONE_SUMMARY")
    # max_messages=5, keep_last=2 → compression fires on turn 3 (5 msgs: u1 a1 u2 a2 u3)
    agent = _make_agent(summary_llm, max_messages=5, keep_last=2)
    state = State()

    await _run_turn_agent(agent, state, "Turn one.")
    await _run_turn_agent(agent, state, "Turn two.")
    await _run_turn_agent(agent, state, "Turn three.")  # triggers compression

    assert summary_llm.calls, "Compression LLM was never called — hook did not fire"

    # state.messages must start with the summary
    assert state.messages[0].role == "system"
    assert (state.messages[0].content or "").startswith(_SUMMARY_PREFIX)
    assert "STANDALONE_SUMMARY" in (state.messages[0].content or "")

    # History stays bounded — must be shorter than the uncompressed 6-message chain
    assert len(state.messages) < 6, (
        f"Expected compression to bound history; got {len(state.messages)} messages"
    )


@pytest.mark.integration
async def test_compression_fires_in_agent_workflow(monkeypatch: pytest.MonkeyPatch):
    """HistoryCompressionHook compresses state.messages via the real Workflow API.

    Drives workflow.stream(agent, state=state) instead of agent.stream().
    Verifies the same compression behaviour is reachable through the workflow
    entry point that production code typically uses.
    """
    call_count = 0

    async def dispatch(*, messages: list, **_: Any) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        return _FakeResponse(f"Reply {call_count}.")

    monkeypatch.setattr(_llm_mod, "acompletion", _adapt_stream(dispatch))

    summary_llm = _SummaryLLM("WORKFLOW_SUMMARY")
    agent = _make_agent(summary_llm, max_messages=5, keep_last=2)
    workflow = _make_workflow()
    state = State()

    await _run_turn_workflow(agent, workflow, state, "Turn one.")
    await _run_turn_workflow(agent, workflow, state, "Turn two.")
    await _run_turn_workflow(
        agent, workflow, state, "Turn three."
    )  # triggers compression

    assert summary_llm.calls, "Compression LLM was never called — hook did not fire"

    assert state.messages[0].role == "system"
    assert (state.messages[0].content or "").startswith(_SUMMARY_PREFIX)
    assert "WORKFLOW_SUMMARY" in (state.messages[0].content or "")

    assert len(state.messages) < 6, (
        f"Expected compression to bound history; got {len(state.messages)} messages"
    )
