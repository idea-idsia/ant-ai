from __future__ import annotations

import pytest

from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.response import ChatLLMResponse
from ant_ai.llm.protocol import ChatLLM


class _AinvokeOnlyLLM(ChatLLM):
    """Backend implementing only ainvoke() — the shape of most test doubles
    and any custom backend with no incremental streaming API."""

    def __init__(self, response: ChatLLMResponse) -> None:
        self._response = response

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        return self._response


class _NoOverridesLLM(ChatLLM):
    """Implements none of invoke/ainvoke/stream."""


@pytest.mark.unit
async def test_stream_default_falls_back_to_ainvoke_as_one_chunk():
    """A backend with no incremental API still supports .stream() via the
    Protocol default, just without live token granularity."""
    response = ChatLLMResponse(
        message=Message(role="assistant", content="hello"),
        tool_calls=[],
        reasoning="thinking",
    )
    llm = _AinvokeOnlyLLM(response)

    chunks = [c async for c in llm.stream([Message(role="user", content="hi")])]

    assert len(chunks) == 1
    assert chunks[0].delta.delta == "hello"
    assert chunks[0].reasoning_delta == "thinking"
    assert chunks[0].tool_calls is None


@pytest.mark.unit
async def test_stream_default_emits_one_chunk_per_tool_call():
    response = ChatLLMResponse(
        message=Message(role="assistant", content=""),
        tool_calls=[
            ToolCall(
                id="call_1", function=ToolFunction(name="echo", arguments='{"x": 1}')
            ),
            ToolCall(
                id="call_2", function=ToolFunction(name="echo", arguments='{"x": 2}')
            ),
        ],
    )
    llm = _AinvokeOnlyLLM(response)

    chunks = [c async for c in llm.stream([Message(role="user", content="hi")])]

    assert [c.tool_calls["id"] for c in chunks] == ["call_1", "call_2"]
    assert [c.tool_calls["index"] for c in chunks] == [0, 1]
    assert [c.tool_calls["arguments"] for c in chunks] == ['{"x": 1}', '{"x": 2}']


@pytest.mark.unit
async def test_invoke_and_ainvoke_have_no_default_implementation():
    """Pins the asymmetry that makes ChatLLM.stream()'s default safe:
    stream() wraps ainvoke(), but invoke()/ainvoke() must NOT get a
    symmetric default (e.g. draining stream()) — a backend implementing
    neither would otherwise recurse between the two defaults forever
    instead of failing with a clear error.
    """
    llm = _NoOverridesLLM()

    with pytest.raises(NotImplementedError):
        await llm.ainvoke([])

    with pytest.raises(NotImplementedError):
        llm.invoke([])


@pytest.mark.unit
async def test_stream_default_surfaces_notimplementederror_without_recursing():
    """End-to-end version of the invariant above: calling .stream() on a
    backend that implements nothing must fail fast via ainvoke()'s
    NotImplementedError, not hang or recurse.
    """
    llm = _NoOverridesLLM()

    with pytest.raises(NotImplementedError):
        async for _ in llm.stream([]):
            pass
