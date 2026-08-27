from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from ant_ai.core.events import ContentDeltaEvent, FinalAnswerEvent, ToolCallingEvent
from ant_ai.core.message import Message, MessageChunk
from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk
from ant_ai.core.result import StepResult
from ant_ai.core.types import State
from ant_ai.steps.llm_step import LLMStep


class FakeLLM:
    """Scripted ChatLLM: .ainvoke() returns a fixed response, .stream() replays chunks."""

    def __init__(self, chunks: list[ChatLLMStreamChunk], response: ChatLLMResponse):
        self.model = "fake-model"
        self._chunks = chunks
        self._response = response

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        return self._response

    def stream(
        self, messages, *, ctx=None, tools=None, response_format=None
    ) -> AsyncIterator[ChatLLMStreamChunk]:
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


def _make_state() -> State:
    state = State()
    state.add_message(Message(role="user", content="hi"))
    return state


async def _collect(gen) -> list[Any]:
    return [item async for item in gen]


@pytest.mark.unit
async def test_streaming_accumulates_to_same_content_as_buffered():
    chunks = [
        ChatLLMStreamChunk(delta=MessageChunk(role="assistant", delta="Hel")),
        ChatLLMStreamChunk(delta=MessageChunk(role="assistant", delta="lo ")),
        ChatLLMStreamChunk(delta=MessageChunk(role="assistant", delta="world")),
    ]
    response = ChatLLMResponse(message=Message(role="assistant", content="Hello world"))
    llm = FakeLLM(chunks, response)

    step = LLMStep(llm=llm, system_message=Message(role="system", content="sys"))

    buffered_items = await _collect(step.run(_make_state(), None))
    streamed_items = await _collect(step.stream(_make_state(), None))

    buffered_final = next(i for i in buffered_items if isinstance(i, FinalAnswerEvent))
    streamed_final = next(i for i in streamed_items if isinstance(i, FinalAnswerEvent))
    assert streamed_final.content == buffered_final.content == "Hello world"

    streamed_result = next(i for i in streamed_items if isinstance(i, StepResult))
    buffered_result = next(i for i in buffered_items if isinstance(i, StepResult))
    assert streamed_result.output.raw == buffered_result.output.raw == "Hello world"


@pytest.mark.unit
async def test_streaming_yields_content_deltas_in_order_before_terminal_event():
    chunks = [
        ChatLLMStreamChunk(delta=MessageChunk(role="assistant", delta="Hel")),
        ChatLLMStreamChunk(delta=MessageChunk(role="assistant", delta="lo")),
    ]
    response = ChatLLMResponse(message=Message(role="assistant", content="Hello"))
    llm = FakeLLM(chunks, response)
    step = LLMStep(llm=llm, system_message=Message(role="system", content="sys"))

    items = await _collect(step.stream(_make_state(), None))

    deltas = [i for i in items if isinstance(i, ContentDeltaEvent)]
    assert [d.delta for d in deltas] == ["Hel", "lo"]
    assert deltas[0].is_first is True
    assert deltas[1].is_first is False
    assert deltas[0].stream_id == deltas[1].stream_id

    final = next(i for i in items if isinstance(i, FinalAnswerEvent))
    assert final.stream_id == deltas[0].stream_id
    # deltas must be forwarded before the terminal event that closes the stream
    assert items.index(deltas[-1]) < items.index(final)


@pytest.mark.unit
async def test_streaming_reassembles_tool_call_from_argument_fragments():
    chunks = [
        ChatLLMStreamChunk(
            delta=MessageChunk(role="assistant", delta=""),
            tool_calls={
                "index": 0,
                "id": "call-1",
                "name": "my_tool",
                "arguments": '{"a"',
            },
        ),
        ChatLLMStreamChunk(
            delta=MessageChunk(role="assistant", delta=""),
            tool_calls={"index": 0, "id": None, "name": None, "arguments": ": 1}"},
        ),
    ]
    response = ChatLLMResponse(message=Message(role="assistant", content=""))
    llm = FakeLLM(chunks, response)
    step = LLMStep(llm=llm, system_message=Message(role="system", content="sys"))

    items = await _collect(step.stream(_make_state(), None))

    deltas = [i for i in items if isinstance(i, ContentDeltaEvent)]
    assert deltas[0].is_first is True
    assert deltas[0].tool_call_id == "call-1"
    assert deltas[0].tool_call_name == "my_tool"
    assert deltas[1].is_first is False
    assert deltas[1].tool_call_id is None

    tool_event = next(i for i in items if isinstance(i, ToolCallingEvent))
    assert len(tool_event.tool_calls) == 1
    call = tool_event.tool_calls[0]
    assert call.id == "call-1"
    assert call.function.name == "my_tool"
    assert call.function.arguments == '{"a": 1}'


@pytest.mark.unit
async def test_streaming_disambiguates_interleaved_tool_calls_by_index():
    chunks = [
        ChatLLMStreamChunk(
            delta=MessageChunk(role="assistant", delta=""),
            tool_calls={"index": 0, "id": "call-0", "name": "tool_a", "arguments": "{"},
        ),
        ChatLLMStreamChunk(
            delta=MessageChunk(role="assistant", delta=""),
            tool_calls={"index": 1, "id": "call-1", "name": "tool_b", "arguments": "{"},
        ),
        ChatLLMStreamChunk(
            delta=MessageChunk(role="assistant", delta=""),
            tool_calls={"index": 0, "id": None, "name": None, "arguments": "}"},
        ),
        ChatLLMStreamChunk(
            delta=MessageChunk(role="assistant", delta=""),
            tool_calls={"index": 1, "id": None, "name": None, "arguments": "}"},
        ),
    ]
    response = ChatLLMResponse(message=Message(role="assistant", content=""))
    llm = FakeLLM(chunks, response)
    step = LLMStep(llm=llm, system_message=Message(role="system", content="sys"))

    items = await _collect(step.stream(_make_state(), None))

    tool_event = next(i for i in items if isinstance(i, ToolCallingEvent))
    calls_by_name = {c.function.name: c for c in tool_event.tool_calls}
    assert calls_by_name["tool_a"].function.arguments == "{}"
    assert calls_by_name["tool_b"].function.arguments == "{}"
