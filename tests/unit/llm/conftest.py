from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from ant_ai.core.message import Message
from ant_ai.core.response import ChatLLMStreamChunk


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message(role="system", content="You are a test system."),
        Message(role="user", content="Hello!"),
    ]


@pytest.fixture
def expected_dict_messages(sample_messages: list[Message]) -> list[dict[str, str]]:
    return [m.model_dump(exclude={"kind"}) for m in sample_messages]


@pytest.fixture
def collect_stream_chunks():
    async def _collect(
        async_iter: AsyncIterator[ChatLLMStreamChunk],
    ) -> tuple[list[ChatLLMStreamChunk], str]:
        chunks: list[ChatLLMStreamChunk] = []
        parts: list[str] = []
        async for c in async_iter:
            chunks.append(c)
            parts.append(c.delta.delta)
        return chunks, "".join(parts)

    return _collect


@pytest.fixture
def make_async_stream():
    async def _make(items: Iterable[Any]):
        for item in items:
            yield item

    return _make


@dataclass(slots=True)
class DummyOpenAIChatResponse:
    content: str | None
    choices: list = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=self.content))]


@dataclass(slots=True)
class Usage:
    in_tokens: int
    out_tokens: int

    def model_dump(self):
        return {"in_tokens": self.in_tokens, "out_tokens": self.out_tokens}


@dataclass(slots=True)
class DummyOpenAIStreamChunk:
    content: str | None
    choices: list = field(init=False, repr=False)
    usage: Usage = field(init=False)

    def __post_init__(self) -> None:
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=self.content))]
        self.usage = Usage(in_tokens=10, out_tokens=15)


@pytest.fixture
def openai_sync_response():
    def _make(content: str | None) -> DummyOpenAIChatResponse:
        return DummyOpenAIChatResponse(content=content)

    return _make


@pytest.fixture
def openai_stream_chunk():
    def _make(content: str | None) -> DummyOpenAIStreamChunk:
        return DummyOpenAIStreamChunk(content=content)

    return _make


class _LiteLLMMessage(SimpleNamespace):
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@pytest.fixture
def litellm_response():
    def _make(
        content: str,
        *,
        role: str = "assistant",
        tool_calls: list[Any] | None = None,
    ):
        if tool_calls is None:
            tool_calls = []
        message = _LiteLLMMessage(role=role, content=content, tool_calls=tool_calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=Usage(**{"in_tokens": 10, "out_tokens": 15}),
        )

    return _make


@pytest.fixture
def litellm_stream_chunk():
    def _make(content: str | None):
        # shape: chunk.choices[0].delta.content
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
            usage=Usage(**{"in_tokens": 10, "out_tokens": 15}),
        )

    return _make


# Helper dummies for tool calls used by lite_llm.to_chatllm_response
@dataclass(slots=True)
class DummyToolFunction:
    name: str | None
    arguments: str | None


@dataclass(slots=True)
class DummyToolCall:
    id: str
    function: DummyToolFunction


@pytest.fixture
def dummy_tool_call():
    def _make(
        *,
        call_id: str = "call_1",
        name: str | None = "do_thing",
        arguments: str | None = '{"x": 1}',
    ) -> DummyToolCall:
        return DummyToolCall(
            id=call_id, function=DummyToolFunction(name=name, arguments=arguments)
        )

    return _make
