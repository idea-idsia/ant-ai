from __future__ import annotations

import os
from collections.abc import AsyncIterator

from litellm import ModelResponse, acompletion, completion
from litellm.types.utils import Choices
from pydantic import BaseModel

from ant_ai.core.message import Message, MessageChunk, ToolFunction
from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk, ToolCall
from ant_ai.core.types import InvocationContext
from ant_ai.llm.protocol import ChatLLM


def to_chatllm_response(
    resp: ModelResponse,
) -> ChatLLMResponse:
    choice: Choices = resp.choices[0]

    message = Message(
        role=choice.message.role,
        content=choice.message.get("content", ""),
    )

    tool_calls = [
        ToolCall(
            id=tc.id,
            function=ToolFunction(
                name=tc.function.name or "",
                arguments=tc.function.arguments or "",
            ),
        )
        for tc in (choice.message.tool_calls or [])
    ]

    reasoning = getattr(choice.message, "reasoning_content", None) or None

    return ChatLLMResponse(
        message=message,
        tool_calls=tool_calls,
        usage=resp.usage.model_dump(),  # ty:ignore[unresolved-attribute]
        reasoning=reasoning,
    )


class LiteLLMChat(ChatLLM):
    """LiteLLM-based chat model. Supports multiple endpoints via LiteLLM."""

    def __init__(self, model: str) -> None:
        self.model: str = model
        self.default_params: dict = {}

    @staticmethod
    def _to_litellm_messages(messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects into LiteLLM-compatible dicts."""
        return [m.model_dump(exclude={"kind"}) for m in messages]

    def _build_completion_kwargs(
        self,
        messages: list[Message],
        *,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
        stream: bool = False,
    ) -> dict:
        """Build kwargs for LiteLLM completion/acompletion calls."""
        kwargs: dict = {
            "model": self.model,
            "messages": self._to_litellm_messages(messages),
            "api_base": os.getenv("LITELLM_API_BASE"),
            "api_key": os.getenv("LITELLM_API_KEY"),
            **self.default_params,
        }

        kwargs["stream"] = stream
        if tools:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format

        return kwargs

    def invoke(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> ChatLLMResponse:
        kwargs = self._build_completion_kwargs(
            messages,
            tools=tools,
            response_format=response_format,
        )
        return to_chatllm_response(completion(**kwargs))

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> ChatLLMResponse:
        kwargs = self._build_completion_kwargs(
            messages,
            tools=tools,
            response_format=response_format,
        )
        return to_chatllm_response(await acompletion(**kwargs))

    def stream(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> AsyncIterator[ChatLLMStreamChunk]:
        async def gen() -> AsyncIterator[ChatLLMStreamChunk]:
            kwargs = self._build_completion_kwargs(
                messages,
                tools=tools,
                response_format=response_format,
                stream=True,
            )

            stream = await acompletion(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                yield ChatLLMStreamChunk(
                    delta=MessageChunk(role="assistant", delta=delta)
                )

        return gen()
