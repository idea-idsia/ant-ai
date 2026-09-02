from __future__ import annotations

import os
from collections.abc import AsyncIterator

from litellm import ModelResponse, acompletion, completion
from litellm.types.utils import Choices
from pydantic import BaseModel

from ant_ai.core.message import Message, MessageChunk, ToolFunction
from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk, ToolCall
from ant_ai.core.types import InvocationContext, LLMSettings
from ant_ai.llm.params import resolve_llm_params
from ant_ai.llm.protocol import ChatLLM


def to_chatllm_response(
    resp: ModelResponse,
) -> ChatLLMResponse:
    choice: Choices = resp.choices[0]

    message = Message(
        role=choice.message.role,
        content=choice.message.get("content", ""),
    )

    tool_calls: list[ToolCall] = [
        ToolCall(
            id=tc.id,
            function=ToolFunction(
                name=tc.function.name or "",  # ty:ignore[unresolved-attribute]
                arguments=tc.function.arguments or "",  # ty:ignore[unresolved-attribute]
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

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        self.model: str = model
        # Endpoint/credentials: explicit arg wins, else the env default. Resolved
        # once here so all construction-time config lives in one place.
        self.api_base: str | None = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key: str | None = api_key or os.getenv("LITELLM_API_KEY")
        # Typed per-instance baseline, merged under any per-request override.
        self.settings: LLMSettings = settings or LLMSettings()
        # Raw provider long-tail; the escape hatch for anything outside the `LLMSettings`' safe surface.
        self.default_params: dict = {}

    @staticmethod
    def _to_litellm_messages(messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects into LiteLLM-compatible dicts."""
        return [m.model_dump(exclude={"kind"}) for m in messages]

    def _build_completion_kwargs(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
        stream: bool = False,
    ) -> dict:
        """Build kwargs for LiteLLM completion/acompletion calls."""
        kwargs: dict = {
            "model": self.model,
            "messages": self._to_litellm_messages(messages),
            "api_base": self.api_base,
            "api_key": self.api_key,
            **resolve_llm_params(self.default_params, self.settings, ctx),
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
            ctx=ctx,
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
            ctx=ctx,
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
                ctx=ctx,
                tools=tools,
                response_format=response_format,
                stream=True,
            )

            stream = await acompletion(**kwargs)
            async for chunk in stream:
                choice_delta = chunk.choices[0].delta
                delta = choice_delta.content or ""
                reasoning_delta = (
                    getattr(choice_delta, "reasoning_content", None) or None
                )
                if delta or reasoning_delta:
                    yield ChatLLMStreamChunk(
                        delta=MessageChunk(role="assistant", delta=delta),
                        reasoning_delta=reasoning_delta,
                    )

                for tc in getattr(choice_delta, "tool_calls", None) or []:
                    yield ChatLLMStreamChunk(
                        delta=MessageChunk(role="assistant", delta=""),
                        tool_calls={
                            "index": tc.index,
                            "id": tc.id,
                            "name": getattr(tc.function, "name", None),
                            "arguments": getattr(tc.function, "arguments", None) or "",
                        },
                    )

        return gen()
