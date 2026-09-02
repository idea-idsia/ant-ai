from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ant_ai.core.message import Message, MessageChunk
from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk
from ant_ai.core.types import InvocationContext, LLMSettings
from ant_ai.llm.params import resolve_llm_params
from ant_ai.llm.protocol import ChatLLM


def _drop_none(**kwargs) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


class OpenAIChat(ChatLLM):
    """
    Interface for a language model that generates chat responses using OpenAI's API.
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: str | None = None,
        *,
        settings: LLMSettings | None = None,
    ):
        self.model: str = model
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        # Typed per-instance baseline, merged under any per-request override.
        self.settings: LLMSettings = settings or LLMSettings()
        # Raw provider long-tail; the escape hatch for anything outside the `LLMSettings`' safe surface.
        self.default_params: dict[str, Any] = {}

    @staticmethod
    def _to_openai_messages(
        messages: list[Message], tools: list | None = None
    ) -> list[ChatCompletionMessageParam]:
        """Converts the internal Message objects into the shape expected by the OpenAI Python SDK. Casting just to make hinters happy."""
        return cast(
            list[ChatCompletionMessageParam],
            [m.model_dump(exclude={"kind"}) for m in messages],
        )

    def _build_kwargs(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None,
        tools: list | None,
        response_format: dict | type[BaseModel] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Assemble kwargs for the OpenAI chat-completions call."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages),
            **resolve_llm_params(self.default_params, self.settings, ctx),
            **_drop_none(tools=tools, response_format=response_format),
        }
        if stream:
            kwargs["stream"] = True
        return kwargs

    def invoke(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> ChatLLMResponse:
        response = self.client.chat.completions.create(
            **self._build_kwargs(
                messages, ctx=ctx, tools=tools, response_format=response_format
            )
        )
        content = response.choices[0].message.content or ""
        return ChatLLMResponse(message=Message(role="assistant", content=content))

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> ChatLLMResponse:
        response = await self.async_client.chat.completions.create(
            **self._build_kwargs(
                messages, ctx=ctx, tools=tools, response_format=response_format
            )
        )
        content = response.choices[0].message.content or ""
        return ChatLLMResponse(message=Message(role="assistant", content=content))

    def stream(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> AsyncIterator[ChatLLMStreamChunk]:
        async def gen() -> AsyncIterator[ChatLLMStreamChunk]:
            stream = await self.async_client.chat.completions.create(
                **self._build_kwargs(
                    messages,
                    ctx=ctx,
                    tools=tools,
                    response_format=response_format,
                    stream=True,
                )
            )

            async for chunk in stream:
                choice_delta = chunk.choices[0].delta
                delta = choice_delta.content
                if delta:
                    yield ChatLLMStreamChunk(
                        delta=MessageChunk(role="assistant", delta=delta)
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
