from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ant_ai.core.message import Message, MessageChunk
from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk
from ant_ai.core.types import InvocationContext
from ant_ai.llm.protocol import ChatLLM


def _drop_none(**kwargs) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


class OpenAIChat(ChatLLM):
    """
    Interface for a language model that generates chat responses using OpenAI's API.
    """

    def __init__(self, model: str = "gpt-5-nano", api_key: str | None = None):
        self.model: str = model
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

    @staticmethod
    def _to_openai_messages(
        messages: list[Message], tools: list | None = None
    ) -> list[ChatCompletionMessageParam]:
        """Converts the internal Message objects into the shape expected by the OpenAI Python SDK. Casting just to make hinters happy."""
        return cast(
            list[ChatCompletionMessageParam],
            [m.model_dump(exclude={"kind"}) for m in messages],
        )

    def invoke(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> ChatLLMResponse:
        openai_messages = self._to_openai_messages(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            **_drop_none(
                tools=tools,
                response_format=response_format,
            ),
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
        openai_messages = self._to_openai_messages(messages)

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            **_drop_none(
                tools=tools,
                response_format=response_format,
            ),
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
        openai_messages = self._to_openai_messages(messages)

        async def gen() -> AsyncIterator[ChatLLMStreamChunk]:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                stream=True,
                **_drop_none(
                    tools=tools,
                    response_format=response_format,
                ),
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue

                yield ChatLLMStreamChunk(
                    delta=MessageChunk(role="assistant", delta=delta)
                )

        return gen()
