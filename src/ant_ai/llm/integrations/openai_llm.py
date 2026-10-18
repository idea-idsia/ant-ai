import os
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any, cast

import openai
from openai import AsyncOpenAI, OpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from ant_ai.core.message import Message, MessageChunk, ToolFunction
from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk, ToolCall
from ant_ai.core.types import InvocationContext
from ant_ai.llm.exceptions import ContextWindowExceededError
from ant_ai.llm.protocol import ChatLLM


def _drop_none(**kwargs) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _to_response_format(
    response_format: dict | type[BaseModel] | None,
) -> dict | None:
    """`create()` rejects Pydantic classes, so send them as the JSON schema `parse()` would."""
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return cast(dict, type_to_response_format_param(response_format))
    return response_format


def _to_chatllm_response(response: ChatCompletion) -> ChatLLMResponse:
    message = response.choices[0].message
    tool_calls: list[ToolCall] = [
        ToolCall(
            id=tc.id,
            function=ToolFunction(
                name=tc.function.name, arguments=tc.function.arguments
            ),
        )
        for tc in (message.tool_calls or [])
        if tc.type == "function"
    ]
    return ChatLLMResponse(
        message=Message(role="assistant", content=message.content or ""),
        tool_calls=tool_calls,
        usage=response.usage.model_dump() if response.usage else None,
        reasoning=getattr(message, "reasoning_content", None) or None,
    )


@contextmanager
def _translate_errors(model: str) -> Iterator[None]:
    """Raise the library's own exception for conditions callers may act on."""
    try:
        yield
    except openai.BadRequestError as exc:
        if exc.code == "context_length_exceeded":
            raise ContextWindowExceededError(model) from exc
        raise


class OpenAIChat(ChatLLM):
    """Chat model backed by the OpenAI Python SDK.

    Args:
        model: Any model the endpoint serves (e.g. `"gpt-5-nano"`).
        api_key: Credential for the endpoint. Falls back to the `OPENAI_API_KEY`
            environment variable when not given, so a deployment can keep its
            secret under its own name and pass it here.
        api_base: Endpoint URL, for any OpenAI-compatible server (vLLM, a
            proxy, …). Falls back to `OPENAI_BASE_URL`, then to the SDK's
            default of `https://api.openai.com/v1`.
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.model: str = model
        self.api_key: str | None = (
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        )
        self.api_base: str | None = (
            api_base if api_base is not None else os.getenv("OPENAI_BASE_URL")
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)

    @staticmethod
    def _to_openai_messages(
        messages: list[Message], tools: list | None = None
    ) -> list[ChatCompletionMessageParam]:
        """Converts the internal Message objects into the shape expected by the OpenAI Python SDK. Casting just to make hinters happy."""
        return cast(
            list[ChatCompletionMessageParam],
            [m.to_provider_dict() for m in messages],
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

        with _translate_errors(self.model):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **_drop_none(
                    tools=tools,
                    response_format=_to_response_format(response_format),
                ),
            )
        return _to_chatllm_response(response)

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> ChatLLMResponse:
        openai_messages = self._to_openai_messages(messages)

        with _translate_errors(self.model):
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **_drop_none(
                    tools=tools,
                    response_format=_to_response_format(response_format),
                ),
            )
        return _to_chatllm_response(response)

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
            with _translate_errors(self.model):
                stream = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    stream=True,
                    **_drop_none(
                        tools=tools,
                        response_format=_to_response_format(response_format),
                    ),
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
                                "arguments": getattr(tc.function, "arguments", None)
                                or "",
                            },
                        )

        return gen()
