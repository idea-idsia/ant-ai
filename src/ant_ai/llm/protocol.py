from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel

from ant_ai.core.message import Message, MessageChunk
from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk
from ant_ai.core.types import InvocationContext


class ChatLLM(Protocol):
    """Interface for a language model that generates chat responses."""

    def invoke(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> ChatLLMResponse:
        """Send messages and return a complete response synchronously.

        Args:
            messages: Conversation history to send to the model.
            ctx: Invocation context, or None if not available.
            tools: Tool schemas to expose to the model, or None for no tools.
            response_format: Constrain the output to a JSON schema or Pydantic model.

        Returns:
            The complete model response.
        """
        raise NotImplementedError

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> ChatLLMResponse:
        """Send messages and return a complete response asynchronously.

        Args:
            messages: Conversation history to send to the model.
            ctx: Invocation context, or None if not available.
            tools: Tool schemas to expose to the model, or None for no tools.
            response_format: Constrain the output to a JSON schema or Pydantic model.

        Returns:
            The complete model response.
        """
        raise NotImplementedError

    def stream(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        tools: list | None = None,
        response_format: dict | type[BaseModel] | None = None,
    ) -> AsyncIterator[ChatLLMStreamChunk]:
        """Send messages and stream the response as chunks.

        Backends that generate tokens incrementally (e.g. OpenAI, LiteLLM)
        override this for true token-by-token delivery. The default here
        falls back to `ainvoke()` and re-emits its result as a single chunk,
        so any `ChatLLM` implementation — including test doubles and
        backends with no incremental API — supports `.stream()` without
        extra work, just without the live granularity.

        Args:
            messages: Conversation history to send to the model.
            ctx: Invocation context, or None if not available.
            tools: Tool schemas to expose to the model, or None for no tools.
            response_format: Constrain the output to a JSON schema or Pydantic model.

        Returns:
            An async iterator of response chunks.
        """

        async def gen() -> AsyncIterator[ChatLLMStreamChunk]:
            response: ChatLLMResponse = await self.ainvoke(
                messages, ctx=ctx, tools=tools, response_format=response_format
            )
            reasoning = getattr(response, "reasoning", None)
            if response.message.content or reasoning:
                yield ChatLLMStreamChunk(
                    delta=MessageChunk(
                        role=response.message.role,
                        delta=response.message.content or "",
                    ),
                    reasoning_delta=reasoning,
                )
            for i, tool_call in enumerate(response.tool_calls or []):
                yield ChatLLMStreamChunk(
                    delta=MessageChunk(role="assistant", delta=""),
                    tool_calls={
                        "index": i,
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                )

        return gen()
