from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel

from ant_ai.core.message import Message
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

        Args:
            messages: Conversation history to send to the model.
            ctx: Invocation context, or None if not available.
            tools: Tool schemas to expose to the model, or None for no tools.
            response_format: Constrain the output to a JSON schema or Pydantic model.

        Returns:
            An async iterator of response chunks.
        """
        raise NotImplementedError
