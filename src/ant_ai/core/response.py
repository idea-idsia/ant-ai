from pydantic import BaseModel, Field

from ant_ai.core.message import Message, MessageChunk, ToolCall


class ChatLLMResponse(BaseModel):
    """Response from a chat-based LLM. It follows a OpenAI-like structure."""

    message: Message = Field(
        description="The assistant message returned by the model.",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls requested by the model. Empty when the model produced a text answer.",
    )
    finish_reason: str | None = Field(
        default=None,
        description="The reason the model stopped generating (e.g. 'stop', 'tool_calls').",
    )
    usage: dict | None = Field(
        default=None,
        description="Token usage statistics reported by the model.",
    )
    reasoning: str | None = Field(
        default=None,
        description="Reasoning/thinking content produced by the model, if supported.",
    )


class ChatLLMStreamChunk(BaseModel):
    """A chunk of a streaming response from a chat-based LLM."""

    delta: MessageChunk = Field(
        description="Newly streamed text fragment, not the accumulated response so far.",
    )
    tool_calls: dict | None = Field(
        default=None,
        description="Partial tool call data streamed incrementally alongside the delta.",
    )
