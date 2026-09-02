from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ant_ai.core.message import AnyMessage, Message


class LLMSettings(BaseModel):
    """Per-request overrides for a single LLM completion call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: Annotated[float, Field(ge=0.0, le=2.0)] | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None

    def overrides(self) -> dict[str, Any]:
        """The kwargs the caller actually set; fields left unset are omitted."""
        return self.model_dump(exclude_none=True)


class InvocationContext(BaseModel):
    """
    Request-scoped execution context. Treat as read-only during a request.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_id: str | None = Field(default=None)
    llm_settings: LLMSettings | None = Field(default=None)
    workflow_settings: dict[str, Any] | None = Field(default=None)


class State(BaseModel):
    """Shared mutable state passed through agent steps and workflow nodes.

    Subclass to add domain-specific fields:

        class MyState(State):
            user_id: str = ""
    """

    messages: list[Message] = Field(default_factory=list)
    artefacts: list[Any] = Field(default_factory=list)
    _compression_context: list[AnyMessage] | None = PrivateAttr(default=None)

    @property
    def last_message(self) -> Message:
        """Returns the last message in the conversation, if any."""
        if not self.messages:
            raise ValueError("No messages in conversation")
        return self.messages[-1]

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
