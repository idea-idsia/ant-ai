from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ant_ai.core.message import Message


class InvocationContext(BaseModel):
    """
    Request-scoped execution context. Treat as read-only during a request.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    llm_settings: dict[str, Any] | None = Field(default=None)
    workflow_settings: dict[str, Any] | None = Field(default=None)


class State(BaseModel):
    """Shared mutable state passed through agent steps and workflow nodes.

    Subclass to add domain-specific fields:

        class MyState(State):
            user_id: str = ""
    """

    messages: list[Message] = Field(default_factory=list)
    artefacts: list[Any] = Field(default_factory=list)

    @property
    def last_message(self) -> Message:
        """Returns the last message in the conversation, if any."""
        if not self.messages:
            raise ValueError("No messages in conversation")
        return self.messages[-1]

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
