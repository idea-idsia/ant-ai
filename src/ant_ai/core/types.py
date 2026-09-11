from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ant_ai.core.message import AnyMessage, Message


class InvocationContext(BaseModel):
    """Request-scoped execution context. Treat as read-only during a request.

    Subclass to carry deployment-specific fields through a run -- tools that
    declare a `ctx` parameter receive the instance, and the A2A/ACP entry
    points build it for you via `from_metadata`:

        class MyContext(InvocationContext):
            tenant: str = ""
            tags: list[str] | None = None

            def trace_attributes(self) -> dict[str, Any]:
                return {**super().trace_attributes(), "tags": self.tags}

        A2AServer(..., context_class=MyContext)
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_id: str | None = Field(default=None)
    llm_settings: dict[str, Any] | None = Field(default=None)
    workflow_settings: dict[str, Any] | None = Field(default=None)

    @classmethod
    def from_metadata(
        cls, *, session_id: str, metadata: Mapping[str, Any] | None = None
    ) -> Self:
        """Build a context from request metadata (e.g. an A2A message's `metadata`).

        Fields are matched by name -- a `user_id` key fills `user_id`, and any field
        a subclass declares is filled the same way. Unknown keys are ignored.
        `session_id` always comes from the argument, never from the metadata.
        Override to map a different wire shape onto your fields.
        """
        return cls.model_validate({**(metadata or {}), "session_id": session_id})

    def trace_attributes(self) -> dict[str, Any]:
        """Fields bound to the run's trace and sent with `workflow.start`.

        Sinks read these by name: the Langfuse sink, for instance, forwards
        `session_id`, `user_id` and `tags`. Override to add your own; keep
        secrets out, since these end up in whatever observability backend is
        configured.
        """
        return {"session_id": self.session_id, "user_id": self.user_id}


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
