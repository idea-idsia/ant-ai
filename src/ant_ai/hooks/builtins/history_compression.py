from __future__ import annotations

from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator

from ant_ai.core.message import (
    Message,
    ToolCallMessage,
    ToolCallResultMessage,
    sanitize_messages,
)
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks.protocol import AgentHook

_SUMMARY_PREFIX = "[Conversation summary] "
_SUMMARISE_PROMPT = (
    "Summarise the following conversation history concisely, "
    "preserving key facts, decisions, and context:\n\n"
)


def _message_text(m: Message) -> str:
    """Return a human-readable single-line representation of a message.

    Plain messages use ``role: content``.  Tool calls list function names and
    their arguments so the summariser sees what was invoked, not an empty line.
    Tool results prefix the tool name so the summariser can match call to result.
    """
    if isinstance(m, ToolCallMessage):
        calls = "; ".join(
            f"{tc.function.name}({tc.function.arguments})" for tc in m.tool_calls
        )
        prefix = f"{m.content} " if m.content else ""
        return f"assistant: {prefix}[tool calls]: {calls}"
    if isinstance(m, ToolCallResultMessage):
        return f"tool [{m.name}]: {m.content or ''}"
    return f"{m.role}: {m.content or ''}"


class HistoryCompressionHook(AgentHook, BaseModel):
    """Monitors conversation history in ``State`` and compresses older messages via LLM summarisation when a configurable threshold is exceeded.

    The compressed summary replaces the older messages as a single ``system``
    message, keeping the context window manageable for long-running agents.
    Triggers in ``before_model`` (once per outer loop step, never on retries);
    preserves the most recent ``keep_last`` messages verbatim. The internal
    summarisation LLM call does not go through hooks to avoid infinite recursion.

    At least one of ``max_messages`` or ``max_token_ratio`` must be provided.
    When ``max_token_ratio`` is set, ``context_window`` is also required.

    Example:

        ```python
        hook = HistoryCompressionHook(
            llm=llm,
            max_messages=30,
            max_token_ratio=0.75,
            context_window=128_000,
        )
        agent = Agent(llm=llm, tools=[...], hooks=[hook])
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: ClassVar[str] = "history_compression"
    llm: SkipValidation[Any] = Field(
        description="Language model used to produce the history summary.",
    )
    max_messages: int | None = Field(
        default=None,
        description="Compress when the conversation reaches this many messages.",
    )
    max_token_ratio: float | None = Field(
        default=None,
        description=(
            "Compress when estimated token usage reaches this fraction of ``context_window`` (e.g. ``0.75`` triggers at 75 %)."
        ),
    )
    context_window: int | None = Field(
        default=None,
        description="Total token capacity of the model; required when ``max_token_ratio`` is set.",
    )
    keep_last: int = Field(
        default=4,
        description="Number of most-recent messages always preserved verbatim.",
    )

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Self:
        if self.max_messages is None and self.max_token_ratio is None:
            raise ValueError(
                "At least one of max_messages or max_token_ratio must be set."
            )
        if self.max_token_ratio is not None and self.context_window is None:
            raise ValueError("context_window is required when max_token_ratio is set.")
        return self

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Rough estimate: 1 token ≈ 4 characters."""
        return sum(len(_message_text(m)) for m in messages) // 4

    def _should_compress(self, messages: list[Message]) -> bool:
        if self.max_messages is not None and len(messages) >= self.max_messages:
            return True
        if self.max_token_ratio is not None and self.context_window is not None:
            ratio = self._estimate_tokens(messages) / self.context_window
            if ratio >= self.max_token_ratio:
                return True
        return False

    def _sanitize(self, messages: list[Message]) -> list[Message]:
        return sanitize_messages(messages)

    async def before_model(self, state: State, ctx: InvocationContext | None) -> None:
        """Sanitize history and compress when a threshold is exceeded.

        Args:
            state: Current agent state whose ``messages`` may be compressed.
            ctx: Invocation context, or None if not available.
        """
        state.messages = self._sanitize(state.messages)

        messages: list[Message] = state.messages
        if len(messages) <= self.keep_last or not self._should_compress(messages):
            return

        # messages[:-0] is [] in Python, so handle keep_last=0 ("compress all") explicitly.
        keep_from: int = (
            len(messages) - self.keep_last if self.keep_last > 0 else len(messages)
        )

        # Slide left: never bisect a tool-result group at the boundary.  A tool
        # role message must always be preceded by its ToolCallMessage; slide left
        # until the boundary lands on the call or on a non-tool message.
        # After _sanitize every ToolCallMessage in the list has complete results,
        # so no slide-right is needed.
        while 0 < keep_from < len(messages) and messages[keep_from].role == "tool":
            keep_from -= 1

        to_compress: list[Message] = messages[:keep_from]
        keep: list[Message] = messages[keep_from:]

        # Nothing compressible after boundary adjustment (e.g. history starts with a
        # tool-call group that would be broken by any split).
        if not to_compress:
            return

        history_text: str = "\n".join(_message_text(m) for m in to_compress)
        summary_request: list[Message] = [
            Message(role="user", content=f"{_SUMMARISE_PROMPT}{history_text}")
        ]

        response = await self.llm.ainvoke(summary_request)
        summary = response.message.content or ""

        state.messages = [
            Message(role="system", content=f"{_SUMMARY_PREFIX}{summary}"),
            *keep,
        ]
        # Record the compressed baseline (everything before the current trigger message)
        # so transport layers can persist it for durability across turns.
        # sanitize_messages removes any tool-call group whose result is the trigger
        # (the last message), which would otherwise leave an orphaned ToolCallMessage
        # in the persisted checkpoint and cause a 400 on the next turn.
        state._compression_context = sanitize_messages(list(state.messages[:-1]))
