from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ant_ai.observer import obs

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from ant_ai.core.message import BaseMessage
    from ant_ai.llm.protocol import ChatLLM


def generation_span(
    llm: ChatLLM,
    llm_input: list[BaseMessage],
    metadata: dict[str, Any],
) -> AbstractAsyncContextManager[Any]:
    """The span of one call of `llm` on `llm_input`."""
    return obs.span(
        getattr(llm, "model", "llm"),
        as_type="generation",
        model=getattr(llm, "model", None),
        input=llm_input,
        metadata=metadata,
    )


def tool_span(
    tool_name: str, args: dict[str, Any], tool_call_id: str
) -> AbstractAsyncContextManager[Any]:
    """The span of one tool call."""
    return obs.span(
        tool_name,
        as_type="tool",
        input=args,
        metadata={"tool_call_id": tool_call_id},
    )
