from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ant_ai.core.events import (
    ContentDeltaEvent,
    FinalAnswerEvent,
    ReasoningEvent,
    ToolCallingEvent,
)
from ant_ai.core.message import ToolCall, ToolFunction
from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    Transition,
    TransitionAction,
)

if TYPE_CHECKING:
    from ant_ai.core.events import BaseEvent
    from ant_ai.core.response import ChatLLMStreamChunk


@dataclass
class _ToolCallFragment:
    id: str | None = None
    name: str | None = None
    arguments: str = ""


class ToolCallAssembler:
    """Joins the streamed fragments of tool calls into whole `ToolCall`s."""

    def __init__(self) -> None:
        self._fragments: dict[int, _ToolCallFragment] = {}
        # Some providers tag every parallel tool call with the same provider-reported index instead of incrementing it, delivering each as one complete chunk (its own id + full arguments). This dict maps those provider indices to the next available index for our own use, so we can yield a separate `ContentDeltaEvent` for each tool call.
        self._index_for: dict[int, int] = {}

    def _starts_new_call(self, idx: int, incoming_id: str | None) -> bool:
        frag: _ToolCallFragment | None = self._fragments.get(idx)
        return (
            frag is not None
            and frag.id is not None
            and bool(incoming_id)
            and incoming_id != frag.id
        )

    def _resolve_index(self, provider_idx: int, incoming_id: str | None) -> int:
        idx: int | None = self._index_for.get(provider_idx)
        if idx is None or self._starts_new_call(idx, incoming_id):
            idx = len(self._fragments)
            self._index_for[provider_idx] = idx
        return idx

    def add(self, fragment: dict[str, Any], stream_id: str) -> ContentDeltaEvent:
        """Add one streamed fragment and return its `ContentDeltaEvent`."""
        incoming_id: str | None = fragment.get("id")
        idx: int = self._resolve_index(fragment["index"], incoming_id)
        frag: _ToolCallFragment = self._fragments.setdefault(idx, _ToolCallFragment())
        is_first: bool = frag.id is None and bool(incoming_id)
        if incoming_id:
            frag.id = incoming_id
        if fragment.get("name"):
            frag.name = fragment["name"]
        args_delta: str = fragment.get("arguments") or ""
        frag.arguments += args_delta
        return ContentDeltaEvent(
            target_kind="tool_calling",
            delta=args_delta,
            stream_id=stream_id,
            is_first=is_first,
            tool_call_index=idx,
            tool_call_id=frag.id if is_first else None,
            tool_call_name=frag.name if is_first else None,
        )

    def tool_calls(self) -> list[ToolCall]:
        """The tool calls assembled so far, in the order they started."""
        return [
            ToolCall(
                id=frag.id or "",
                function=ToolFunction(name=frag.name or "", arguments=frag.arguments),
            )
            for frag in self._fragments.values()
        ]


class StreamedResponse:
    """Accumulates a streamed LLM response, one `ContentDeltaEvent` per fragment."""

    def __init__(self, stream_id: str) -> None:
        self.stream_id: str = stream_id
        self.raw: str = ""
        self.reasoning: str | None = None
        self._tool_calls = ToolCallAssembler()

    def add(self, chunk: ChatLLMStreamChunk) -> list[ContentDeltaEvent]:
        """Accumulate `chunk` and return the events of its fragments."""
        events: list[ContentDeltaEvent] = []
        if chunk.reasoning_delta:
            events.append(
                self._delta("reasoning", chunk.reasoning_delta, self.reasoning is None)
            )
            self.reasoning = (self.reasoning or "") + chunk.reasoning_delta
        if chunk.delta.delta:
            events.append(self._delta("content", chunk.delta.delta, not self.raw))
            self.raw += chunk.delta.delta
        if chunk.tool_calls:
            events.append(self._tool_calls.add(chunk.tool_calls, self.stream_id))
        return events

    def tool_calls(self) -> list[ToolCall]:
        """The tool calls of the response."""
        return self._tool_calls.tool_calls()

    def _delta(
        self, kind: Literal["reasoning", "content"], delta: str, is_first: bool
    ) -> ContentDeltaEvent:
        return ContentDeltaEvent(
            target_kind=kind,
            delta=delta,
            stream_id=self.stream_id,
            is_first=is_first,
        )


def _terminal_event(
    raw: str, tool_calls: list[ToolCall], stream_id: str | None
) -> tuple[ToolCallingEvent | FinalAnswerEvent, Transition]:
    if tool_calls:
        return (
            ToolCallingEvent(
                content=raw, tool_calls=tuple(tool_calls), stream_id=stream_id
            ),
            Transition(action=TransitionAction.CONTINUE, next_step="tool"),
        )
    return (
        FinalAnswerEvent(content=raw, stream_id=stream_id),
        Transition(action=TransitionAction.END),
    )


def response_items(
    raw: str,
    tool_calls: list[ToolCall],
    reasoning: str | None,
    stream_id: str | None,
) -> Iterator[BaseEvent | StepResult]:
    """Yield the optional `ReasoningEvent`, terminal event, and `StepResult` of a
    fully collected response.
    """
    if reasoning:
        yield ReasoningEvent(content=reasoning, stream_id=stream_id)

    event, transition = _terminal_event(raw, tool_calls, stream_id=stream_id)
    yield event
    yield StepResult(
        output=LLMOutput(raw=raw, tool_calls=tuple(tool_calls)),
        transition=transition,
    )
