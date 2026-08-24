from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ant_ai.core.events import (
    ContentDeltaEvent,
    FinalAnswerEvent,
    ReasoningEvent,
    ToolCallingEvent,
)
from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.response import ChatLLMResponse
from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    Transition,
    TransitionAction,
)
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.protocol import ChatLLM
from ant_ai.observer import obs


@dataclass
class _ToolCallFragment:
    id: str | None = None
    name: str | None = None
    arguments: str = ""


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


class LLMStep(BaseModel):
    """Invokes the language model and wraps the response in a `StepResult[LLMOutput]`.

    Emits a `ToolCallingEvent` and routes to `"tool"` when the model requests
    tool calls, or emits a `FinalAnswerEvent` and ends the loop otherwise.

    Mirrors `ChatLLM`'s own `ainvoke`/`stream` split: `run()` is the whole-response
    path via `.ainvoke()`, unchanged regardless of streaming; `stream()` is a
    separate method that calls the LLM's `.stream()` API and also emits live
    `ContentDeltaEvent`s as the model generates. Callers (the agent loop)
    choose which method to invoke based on whether live streaming is safe and
    wanted for that call — this class has no streaming flag of its own.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "llm"

    llm: SkipValidation[ChatLLM]
    system_message: Message
    serialized_tools: list[dict] = Field(default_factory=list)

    response_format: type[BaseModel] | None = Field(default=None, exclude=True)

    def _build_llm_input(self, state: State) -> list[Message]:
        return [self.system_message, *state.messages]

    def _generation_span(self, llm_input: list[Message], state: State):
        return obs.span(
            getattr(self.llm, "model", "llm"),
            as_type="generation",
            model=getattr(self.llm, "model", None),
            input=llm_input,
            metadata={
                "message_count": len(state.messages),
                "tool_count": len(self.serialized_tools),
                "has_response_format": self.response_format is not None,
            },
        )

    async def _finish(
        self,
        raw: str,
        tool_calls: list[ToolCall],
        reasoning: str | None,
        stream_id: str | None,
    ):
        """Yields the optional `ReasoningEvent`, terminal event, and `StepResult`
        shared by `run()` and `stream()` once a response has been fully collected.
        """
        if reasoning:
            yield ReasoningEvent(content=reasoning, stream_id=stream_id)

        event, transition = _terminal_event(raw, tool_calls, stream_id=stream_id)
        yield event
        yield StepResult(
            output=LLMOutput(raw=raw, tool_calls=tuple(tool_calls)),
            transition=transition,
        )

    async def run(
        self,
        state: State,
        ctx: InvocationContext | None,
    ):
        llm_input: list[Message] = self._build_llm_input(state)

        async with self._generation_span(llm_input, state) as span:
            response: ChatLLMResponse = await self.llm.ainvoke(
                llm_input,
                ctx=ctx,
                tools=self.serialized_tools or None,
                response_format=self.response_format,
            )

            raw: str = response.message.content or ""
            tool_calls: list[ToolCall] = response.tool_calls or []

            update_payload: dict[str, object] = {
                "output": raw,
                "metadata": {
                    "tool_call_count": len(tool_calls),
                },
            }

            response_model = getattr(response, "model", None)
            if response_model is not None:
                update_payload["model"] = response_model

            usage_details = getattr(response, "usage", None)
            if usage_details is not None:
                update_payload["usage"] = usage_details

            span.update(**update_payload)

        reasoning = getattr(response, "reasoning", None)
        async for item in self._finish(raw, tool_calls, reasoning, stream_id=None):
            yield item

    async def stream(
        self,
        state: State,
        ctx: InvocationContext | None,
    ):
        """Live token-level counterpart to `run()`.

        Calls the LLM's `.stream()` API, yielding `ContentDeltaEvent`s as
        fragments arrive, then the same terminal event/`StepResult` sequence
        as `run()` would produce for the same response, correlated via
        `stream_id`.
        """
        llm_input: list[Message] = self._build_llm_input(state)
        stream_id = str(uuid4())

        async with self._generation_span(llm_input, state) as span:
            raw, tool_calls, reasoning = "", [], None
            async for (
                delta_event,
                raw_frag,
                reasoning_frag,
                tool_call,
            ) in self._stream_deltas(llm_input, ctx, stream_id):
                if delta_event is not None:
                    yield delta_event
                raw += raw_frag
                if reasoning_frag:
                    reasoning = (reasoning or "") + reasoning_frag
                if tool_call is not None:
                    tool_calls.append(tool_call)

            # ChatLLMStreamChunk carries no usage/model fields, unlike
            # ChatLLMResponse, so those span attributes are unavailable here.
            span.update(output=raw, metadata={"tool_call_count": len(tool_calls)})

        async for item in self._finish(raw, tool_calls, reasoning, stream_id=stream_id):
            yield item

    async def _stream_deltas(
        self,
        llm_input: list[Message],
        ctx: InvocationContext | None,
        stream_id: str,
    ):
        """Consume the LLM's stream, yielding (delta_event, raw, reasoning, tool_call) tuples.

        `raw`/`reasoning` are the fragment to accumulate for this chunk (usually
        empty); `tool_call` is a finished `ToolCall` once its argument stream
        closes (detected by a change of tool-call index or end of stream), None
        otherwise. Fan-in for the final index happens in `stream()` once the
        loop below and this generator both finish.
        """
        reasoning_started = False
        content_started = False
        tool_frags: dict[int, _ToolCallFragment] = {}

        async for chunk in self.llm.stream(
            messages=llm_input,
            ctx=ctx,
            tools=self.serialized_tools or None,
            response_format=self.response_format,
        ):
            if chunk.reasoning_delta:
                yield (
                    ContentDeltaEvent(
                        target_kind="reasoning",
                        delta=chunk.reasoning_delta,
                        stream_id=stream_id,
                        is_first=not reasoning_started,
                    ),
                    "",
                    chunk.reasoning_delta,
                    None,
                )
                reasoning_started = True

            if chunk.delta.delta:
                yield (
                    ContentDeltaEvent(
                        target_kind="content",
                        delta=chunk.delta.delta,
                        stream_id=stream_id,
                        is_first=not content_started,
                    ),
                    chunk.delta.delta,
                    "",
                    None,
                )
                content_started = True

            if chunk.tool_calls:
                idx = chunk.tool_calls["index"]
                frag: _ToolCallFragment = tool_frags.setdefault(
                    idx, _ToolCallFragment()
                )
                is_first_frag = frag.id is None and bool(chunk.tool_calls.get("id"))
                if chunk.tool_calls.get("id"):
                    frag.id = chunk.tool_calls["id"]
                if chunk.tool_calls.get("name"):
                    frag.name = chunk.tool_calls["name"]
                args_delta = chunk.tool_calls.get("arguments") or ""
                frag.arguments += args_delta
                yield (
                    ContentDeltaEvent(
                        target_kind="tool_calling",
                        delta=args_delta,
                        stream_id=stream_id,
                        is_first=is_first_frag,
                        tool_call_index=idx,
                        tool_call_id=frag.id if is_first_frag else None,
                        tool_call_name=frag.name if is_first_frag else None,
                    ),
                    "",
                    "",
                    None,
                )

        for frag in tool_frags.values():
            yield (
                None,
                "",
                "",
                ToolCall(
                    id=frag.id or "",
                    function=ToolFunction(
                        name=frag.name or "", arguments=frag.arguments
                    ),
                ),
            )
