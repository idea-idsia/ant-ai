from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from ant_ai.core.events import Event
from ant_ai.core.exceptions import HookBlockedError, HookMaxRetriesError
from ant_ai.core.message import Message
from ant_ai.core.result import LLMOutput, StepResult
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks import (
    HookLayer,
    PostModelBlock,
    PostModelDecision,
    PostModelFallback,
    PostModelPass,
    PostModelRetry,
    WrapCall,
)
from ant_ai.memory.protocol import Memory
from ant_ai.observer import obs
from ant_ai.steps import Step
from ant_ai.tools.registry import ToolRegistry


class BaseAgentLoop(ABC, BaseModel):
    """
    Base class for all agent loop implementations.

    Provides hook integration, retry/critique machinery, and observability
    helpers. Subclasses implement `stream()` to define their reasoning strategy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hooks: HookLayer = Field(default_factory=HookLayer)
    max_retries: int = Field(default=3, ge=1)
    memory: Memory | None = None
    streaming: bool = Field(
        default=False,
        description="Forward LLM token deltas live instead of buffering whole events, when no registered hook needs the complete response.",
    )

    def _streaming_active(self) -> bool:
        """Whether this run should stream deltas live.

        Only true when the caller opted in AND no registered hook overrides
        `after_model`/`wrap_model_call` beyond the no-op default — such a
        hook needs the complete response to decide, and tokens already
        streamed to a client cannot be retracted.
        """
        return self.streaming and self.hooks.is_stream_safe()

    async def _retrieve_memories(
        self, state: State, ctx: InvocationContext | None
    ) -> None:
        """Retrieve relevant memories from the store and prepend them to state."""
        if self.memory is None:
            return
        user_query: str | None = next(
            (m.content for m in reversed(state.messages) if m.role == "user"), ""
        )
        if not user_query:
            return
        memory_msgs: list[Message] = await self.memory.retrieve(user_query, ctx=ctx)
        state.messages[0:0] = memory_msgs

    async def _consolidate_memories(
        self,
        messages: list[Message | None],
        ctx: InvocationContext | None,
    ) -> None:
        """Persist the current-turn exchange to memory."""
        if self.memory is None:
            return
        to_store: list[Message] = [m for m in messages if m is not None and m.content]
        if not to_store:
            return
        await self.memory.update(to_store, ctx=ctx)

    @abstractmethod
    def stream(
        self,
        state: State,
        ctx: InvocationContext | None,
        *,
        max_steps: int = 10,
        response_schema: type[BaseModel] | None = None,
    ) -> AsyncIterator[Event]: ...

    @abstractmethod
    def register_tool(self, registry: ToolRegistry) -> None:
        """Update internal steps to reflect a newly registered tool in registry."""
        ...

    async def _run_model_with_hooks(
        self,
        step: Step,
        state: State,
        ctx: InvocationContext | None,
        *,
        allow_streaming: bool = True,
    ) -> AsyncIterator[Event | StepResult]:
        """Yields buffered events then StepResult, always going through the hook path.

        When streaming is active and allowed for this call, invokes the
        step's `stream()` method (live token deltas) instead of `run()`
        (whole-response), mirroring `ChatLLM`'s own `ainvoke`/`stream` split.
        `allow_streaming=False` forces the whole-response path regardless of
        `_streaming_active()` — used for steps whose output may be silently
        rewritten afterward (e.g. structured-output coercion), where
        streaming live would be misleading.
        """
        use_stream = allow_streaming and self._streaming_active()
        run_call: WrapCall = (
            getattr(step, "stream", step.run) if use_stream else step.run
        )
        wrapped: WrapCall = self.hooks.wrap_model_call(run_call)
        if use_stream:
            async for item in self._stream_wrapped(step, wrapped, state, ctx):
                yield item
            return
        events, result = await self._consume_wrapped(step, wrapped, state, ctx)
        events, result = await self._apply_hooks(
            step, wrapped, state, ctx, events, result
        )
        for event in events:
            yield event
        yield result

    async def _stream_wrapped(
        self,
        step: Step,
        wrapped_run: WrapCall,
        state: State,
        ctx: InvocationContext | None,
    ) -> AsyncIterator[Event | StepResult]:
        """Forward a step's events live, without buffering, on the hook-safe path.

        Still runs `after_model` for lifecycle-contract correctness, but the
        decision must be PostModelPass: a retry/block/fallback verdict here
        would mean `is_stream_safe()` was wrong after tokens were already
        irrevocably sent to the caller, which must fail loudly rather than
        silently drop or duplicate output.
        """
        result: StepResult | None = None
        async for item in self._observe_step(step, wrapped_run(state, ctx)):
            if isinstance(item, StepResult):
                result = item
            else:
                yield item

        if result is None:
            raise RuntimeError("Step produced no result")

        decision: PostModelDecision = await self.hooks.run_after_model(result, ctx)
        if not isinstance(decision, PostModelPass):
            raise RuntimeError(
                "Hook requested retry/block/fallback on a streamed response "
                "after tokens were already forwarded to the caller. This "
                "indicates HookLayer.is_stream_safe() incorrectly reported "
                "the hook chain as stream-safe."
            )
        yield decision.result

    async def _consume_wrapped(
        self,
        step: Step,
        wrapped_run: WrapCall,
        state: State,
        ctx: InvocationContext | None,
    ) -> tuple[list[Event], StepResult]:
        """Consume a wrapped step callable, buffering all events and extracting the StepResult."""
        events: list[Event] = []
        result: StepResult | None = None

        async for item in self._observe_step(step, wrapped_run(state, ctx)):
            if isinstance(item, StepResult):
                result: StepResult = item
            else:
                events.append(item)

        if result is None:
            raise RuntimeError("Step produced no result")
        return events, result

    async def _apply_hooks(
        self,
        step: Step,
        wrapped_run: WrapCall,
        state: State,
        ctx: InvocationContext | None,
        buffered_events: list[Event],
        result: StepResult,
    ) -> tuple[list[Event], StepResult]:
        """
        Run ``after_model`` hooks and handle decisions.

        On RETRY: if the budget is not yet spent, append the failed response
        and critique to a running retry state and re-run the model (bypassing
        ``before_model``). When ``retries_used >= max_retries``, raise
        ``HookMaxRetriesError`` immediately — no separate post-loop check.
        On BLOCK: raise ``HookBlockedError`` immediately.
        On FALLBACK: return the pre-built result directly.
        """
        current_events: list[Event] = buffered_events
        current_result: StepResult = result
        retry_state: State = state.model_copy(deep=True)
        retries_used: int = 0

        while True:
            decision: PostModelDecision = await self.hooks.run_after_model(
                current_result, ctx
            )
            match decision:
                case PostModelPass(result=r):
                    return current_events, r
                case PostModelFallback(result=r):
                    return [], r
                case PostModelBlock(reason=reason):
                    raise HookBlockedError(f"Hook blocked response: {reason}")
                case PostModelRetry(reason=reason):
                    if retries_used >= self.max_retries:
                        raise HookMaxRetriesError(
                            f"Hook still failing after {self.max_retries} repair attempts. "
                            f"Last reason: {reason}"
                        )
                    critique = (
                        reason or "Your previous response was invalid. Please revise."
                    )
                    current_events, current_result = await self._retry_with_critique(
                        step, wrapped_run, retry_state, current_result, critique, ctx
                    )
                    retries_used += 1

    async def _retry_with_critique(
        self,
        step: Step,
        wrapped_run: WrapCall,
        retry_state: State,
        previous_result: StepResult,
        critique: str,
        ctx: InvocationContext | None,
    ) -> tuple[list[Event], StepResult]:
        """Append the failed response and critique to retry_state, then re-run.

        Mutates retry_state in-place so each successive retry sees the full
        correction history: failed_0, critique_0, failed_1, critique_1, ...
        """
        if not isinstance(previous_result.output, LLMOutput):
            raise TypeError(
                f"Expected LLMOutput for retry, got {type(previous_result.output).__name__}"
            )
        retry_state.add_message(
            Message(role="assistant", content=previous_result.output.raw)
        )
        retry_state.add_message(
            Message(role="user", content=f"[Validation feedback]\n{critique}")
        )

        return await self._consume_wrapped(step, wrapped_run, retry_state, ctx)

    async def _observe_step(
        self,
        step: Step,
        items: AsyncIterator[Event | StepResult],
    ) -> AsyncIterator[Event | StepResult]:
        result: StepResult | None = None

        await obs.event("step.start", step=step.name)
        try:
            async for item in items:
                if isinstance(item, StepResult):
                    result: StepResult = item
                yield item
        except Exception as exc:
            await obs.exception("step.error", exc, step=step.name)
            raise

        if result is None:
            raise RuntimeError("Step produced no result")
        await obs.event(
            "step.end",
            step=step.name,
            transition=str(result.transition.action),
        )
