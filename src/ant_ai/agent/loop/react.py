from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ant_ai.agent.loop.loop import BaseAgentLoop
from ant_ai.agent.loop.structured import structuring_request
from ant_ai.agent.loop.transcript import (
    final_answer_event,
    max_steps_event,
    record_answer,
    record_tool_calls,
    record_tool_results,
)
from ant_ai.core.events import FinalAnswerEvent
from ant_ai.core.result import LLMOutput, StepResult, ToolOutput, TransitionAction
from ant_ai.steps import LLMStep, ToolStep

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from ant_ai.core.events import BaseEvent
    from ant_ai.core.types import InvocationContext, State
    from ant_ai.hooks import WrapCall
    from ant_ai.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolRequest:
    """LLM produced tool calls and act_step is configured — run the tools."""


@dataclass(frozen=True)
class FinalResponse:
    """LLM produced a final text answer, end the loop."""


@dataclass(frozen=True)
class _Plan:
    """How the iterations of one `stream()` call run the reason step."""

    active_step: LLMStep
    coerce_schema: type[BaseModel] | None
    max_steps: int


@dataclass
class _Turn:
    """What one reason-act iteration produced."""

    loop_step: int
    result: StepResult | None = None
    stream_id: str | None = None
    done: bool = False

    def reply(self) -> tuple[StepResult, LLMOutput]:
        """The LLM step's result and its output."""
        if self.result is None:
            raise RuntimeError("LLM step produced no result")
        if not isinstance(self.result.output, LLMOutput):
            raise TypeError(
                f"Expected LLMOutput, got {type(self.result.output).__name__}"
            )
        return self.result, self.result.output


class ReActLoop(BaseAgentLoop):
    """
    Runs the default ReAct loop.
    """

    reason_step: LLMStep
    act_step: ToolStep | None = None

    async def stream(
        self,
        state: State,
        ctx: InvocationContext | None,
        *,
        max_steps: int = 10,
        response_schema: type[BaseModel] | None = None,
    ) -> AsyncIterator[BaseEvent]:
        plan: _Plan = self._plan(max_steps, response_schema)
        for loop_step in range(1, max_steps + 1):
            turn = _Turn(loop_step=loop_step)
            async for event in self._turn(plan, turn, state, ctx):
                yield event
            if turn.done:
                return

        yield max_steps_event(max_steps)

    def _plan(self, max_steps: int, response_schema: type[BaseModel] | None) -> _Plan:
        return _Plan(
            active_step=self.reason_step.model_copy(
                update={"response_format": response_schema}
            )
            if self.act_step is None and response_schema
            else self.reason_step,
            coerce_schema=response_schema if self.act_step is not None else None,
            max_steps=max_steps,
        )

    def _step_for(self, plan: _Plan, loop_step: int) -> LLMStep:
        # On the final iteration with coerce_schema, strip tools and apply
        # response_format so the LLM synthesizes a structured final answer
        # from the full accumulated context — no extra LLM call needed.
        if loop_step == plan.max_steps and plan.coerce_schema is not None:
            return self.reason_step.model_copy(
                update={"response_format": plan.coerce_schema, "serialized_tools": []}
            )
        return plan.active_step

    async def _turn(
        self,
        plan: _Plan,
        turn: _Turn,
        state: State,
        ctx: InvocationContext | None,
    ) -> AsyncIterator[BaseEvent]:
        await self.hooks.run_before_model(state, ctx)
        step_to_run: LLMStep = self._step_for(plan, turn.loop_step)

        # Whenever coerce_schema is set, any final response this iteration
        # produces goes through _make_final_answer -> _coerce_to_schema,
        # which may silently rewrite the raw text via a second LLM call —
        # not just on the forced last step, but on an earlier iteration
        # where the model answers directly instead of calling a tool. So
        # streaming must stay off for the whole invocation in that case,
        # not just on the forced-final step.
        async for event in self._reason(
            step_to_run, state, ctx, turn, allow_streaming=plan.coerce_schema is None
        ):
            yield event

        llm_result, output = turn.reply()
        match self._classify_llm_result(llm_result):
            case ToolRequest():
                async for event in self._act(output, state, ctx, turn):
                    yield event
            case FinalResponse():
                final_event: FinalAnswerEvent = await self._make_final_answer(
                    output.raw,
                    turn.loop_step,
                    plan.coerce_schema,
                    ctx,
                    stream_id=turn.stream_id,
                )
                record_answer(state, final_event.content)
                turn.done = True
                yield final_event

    async def _reason(
        self,
        step: LLMStep,
        state: State,
        ctx: InvocationContext | None,
        turn: _Turn,
        *,
        allow_streaming: bool,
    ) -> AsyncIterator[BaseEvent]:
        async for item in self._run_model_with_hooks(
            step, state, ctx, allow_streaming=allow_streaming
        ):
            if isinstance(item, StepResult):
                turn.result = item
            elif isinstance(item, FinalAnswerEvent):
                # LLMStep emits a FinalAnswerEvent before its StepResult as a side-effect.
                # We emit the real FinalAnswerEvent below after optional schema handling,
                # but keep its stream_id so a streamed answer's A2A artifact can be closed.
                turn.stream_id = item.stream_id
            else:
                yield item

    async def _act(
        self,
        output: LLMOutput,
        state: State,
        ctx: InvocationContext | None,
        turn: _Turn,
    ) -> AsyncIterator[BaseEvent]:
        if self.act_step is None:
            raise RuntimeError(
                "LLM requested tool calls but no tools are configured on this agent."
            )
        record_tool_calls(state, output)
        wrapped_tools: WrapCall = self.hooks.wrap_tool_call(self.act_step.run)
        act_result: StepResult | None = None
        async for item in self._observe_step(self.act_step, wrapped_tools(state, ctx)):
            if isinstance(item, StepResult):
                act_result = item
            else:
                yield item
        if act_result is None:
            raise RuntimeError("Tool step produced no result")
        if not isinstance(act_result.output, ToolOutput):
            raise TypeError(
                f"Expected ToolOutput, got {type(act_result.output).__name__}"
            )
        record_tool_results(state, act_result.output)
        if act_result.transition.action == TransitionAction.END:
            # A tool asked for human input. The transcript is complete
            # up to the question; the caller resumes by adding the
            # user's reply and running again.
            turn.done = True

    def _classify_llm_result(self, result: StepResult) -> ToolRequest | FinalResponse:
        """Classify what the loop should do next based on the LLM step result."""
        if not isinstance(result.output, LLMOutput):
            raise TypeError(f"Expected LLMOutput, got {type(result.output).__name__}")
        has_tool_calls: bool = result.output.has_tool_calls
        is_continue: bool = result.transition.action != TransitionAction.END
        if is_continue and has_tool_calls:
            if self.act_step is None:
                raise RuntimeError(
                    "LLM requested tool calls but no tools are configured on this agent."
                )
            return ToolRequest()
        return FinalResponse()

    async def _make_final_answer(
        self,
        text: str,
        loop_step: int,
        response_schema: type[BaseModel] | None,
        ctx: InvocationContext | None,
        *,
        stream_id: str | None = None,
    ) -> FinalAnswerEvent:
        final_text: str = text
        if response_schema is not None:
            final_text: str = await self._coerce_to_schema(text, response_schema, ctx)
        return final_answer_event(final_text, loop_step, stream_id)

    async def _coerce_to_schema(
        self,
        text: str,
        schema: type[BaseModel],
        ctx: InvocationContext | None,
    ) -> str:
        """
        Return valid JSON matching schema.

        Try direct validation first — if the LLM already produced valid JSON,
        return it as-is. Only call _structure() as a repair pass on failure.
        """
        try:
            schema.model_validate_json(text)
            return text
        except Exception:
            return await self._structure(text, schema, ctx)

    async def _structure(
        self,
        text: str,
        schema: type[BaseModel],
        ctx: InvocationContext | None,
    ) -> str:
        """One extra LLM call that converts free text into JSON matching schema."""
        structuring_step, structuring_state = structuring_request(
            self.reason_step.llm, text, schema
        )
        _, result = await self._consume_wrapped(
            structuring_step,
            structuring_step.run,
            structuring_state,
            ctx,
        )
        if not isinstance(result.output, LLMOutput):
            raise TypeError(
                f"Expected LLMOutput from structuring step, got {type(result.output).__name__}"
            )
        return result.output.raw

    def register_tool(self, registry: ToolRegistry) -> None:
        """Update internal steps to reflect a newly registered tool in registry."""
        self.reason_step.serialized_tools = registry.to_serialized()
        if self.act_step is None:
            self.act_step = ToolStep(registry=registry)
