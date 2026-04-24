from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic import BaseModel

from ant_ai.agent.loop.loop import BaseAgentLoop
from ant_ai.core.events import (
    Event,
    EventOrigin,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
)
from ant_ai.core.message import Message, ToolCallMessage, ToolCallResultMessage
from ant_ai.core.result import (
    ClarificationNeededOutput,
    LLMOutput,
    StepResult,
    ToolOutput,
    TransitionAction,
)
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks import WrapCall
from ant_ai.steps import LLMStep, ToolStep
from ant_ai.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolRequest:
    """LLM produced tool calls and act_step is configured — run the tools."""


@dataclass(frozen=True)
class FinalResponse:
    """LLM produced a final text answer, end the loop."""


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
    ) -> AsyncIterator[Event]:
        active_step: LLMStep = (
            self.reason_step.model_copy(update={"response_format": response_schema})
            if self.act_step is None and response_schema
            else self.reason_step
        )
        coerce_schema: type[BaseModel] | None = (
            response_schema if self.act_step is not None else None
        )

        for loop_step in range(1, max_steps + 1):
            llm_result: StepResult | None = None

            await self.hooks.run_before_model(state, ctx)

            async for item in self._run_model_with_hooks(active_step, state, ctx):
                if isinstance(item, StepResult):
                    llm_result: StepResult = item
                elif not isinstance(item, FinalAnswerEvent):
                    # LLMStep emits a FinalAnswerEvent before its StepResult as a side-effect.
                    # We emit the real FinalAnswerEvent below after optional schema handling.
                    yield item

            if llm_result is None:
                raise RuntimeError("LLM step produced no result")
            if not isinstance(llm_result.output, LLMOutput):
                raise TypeError(
                    f"Expected LLMOutput, got {type(llm_result.output).__name__}"
                )

            match self._classify_llm_result(llm_result):
                case ToolRequest():
                    if self.act_step is None:
                        raise RuntimeError(
                            "LLM requested tool calls but no tools are configured on this agent."
                        )
                    state.add_message(
                        ToolCallMessage(tool_calls=list(llm_result.output.tool_calls))
                    )
                    wrapped_tools: WrapCall = self.hooks.wrap_tool_call(
                        self.act_step.run
                    )
                    act_result: StepResult | None = None
                    async for item in self._observe_step(
                        self.act_step, wrapped_tools(state, ctx)
                    ):
                        if isinstance(item, StepResult):
                            act_result = item
                        else:
                            yield item
                    if act_result is None:
                        raise RuntimeError("Tool step produced no result")
                    if isinstance(act_result.output, ClarificationNeededOutput):
                        return
                    if not isinstance(act_result.output, ToolOutput):
                        raise TypeError(
                            f"Expected ToolOutput, got {type(act_result.output).__name__}"
                        )
                    for r in act_result.output.results:
                        state.add_message(
                            ToolCallResultMessage(
                                name=r["name"],
                                tool_call_id=r["tool_call_id"],
                                content=r["content"],
                            )
                        )

                case FinalResponse():
                    final_event = await self._make_final_answer(
                        llm_result.output.raw, loop_step, coerce_schema, ctx
                    )
                    state.add_message(
                        Message(role="assistant", content=final_event.content)
                    )
                    yield final_event
                    return

        yield MaxStepsReachedEvent(
            origin=EventOrigin(layer="agent", run_step=max_steps),
        )

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
    ) -> FinalAnswerEvent:
        final_text: str = text
        if response_schema is not None:
            final_text = await self._coerce_to_schema(text, response_schema, ctx)

        return FinalAnswerEvent(
            origin=EventOrigin(layer="agent", run_step=loop_step),
            content=final_text,
        )

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
        """
        One extra LLM call that converts free text into JSON matching schema.

        Uses response_format so constrained decoding guarantees valid output —
        no retry loop, no hook needed.
        """
        structuring_step = LLMStep(
            llm=self.reason_step.llm,
            system_message=Message(
                role="system",
                content=(
                    "Convert the following text into a JSON object. "
                    "Respond with valid JSON only, no explanation."
                ),
            ),
            serialized_tools=[],
            response_format=schema,
        )
        structuring_state = State()
        structuring_state.add_message(Message(role="user", content=text))

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
