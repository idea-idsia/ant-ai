from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, SkipValidation

from ant_ai.core.message import ToolCallMessage
from ant_ai.steps.tool_call import run_tool_call
from ant_ai.steps.tool_results import ToolResults
from ant_ai.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from asyncio import Task

    from ant_ai.core.message import BaseMessage, ToolCall, ToolCallResultMessage
    from ant_ai.core.result import ClarificationNeededOutput
    from ant_ai.core.types import InvocationContext, State


class ToolStep(BaseModel):
    """Executes all tool calls from the preceding `LLMStep` concurrently.

    Always yields a `StepResult[ToolOutput]` holding a result for every call.
    If any tool asked for human input (`ClarificationNeededOutput`), that call
    is answered with its question, a `ClarificationNeededEvent` is emitted, and
    the transition is `END` so the run stops for the user's reply -- with the
    transcript well-formed for resumption.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "tool"
    registry: SkipValidation[ToolRegistry]

    async def run(
        self,
        state: State,
        ctx: InvocationContext | None,
    ):
        last: BaseMessage = state.last_message
        if not isinstance(last, ToolCallMessage):
            raise TypeError(
                f"ToolStep expects last message to be ToolCallMessage, got {type(last).__name__}"
            )

        tasks: list[Task[ToolCallResultMessage | ClarificationNeededOutput]] = [
            asyncio.create_task(self._run_one(tc, ctx)) for tc in last.tool_calls
        ]
        results = ToolResults()

        try:
            for fut in asyncio.as_completed(tasks):
                event = results.add(await fut)
                if event is not None:
                    yield event
        finally:
            # Leaving early (cancellation, a tool raising, the consumer closing
            # the generator) must take the in-flight tools down too: gather
            # alone would wait for them to finish on their own, so a cancelled
            # run would keep executing until its slowest tool returned.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        for event in results.answer_clarifications():
            yield event
        yield results.step_result()

    async def _run_one(
        self,
        tool_call: ToolCall,
        ctx: InvocationContext | None,
    ) -> ToolCallResultMessage | ClarificationNeededOutput:
        """Execute a single tool call against this step's registry."""
        return await run_tool_call(self.registry, tool_call, ctx)
