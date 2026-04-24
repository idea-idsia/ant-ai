from __future__ import annotations

import asyncio
import json
from asyncio import Task
from typing import Any

from pydantic import BaseModel, ConfigDict, SkipValidation

from ant_ai.core.events import ClarificationNeededEvent, ToolResultEvent
from ant_ai.core.message import (
    Message,
    ToolCall,
    ToolCallMessage,
    ToolCallResultMessage,
)
from ant_ai.core.result import (
    ClarificationNeededOutput,
    StepResult,
    ToolOutput,
    Transition,
    TransitionAction,
)
from ant_ai.core.types import InvocationContext, State
from ant_ai.observer import obs
from ant_ai.tools.registry import ToolRegistry
from ant_ai.tools.tool import Tool


def _serialize_result(res: Any) -> str:
    """Serialize a tool result to a JSON string or plain string.

    Args:
        res: The raw value returned by a tool.

    Returns:
        A JSON string for Pydantic models, dicts, and lists; `str(res)` otherwise.
    """
    if isinstance(res, BaseModel):
        return res.model_dump_json()
    if isinstance(res, (dict, list)):
        return json.dumps(res)
    return str(res)


class ToolStep(BaseModel):
    """Executes all tool calls from the preceding `LLMStep` concurrently.

    Yields a `StepResult[ToolOutput]` on success, or
    `StepResult[ClarificationNeededOutput]` with `TransitionAction.END` when
    any tool signals that human input is needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "tool"
    registry: SkipValidation[ToolRegistry]

    async def run(
        self,
        state: State,
        ctx: InvocationContext | None,
    ):
        last: Message = state.last_message
        if not isinstance(last, ToolCallMessage):
            raise TypeError(
                f"ToolStep expects last message to be ToolCallMessage, got {type(last).__name__}"
            )
        tool_calls: list[ToolCall] = last.tool_calls

        tasks: list[Task[ToolCallResultMessage | ClarificationNeededOutput]] = [
            asyncio.create_task(self._run_one(tc, ctx)) for tc in tool_calls
        ]

        result_dicts: list[dict[str, Any]] = []
        clarification: ClarificationNeededOutput | None = None

        try:
            for fut in asyncio.as_completed(tasks):
                outcome: ToolCallResultMessage | ClarificationNeededOutput = await fut

                if isinstance(outcome, ClarificationNeededOutput):
                    clarification: ClarificationNeededOutput = clarification or outcome
                    continue

                msg: ToolCallResultMessage = outcome
                result_dicts.append(
                    {
                        "tool_call_id": msg.tool_call_id,
                        "name": msg.name,
                        "content": msg.content,
                    }
                )
                yield ToolResultEvent(
                    content=msg.content or "",
                    message=msg,
                )
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

        if clarification is not None:
            yield ClarificationNeededEvent(
                content=clarification.question,
                metadata={
                    "tool_call_id": clarification.tool_call_id,
                    "name": clarification.tool_name,
                },
            )
            yield StepResult(
                output=clarification,
                transition=Transition(action=TransitionAction.END),
            )
            return

        yield StepResult(
            output=ToolOutput(results=tuple(result_dicts)),
            transition=Transition(action=TransitionAction.CONTINUE, next_step="llm"),
        )

    async def _run_one(
        self,
        tool_call: ToolCall,
        ctx: InvocationContext | None,
    ) -> ToolCallResultMessage | ClarificationNeededOutput:
        """Execute a single tool call and return its result or a clarification request.

        Args:
            tool_call: The tool call to execute, including name and arguments.
            ctx: Invocation context, or None if not available.

        Returns:
            A `ToolCallResultMessage` with the serialized result, or a
            `ClarificationNeededOutput` if the tool needs human input.
        """
        tool_name: str = getattr(tool_call.function, "name", "<unknown>")
        tool_call_id: str = getattr(tool_call, "id", "<unknown>")

        if tool_name not in self.registry:
            return ToolCallResultMessage(
                name=tool_name,
                tool_call_id=tool_call_id,
                content=f"ERROR: Tool '{tool_name}' not found in registry.",
            )

        args_str: str = tool_call.function.arguments or ""
        try:
            parsed_args: dict[str, Any] = self._parse_args(args_str, tool_name)
        except ValueError as exc:
            return ToolCallResultMessage(
                name=tool_name,
                tool_call_id=tool_call_id,
                content=f"ERROR: {exc}",
            )

        tool: Tool = self.registry[tool_name]
        try:
            async with obs.span(
                tool_name,
                as_type="tool",
                input=parsed_args,
                metadata={"tool_call_id": tool_call_id},
            ) as span:
                res: Any = await tool.ainvoke(**parsed_args)
                if isinstance(res, ClarificationNeededOutput):
                    return ClarificationNeededOutput(
                        question=res.question,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                    )
                serialized = _serialize_result(res)
                span.update(output=serialized)
            return ToolCallResultMessage(
                name=tool_name,
                tool_call_id=tool_call_id,
                content=serialized,
            )
        except Exception as exc:
            return ToolCallResultMessage(
                name=tool_name,
                tool_call_id=tool_call_id,
                content=f"ERROR: {exc}",
            )

    @staticmethod
    def _parse_args(args_str: str, tool_name: str) -> dict[str, Any]:
        """Parse a JSON arguments string into a dict.

        Args:
            args_str: Raw JSON string from the tool call function arguments.
            tool_name: Name of the tool, used in the error message if parsing fails.

        Returns:
            Parsed arguments as a dict. Returns an empty dict for empty input.

        Raises:
            ValueError: If `args_str` is not valid JSON or not a JSON object.
        """
        if not args_str:
            return {}
        try:
            result = json.loads(args_str)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        raise ValueError(
            f"Could not parse arguments for tool '{tool_name}' as JSON: {args_str!r}"
        )
