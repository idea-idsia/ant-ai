from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ant_ai.core.message import ToolCallResultMessage
from ant_ai.core.result import ClarificationNeededOutput
from ant_ai.steps.tracing import tool_span
from ant_ai.tools.tool import current_tool_call

if TYPE_CHECKING:
    from ant_ai.core.message import ToolCall
    from ant_ai.core.types import InvocationContext
    from ant_ai.tools.registry import ToolRegistry
    from ant_ai.tools.tool import Tool


def serialize_result(res: Any) -> str:
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


def parse_args(args_str: str, tool_name: str) -> dict[str, Any]:
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


def _error(tool_name: str, tool_call_id: str, message: str) -> ToolCallResultMessage:
    return ToolCallResultMessage(
        name=tool_name,
        tool_call_id=tool_call_id,
        content=f"ERROR: {message}",
        is_error=True,
    )


async def run_tool_call(
    registry: ToolRegistry,
    tool_call: ToolCall,
    ctx: InvocationContext | None,
) -> ToolCallResultMessage | ClarificationNeededOutput:
    """Execute a single tool call and return its result or a clarification request.

    Args:
        registry: The tools the call can name.
        tool_call: The tool call to execute, including name and arguments.
        ctx: Invocation context, or None if not available.

    Returns:
        A `ToolCallResultMessage` with the serialized result, or a
        `ClarificationNeededOutput` if the tool needs human input.
    """
    tool_name: str = getattr(tool_call.function, "name", "<unknown>")
    tool_call_id: str = getattr(tool_call, "id", "<unknown>")

    if tool_name not in registry:
        return _error(
            tool_name, tool_call_id, f"Tool '{tool_name}' not found in registry."
        )

    try:
        parsed_args: dict[str, Any] = parse_args(
            tool_call.function.arguments or "", tool_name
        )
    except ValueError as exc:
        return _error(tool_name, tool_call_id, str(exc))

    tool: Tool = registry[tool_name]
    call_kwargs: dict[str, Any] = dict(parsed_args)
    if tool.wants_context:
        call_kwargs["ctx"] = ctx
    token = current_tool_call.set(tool_call)
    try:
        async with tool_span(tool_name, parsed_args, tool_call_id) as span:
            res: Any = await tool.ainvoke(**call_kwargs)
            if isinstance(res, ClarificationNeededOutput):
                return ClarificationNeededOutput(
                    question=res.question,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )
            serialized = serialize_result(res)
            span.update(output=serialized)
        return ToolCallResultMessage(
            name=tool_name,
            tool_call_id=tool_call_id,
            content=serialized,
        )
    except Exception as exc:
        return _error(tool_name, tool_call_id, str(exc))
    finally:
        current_tool_call.reset(token)
