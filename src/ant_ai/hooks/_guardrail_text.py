from __future__ import annotations

from ant_ai.core.result import LLMOutput, StepResult


def model_output_text(result: StepResult) -> str | None:
    """Return the text a guardrail hook should inspect, or ``None`` to pass through.

    Returns ``None`` for non-``LLMOutput`` results (e.g. ``ToolOutput`` steps)
    so guardrail hooks only ever act on model-generated content. When the
    model produced no text — it wrote content only via a tool call — falls
    back to the serialized tool-call arguments so content written via tools
    (e.g. a filesystem tool) is still checked.
    """
    if not isinstance(result.output, LLMOutput):
        return None

    raw = result.output.raw
    if raw and raw.strip():
        return raw

    tool_calls = result.output.tool_calls
    if not tool_calls:
        return None
    return "\n".join(tc.function.arguments for tc in tool_calls)
