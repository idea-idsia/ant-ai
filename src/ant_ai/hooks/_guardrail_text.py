from __future__ import annotations

from typing import TYPE_CHECKING

from ant_ai.core.result import LLMOutput, StepResult
from ant_ai.hooks.protocol import (
    AgentHook,
    PostModelBlock,
    PostModelDecision,
    PostModelPass,
    PostModelRetry,
)

if TYPE_CHECKING:
    from ant_ai.core.types import InvocationContext


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


def reject(reason: str, block: bool = False) -> PostModelBlock | PostModelRetry:
    """Retry with `reason` as critique, or block on it when `block` is set."""
    return PostModelBlock(reason=reason) if block else PostModelRetry(reason=reason)


class TextGuardrailHook(AgentHook):
    """Guardrail hook that judges the text of each model step.

    Passes results with no text to inspect (see `model_output_text`) and
    delegates the rest to `check_text`.
    """

    async def after_model(
        self,
        result: StepResult,
        ctx: InvocationContext | None,
    ) -> PostModelDecision:
        text = model_output_text(result)
        if text is None:
            return PostModelPass(result=result)
        failure = await self.check_text(text)
        return PostModelPass(result=result) if failure is None else failure

    async def check_text(self, text: str) -> PostModelDecision | None:
        """Judge `text`: return None to accept it, or the decision to take instead."""
        raise NotImplementedError
