from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, SkipValidation

from ant_ai.core.result import LLMOutput, StepResult
from ant_ai.core.types import InvocationContext
from ant_ai.hooks.protocol import (
    AgentHook,
    PostModelDecision,
    PostModelPass,
    PostModelRetry,
)


class GuardrailsAIHook(AgentHook, BaseModel):
    """
    Wraps a ``guardrails.Guard`` instance as an ``AgentHook``.

    Only overrides ``after_model`` — validates the LLM output text and
    returns ``PostModelRetry`` if validation fails.

    Example::

        from guardrails import Guard
        from guardrails.hub import ValidJson

        hook = GuardrailsAIHook(guard=Guard().use(ValidJson))
        agent = Agent(..., hooks=[hook])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "guardrails_ai"
    guard: SkipValidation[Any]  # guardrails.Guard

    async def after_model(
        self,
        result: StepResult,
        ctx: InvocationContext | None,
    ) -> PostModelDecision:
        if not isinstance(result.output, LLMOutput):
            return PostModelPass(result=result)

        # Guard.validate is synchronous — run in a thread to avoid blocking.
        outcome = await asyncio.to_thread(self.guard.validate, result.output.raw)

        if outcome.validation_passed:
            return PostModelPass(result=result)

        return PostModelRetry(reason=str(outcome.error))
