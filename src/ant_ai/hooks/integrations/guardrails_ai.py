from __future__ import annotations

import asyncio
import threading
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SkipValidation

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

    Args:
        guard: A configured ``guardrails.Guard`` instance.
        num_reasks: How many times guardrails may internally call an LLM to
            fix invalid output before handing control back to ant-ai.  The
            default (``0``) disables guardrails' own reask loop so ant-ai's
            retry mechanism stays in control.  Set to a positive value only
            when the guard has an ``llm_api`` configured and you want
            guardrails to attempt self-correction before ant-ai retries.
        api_key: API key forwarded to the LLM used by guardrails during
            internal reasks.  Only relevant when ``num_reasks > 0``.

    .. note::
        ``Guard`` is not thread-safe. Validation calls on a shared hook
        instance are serialized with an internal lock so concurrent agent
        invocations (e.g. inside an A2A server) do not race on the guard's
        internal history.

    Example::

        ```python
        from guardrails import Guard
        from guardrails.hub import ValidJson

        hook = GuardrailsAIHook(guard=Guard().use(ValidJson))
        agent = Agent(..., hooks=[hook])
        ```

    See ``examples/guardrails_agent.py`` for a full safety pipeline using
    ``ToxicLanguage`` and ``DetectPII`` validators.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: ClassVar[str] = "guardrails_ai"
    guard: SkipValidation[Any]  # guardrails.Guard
    num_reasks: int = Field(default=0, ge=0)
    api_key: str | None = None
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    async def after_model(
        self,
        result: StepResult,
        ctx: InvocationContext | None,
    ) -> PostModelDecision:
        if not isinstance(result.output, LLMOutput):
            return PostModelPass(result=result)

        # Guard.validate is synchronous and mutates internal state (history,
        # reask counter). Serialize via _lock so concurrent callers sharing
        # this hook instance do not race on that state.
        raw: str = result.output.raw

        def _validate() -> Any:
            with self._lock:
                kwargs: dict[str, Any] = {}
                if self.api_key is not None:
                    kwargs["api_key"] = self.api_key
                return self.guard.validate(raw, num_reasks=self.num_reasks, **kwargs)

        outcome = await asyncio.to_thread(_validate)

        if outcome.validation_passed:
            return PostModelPass(result=result)

        # Guard returned output despite failures (e.g. on_fail=NOOP) — monitor only.
        if outcome.validated_output is not None:
            return PostModelPass(result=result)

        reason: str = _failure_reason(outcome.validation_summaries)
        return PostModelRetry(reason=reason)


_FALLBACK_REASON = "validation failed"


def _failure_reason(summaries: list | None) -> str:
    """Build a retry critique from guardrails ValidationSummary objects.

    outcome.error is only populated when validate() raises an exception;
    for normal FailResult failures the details live in validation_summaries.
    """
    parts: list[str] = [
        f"{s.validator_name}: {s.failure_reason}"
        for s in (summaries or [])
        if s.failure_reason
    ]
    return "; ".join(parts) if parts else _FALLBACK_REASON
