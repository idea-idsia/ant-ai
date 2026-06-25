from __future__ import annotations

import asyncio
import threading
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, PrivateAttr, SkipValidation

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
        raw = result.output.raw
        if not raw or not raw.strip():
            return PostModelPass(result=result)

        def _validate() -> Any:
            with self._lock:
                return self.guard.validate(raw, num_reasks=0)

        try:
            outcome = await asyncio.to_thread(_validate)
        except Exception as exc:  # noqa: BLE001
            return PostModelRetry(reason=f"guardrails validation error: {exc}")

        # When on_fail="reask" and num_reasks=0, guardrails quirk: it sets
        # validation_passed=True even though validators failed (it expected to
        # reask but couldn't). Detect real failures via validator_status.
        failed = [
            s for s in (outcome.validation_summaries or [])
            if getattr(s, "validator_status", None) == "fail"
        ]
        if outcome.validation_passed and not failed:
            return PostModelPass(result=result)

        return PostModelRetry(reason=_failure_reason(outcome.validation_summaries))


def _failure_reason(summaries: list | None) -> str:
    """Build a retry critique from guardrails ValidationSummary objects."""
    parts = [
        f"{s.validator_name}: {s.failure_reason}"
        for s in (summaries or [])
        if s.failure_reason
        and getattr(s, "validator_status", "fail") == "fail"
    ]
    return "; ".join(parts) if parts else "validation failed"
