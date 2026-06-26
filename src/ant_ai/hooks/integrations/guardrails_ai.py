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
        raw = result.output.raw
        if not raw or not raw.strip():
            # No text content: check tool call arguments instead (the LLM may
            # have written content via tool calls, e.g. FilesystemTool).
            tool_calls = result.output.tool_calls
            if not tool_calls:
                return PostModelPass(result=result)
            raw = "\n".join(tc.function.arguments for tc in tool_calls)

        def _validate() -> Any:
            with self._lock:
                kwargs: dict[str, Any] = {}
                if self.api_key is not None:
                    kwargs["api_key"] = self.api_key
                return self.guard.validate(raw, num_reasks=self.num_reasks, **kwargs)

        try:
            outcome = await asyncio.to_thread(_validate)
        except Exception as exc:  # noqa: BLE001
            return PostModelRetry(reason=f"guardrails validation error: {exc}")

        # When on_fail="reask" and num_reasks=0, guardrails quirk: it sets
        # validation_passed=True even though validators failed (it expected to
        # reask but couldn't). Detect real failures via validator_status.
        failed = [
            s
            for s in (outcome.validation_summaries or [])
            if getattr(s, "validator_status", None) == "fail"
        ]
        if outcome.validation_passed and not failed:
            return PostModelPass(result=result)

        history = getattr(self.guard, "history", None)
        return PostModelRetry(
            reason=_failure_reason(
                outcome.validation_summaries,
                history.last if history else None,
            )
        )


_FALLBACK_REASON = "validation failed"


def _failure_reason(summaries: list | None, history_call: Any = None) -> str:
    """Build a retry critique from guardrails ValidationSummary objects.

    outcome.error is only populated when validate() raises an exception;
    for normal FailResult failures the details live in validation_summaries.
    When summaries are empty (e.g. validators that return FailResult with an
    empty error_message are filtered out by guardrails), fall back to the
    guard history to at least surface which validators failed.
    """
    parts: list[str] = [
        f"{s.validator_name}: {s.failure_reason}"
        for s in (summaries or [])
        if s.failure_reason and getattr(s, "validator_status", "fail") == "fail"
    ]
    if parts:
        return "; ".join(parts)

    if history_call is not None:
        it = history_call.iterations.last
        history_parts: list[str] = []
        for log in (it.validator_logs if it else []) or []:
            vr = log.validation_result
            if getattr(vr, "outcome", None) is None or vr.outcome.value != "fail":
                continue
            msg: str = getattr(vr, "error_message", "") or ""
            history_parts.append(
                f"{log.validator_name}: {msg}" if msg else log.validator_name
            )
        if history_parts:
            return "; ".join(history_parts)

    return _FALLBACK_REASON
