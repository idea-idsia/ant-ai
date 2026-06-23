from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    ToolOutput,
    Transition,
    TransitionAction,
)
from ant_ai.hooks.integrations.guardrails_ai import GuardrailsAIHook
from ant_ai.hooks.protocol import PostModelPass, PostModelRetry


def _llm_result(raw: str) -> StepResult:
    return StepResult(
        output=LLMOutput(raw=raw),
        transition=Transition(action=TransitionAction.END),
    )


@dataclass
class _MockSummary:
    validator_name: str
    failure_reason: str | None


@dataclass
class _MockOutcome:
    validation_passed: bool
    validation_summaries: list = field(default_factory=list)


def _tool_result() -> StepResult:
    return StepResult(
        output=ToolOutput(
            results=({"tool_call_id": "c1", "name": "my_tool", "content": "ok"},)
        ),
        transition=Transition(action=TransitionAction.CONTINUE),
    )


def _make_guard(passed: bool, failure_reason: str = "") -> MagicMock:
    summaries = []
    if not passed and failure_reason:
        summaries.append(
            _MockSummary(validator_name="test", failure_reason=failure_reason)
        )
    outcome = _MockOutcome(validation_passed=passed, validation_summaries=summaries)
    guard = MagicMock()
    guard.validate.return_value = outcome
    return guard


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_pass_when_validation_succeeds():
    hook = GuardrailsAIHook(guard=_make_guard(passed=True))
    decision = await hook.after_model(_llm_result("clean output"), ctx=None)
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_retry_when_validation_fails():
    hook = GuardrailsAIHook(
        guard=_make_guard(passed=False, failure_reason="toxic content detected")
    )
    decision = await hook.after_model(_llm_result("bad output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "toxic content detected" in decision.reason


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_retry_reason_falls_back_when_no_summaries():
    hook = GuardrailsAIHook(guard=_make_guard(passed=False))
    decision = await hook.after_model(_llm_result("bad output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "validation failed"


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_skips_non_llm_output():
    guard = _make_guard(passed=True)
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_tool_result(), ctx=None)
    assert isinstance(decision, PostModelPass)
    guard.validate.assert_not_called()
