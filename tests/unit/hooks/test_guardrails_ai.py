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
async def test_retry_reason_format():
    hook = GuardrailsAIHook(
        guard=_make_guard(passed=False, failure_reason="toxic content detected")
    )
    decision = await hook.after_model(_llm_result("bad output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "test: toxic content detected"


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_retry_reason_multiple_validators():
    summaries = [
        _MockSummary(
            validator_name="detect_pii", failure_reason="EMAIL_ADDRESS detected"
        ),
        _MockSummary(
            validator_name="toxic_language", failure_reason="toxicity score 0.92"
        ),
    ]
    outcome = _MockOutcome(validation_passed=False, validation_summaries=summaries)
    guard = MagicMock()
    guard.validate.return_value = outcome
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_llm_result("bad output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert (
        decision.reason
        == "detect_pii: EMAIL_ADDRESS detected; toxic_language: toxicity score 0.92"
    )


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_retry_reason_falls_back_when_no_summaries():
    hook = GuardrailsAIHook(guard=_make_guard(passed=False))
    decision = await hook.after_model(_llm_result("bad output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "validation failed"


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_validate_called_with_num_reasks_zero():
    guard = _make_guard(passed=True)
    hook = GuardrailsAIHook(guard=guard)
    await hook.after_model(_llm_result("clean output"), ctx=None)
    guard.validate.assert_called_once_with("clean output", num_reasks=0)


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_validate_called_with_custom_num_reasks():
    guard = _make_guard(passed=True)
    hook = GuardrailsAIHook(guard=guard, num_reasks=2)
    await hook.after_model(_llm_result("clean output"), ctx=None)
    guard.validate.assert_called_once_with("clean output", num_reasks=2)


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_validate_forwards_api_key():
    guard = _make_guard(passed=True)
    hook = GuardrailsAIHook(guard=guard, api_key="sk-test")
    await hook.after_model(_llm_result("clean output"), ctx=None)
    guard.validate.assert_called_once_with(
        "clean output", num_reasks=0, api_key="sk-test"
    )


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_validate_no_api_key_kwarg_when_none():
    guard = _make_guard(passed=True)
    hook = GuardrailsAIHook(guard=guard)
    await hook.after_model(_llm_result("clean output"), ctx=None)
    _, kwargs = guard.validate.call_args
    assert "api_key" not in kwargs


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_skips_non_llm_output():
    guard = _make_guard(passed=True)
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_tool_result(), ctx=None)
    assert isinstance(decision, PostModelPass)
    guard.validate.assert_not_called()
