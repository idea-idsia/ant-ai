from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import ANY, MagicMock

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
    validator_status: str = "fail"


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
    guard = _make_guard(passed=False, failure_reason="toxic content detected")
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_llm_result("bad output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "toxic content detected" in decision.reason
    guard.validate.assert_called_once_with(ANY, num_reasks=0)


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_retry_reason_falls_back_when_no_summaries():
    guard = _make_guard(passed=False)
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_llm_result("bad output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "validation failed"
    guard.validate.assert_called_once_with(ANY, num_reasks=0)


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_retry_when_validate_raises():
    guard = MagicMock()
    guard.validate.side_effect = RuntimeError("internal guardrails error")
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_llm_result("some output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "guardrails validation error" in decision.reason


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_retry_when_validation_passed_true_but_summaries_show_failure():
    # Guardrails quirk: on_fail="reask" + num_reasks=0 returns validation_passed=True
    # even though validators failed. The hook must detect this via validator_status.
    outcome = _MockOutcome(
        validation_passed=True,
        validation_summaries=[
            _MockSummary(
                validator_name="DetectPII",
                failure_reason="EMAIL_ADDRESS detected in output",
                validator_status="fail",
            )
        ],
    )
    guard = MagicMock()
    guard.validate.return_value = outcome
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_llm_result("contact me at foo@bar.com"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "EMAIL_ADDRESS detected in output" in decision.reason


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_skips_empty_raw():
    guard = _make_guard(passed=False, failure_reason="should not be called")
    hook = GuardrailsAIHook(guard=guard)
    result = StepResult(
        output=LLMOutput(raw=""),
        transition=Transition(action=TransitionAction.CONTINUE),
    )
    decision = await hook.after_model(result, ctx=None)
    assert isinstance(decision, PostModelPass)
    guard.validate.assert_not_called()


@pytest.mark.unit
@pytest.mark.guardrailsai
async def test_skips_non_llm_output():
    guard = _make_guard(passed=True)
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_tool_result(), ctx=None)
    assert isinstance(decision, PostModelPass)
    guard.validate.assert_not_called()
