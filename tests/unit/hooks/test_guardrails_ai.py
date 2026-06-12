from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    ToolOutput,
    Transition,
    TransitionAction,
)
from ant_ai.hooks.adapters.guardrails_ai import GuardrailsAIHook
from ant_ai.hooks.protocol import PostModelPass, PostModelRetry


def _llm_result(raw: str) -> StepResult:
    return StepResult(
        output=LLMOutput(raw=raw),
        transition=Transition(action=TransitionAction.END),
    )


def _tool_result() -> StepResult:
    return StepResult(
        output=ToolOutput(tool_call_id="c1", name="my_tool", result="ok"),
        transition=Transition(action=TransitionAction.CONTINUE),
    )


def _make_guard(passed: bool, error: str = "") -> MagicMock:
    outcome = MagicMock()
    outcome.validation_passed = passed
    outcome.error = error
    guard = MagicMock()
    guard.validate.return_value = outcome
    return guard


@pytest.mark.unit
async def test_pass_when_validation_succeeds():
    hook = GuardrailsAIHook(guard=_make_guard(passed=True))
    decision = await hook.after_model(_llm_result("clean output"), ctx=None)
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
async def test_retry_when_validation_fails():
    hook = GuardrailsAIHook(
        guard=_make_guard(passed=False, error="toxic content detected")
    )
    decision = await hook.after_model(_llm_result("bad output"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "toxic content detected" in decision.reason


@pytest.mark.unit
async def test_skips_non_llm_output():
    guard = _make_guard(passed=True)
    hook = GuardrailsAIHook(guard=guard)
    decision = await hook.after_model(_tool_result(), ctx=None)
    assert isinstance(decision, PostModelPass)
    guard.validate.assert_not_called()
