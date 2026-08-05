from __future__ import annotations

import pytest

from ant_ai.core.message import ToolCall, ToolFunction
from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    ToolOutput,
    Transition,
    TransitionAction,
)
from ant_ai.hooks.integrations.pii_guardrail import PIIGuardrailHook
from ant_ai.hooks.protocol import PostModelBlock, PostModelPass, PostModelRetry


def _llm_result(raw: str) -> StepResult:
    return StepResult(
        output=LLMOutput(raw=raw),
        transition=Transition(action=TransitionAction.END),
    )


def _tool_result() -> StepResult:
    return StepResult(
        output=ToolOutput(
            results=({"tool_call_id": "c1", "name": "my_tool", "content": "ok"},)
        ),
        transition=Transition(action=TransitionAction.CONTINUE),
    )


@pytest.mark.unit
async def test_pass_when_no_pii_detected():
    hook = PIIGuardrailHook()
    decision = await hook.after_model(_llm_result("Nothing sensitive here."), ctx=None)
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
async def test_retry_when_email_detected():
    hook = PIIGuardrailHook()
    decision = await hook.after_model(
        _llm_result("Contact me at foo@bar.com"), ctx=None
    )
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "PII detected: EMAIL"


@pytest.mark.unit
async def test_reason_lists_multiple_entity_types_without_leaking_values():
    hook = PIIGuardrailHook()
    decision = await hook.after_model(
        _llm_result("Reach me at foo@bar.com or 555-123-4567"), ctx=None
    )
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "PII detected: EMAIL, PHONE"
    assert "foo@bar.com" not in decision.reason
    assert "555-123-4567" not in decision.reason


@pytest.mark.unit
async def test_block_mode_returns_post_model_block():
    hook = PIIGuardrailHook(on_detect="block")
    decision = await hook.after_model(
        _llm_result("Contact me at foo@bar.com"), ctx=None
    )
    assert isinstance(decision, PostModelBlock)
    assert decision.reason == "PII detected: EMAIL"


@pytest.mark.unit
async def test_entity_types_filter_restricts_detection():
    hook = PIIGuardrailHook(entity_types=["PHONE"])
    decision = await hook.after_model(
        _llm_result("Contact me at foo@bar.com"), ctx=None
    )
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
async def test_allowlist_exempts_exact_value():
    hook = PIIGuardrailHook(allowlist=["support@example.com"])
    decision = await hook.after_model(
        _llm_result("Email support@example.com for help"), ctx=None
    )
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
async def test_skips_empty_raw_with_no_tool_calls():
    hook = PIIGuardrailHook()
    result = StepResult(
        output=LLMOutput(raw=""),
        transition=Transition(action=TransitionAction.CONTINUE),
    )
    decision = await hook.after_model(result, ctx=None)
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
async def test_scans_tool_call_arguments_when_raw_empty():
    hook = PIIGuardrailHook()
    tc = ToolCall(
        id="c1",
        function=ToolFunction(
            name="filesystem_tool",
            arguments='{"path": "out.py", "content": "Author: foo@bar.com"}',
        ),
    )
    result = StepResult(
        output=LLMOutput(raw="", tool_calls=(tc,)),
        transition=Transition(action=TransitionAction.CONTINUE),
    )
    decision = await hook.after_model(result, ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "PII detected: EMAIL"


@pytest.mark.unit
async def test_skips_non_llm_output():
    hook = PIIGuardrailHook()
    decision = await hook.after_model(_tool_result(), ctx=None)
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
async def test_retry_when_scan_raises(monkeypatch):
    hook = PIIGuardrailHook()

    def _boom(*args, **kwargs):
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr("ant_ai.hooks.integrations.pii_guardrail.datafog_scan", _boom)
    decision = await hook.after_model(_llm_result("some text"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "PII guardrail scan error" in decision.reason
