from __future__ import annotations

import pytest

from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    ToolOutput,
    Transition,
    TransitionAction,
)
from ant_ai.hooks.builtins.llm_guardrail import LLMGuardrailHook
from ant_ai.hooks.protocol import PostModelBlock, PostModelPass, PostModelRetry


class _DummyResponse:
    def __init__(self, content: str):
        self.message = Message(role="assistant", content=content)
        self.tool_calls = []


class _StubJudgeLLM:
    """Records calls and returns a configurable verdict JSON."""

    def __init__(self, verdict_json: str = '{"passed": true}'):
        self.calls: list[dict] = []
        self._verdict_json = verdict_json

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        self.calls.append(
            {"messages": list(messages), "response_format": response_format}
        )
        return _DummyResponse(self._verdict_json)


class _RaisingJudgeLLM:
    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        raise RuntimeError("judge unreachable")


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
async def test_pass_when_judge_approves():
    hook = LLMGuardrailHook(
        judge_llm=_StubJudgeLLM('{"passed": true}'), criteria="be nice"
    )
    decision = await hook.after_model(_llm_result("a friendly answer"), ctx=None)
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
async def test_retry_when_judge_rejects():
    judge = _StubJudgeLLM('{"passed": false, "reason": "contains banned content"}')
    hook = LLMGuardrailHook(judge_llm=judge, criteria="no banned content")
    decision = await hook.after_model(_llm_result("banned stuff"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "contains banned content"


@pytest.mark.unit
async def test_block_mode_returns_post_model_block():
    judge = _StubJudgeLLM('{"passed": false, "reason": "off topic"}')
    hook = LLMGuardrailHook(judge_llm=judge, criteria="stay on topic", on_fail="block")
    decision = await hook.after_model(_llm_result("off topic text"), ctx=None)
    assert isinstance(decision, PostModelBlock)
    assert decision.reason == "off topic"


@pytest.mark.unit
async def test_retry_reason_falls_back_when_no_reason_given():
    judge = _StubJudgeLLM('{"passed": false}')
    hook = LLMGuardrailHook(judge_llm=judge, criteria="be nice")
    decision = await hook.after_model(_llm_result("bad text"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "LLM guardrail check failed"


@pytest.mark.unit
async def test_retry_when_judge_call_raises():
    hook = LLMGuardrailHook(judge_llm=_RaisingJudgeLLM(), criteria="be nice")
    decision = await hook.after_model(_llm_result("some text"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "LLM guardrail judge error" in decision.reason


@pytest.mark.unit
async def test_retry_when_judge_returns_invalid_json():
    hook = LLMGuardrailHook(judge_llm=_StubJudgeLLM("not json"), criteria="be nice")
    decision = await hook.after_model(_llm_result("some text"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "LLM guardrail judge error" in decision.reason


@pytest.mark.unit
async def test_judge_called_with_response_format_and_criteria_in_prompt():
    judge = _StubJudgeLLM('{"passed": true}')
    hook = LLMGuardrailHook(judge_llm=judge, criteria="no medical advice")
    await hook.after_model(_llm_result("here is some text"), ctx=None)

    assert len(judge.calls) == 1
    call = judge.calls[0]
    assert call["response_format"] is not None
    joined = "\n".join(m.content or "" for m in call["messages"])
    assert "no medical advice" in joined
    assert "here is some text" in joined


@pytest.mark.unit
async def test_skips_empty_raw_with_no_tool_calls():
    judge = _StubJudgeLLM('{"passed": false, "reason": "should not be called"}')
    hook = LLMGuardrailHook(judge_llm=judge, criteria="be nice")
    result = StepResult(
        output=LLMOutput(raw=""),
        transition=Transition(action=TransitionAction.CONTINUE),
    )
    decision = await hook.after_model(result, ctx=None)
    assert isinstance(decision, PostModelPass)
    assert judge.calls == []


@pytest.mark.unit
async def test_judges_tool_call_arguments_when_raw_empty():
    judge = _StubJudgeLLM('{"passed": false, "reason": "leaked secret"}')
    hook = LLMGuardrailHook(judge_llm=judge, criteria="no secrets")
    tc = ToolCall(
        id="c1",
        function=ToolFunction(
            name="filesystem_tool",
            arguments='{"path": "out.py", "content": "API_KEY=secret"}',
        ),
    )
    result = StepResult(
        output=LLMOutput(raw="", tool_calls=(tc,)),
        transition=Transition(action=TransitionAction.CONTINUE),
    )
    decision = await hook.after_model(result, ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "leaked secret"


@pytest.mark.unit
async def test_skips_non_llm_output():
    judge = _StubJudgeLLM('{"passed": false, "reason": "should not be called"}')
    hook = LLMGuardrailHook(judge_llm=judge, criteria="be nice")
    decision = await hook.after_model(_tool_result(), ctx=None)
    assert isinstance(decision, PostModelPass)
    assert judge.calls == []


@pytest.mark.unit
async def test_subclass_overriding_build_judge_messages():
    class CustomGuardrail(LLMGuardrailHook):
        def build_judge_messages(self, raw: str) -> list[Message]:
            return [Message(role="user", content=f"custom prompt: {raw}")]

    judge = _StubJudgeLLM('{"passed": true}')
    hook = CustomGuardrail(judge_llm=judge, criteria="unused")
    await hook.after_model(_llm_result("payload"), ctx=None)

    assert len(judge.calls) == 1
    [message] = judge.calls[0]["messages"]
    assert message.content == "custom prompt: payload"
