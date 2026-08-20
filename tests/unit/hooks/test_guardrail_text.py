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
from ant_ai.hooks._guardrail_text import model_output_text


def _llm_result(raw: str, tool_calls: tuple[ToolCall, ...] = ()) -> StepResult:
    return StepResult(
        output=LLMOutput(raw=raw, tool_calls=tool_calls),
        transition=Transition(action=TransitionAction.CONTINUE),
    )


def _tool_call(name: str, arguments: str, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, function=ToolFunction(name=name, arguments=arguments))


@pytest.mark.unit
def test_returns_raw_when_present():
    assert model_output_text(_llm_result("hello")) == "hello"


@pytest.mark.unit
def test_whitespace_only_raw_falls_back_to_tool_calls():
    tc = _tool_call("my_tool", '{"a": 1}')
    result = _llm_result("   \n\t  ", tool_calls=(tc,))
    assert model_output_text(result) == '{"a": 1}'


@pytest.mark.unit
def test_whitespace_only_raw_with_no_tool_calls_returns_none():
    assert model_output_text(_llm_result("   ")) is None


@pytest.mark.unit
def test_empty_raw_with_no_tool_calls_returns_none():
    assert model_output_text(_llm_result("")) is None


@pytest.mark.unit
def test_multiple_tool_calls_are_joined_with_newline():
    tcs = (
        _tool_call("tool_a", '{"path": "a.py"}', call_id="c1"),
        _tool_call("tool_b", '{"path": "b.py"}', call_id="c2"),
    )
    result = _llm_result("", tool_calls=tcs)
    assert model_output_text(result) == '{"path": "a.py"}\n{"path": "b.py"}'


@pytest.mark.unit
def test_non_llm_output_returns_none():
    result = StepResult(
        output=ToolOutput(
            results=({"tool_call_id": "c1", "name": "my_tool", "content": "ok"},)
        ),
        transition=Transition(action=TransitionAction.CONTINUE),
    )
    assert model_output_text(result) is None
