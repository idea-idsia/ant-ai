from __future__ import annotations

from typing import Any

import pytest

from ant_ai.core.events import ClarificationNeededEvent, ToolResultEvent
from ant_ai.core.message import ToolCall, ToolCallMessage, ToolFunction
from ant_ai.core.result import (
    ClarificationNeededOutput,
    StepResult,
    ToolOutput,
    TransitionAction,
)
from ant_ai.core.types import InvocationContext, State
from ant_ai.steps.tool_step import ToolStep, _serialize_result
from ant_ai.tools.registry import ToolRegistry
from ant_ai.tools.tool import tool as tool_decorator


async def _collect(gen) -> list[Any]:
    items = []
    async for item in gen:
        items.append(item)
    return items


def _make_tool_call(
    tool_name: str, args: str = "{}", call_id: str = "call-1"
) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=ToolFunction(name=tool_name, arguments=args),
    )


def _make_state(*tool_calls: ToolCall) -> State:
    msg = ToolCallMessage(tool_calls=list(tool_calls))
    state = State()
    state.add_message(msg)
    return state


@pytest.mark.unit
def test_serialize_result_base_model():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    assert _serialize_result(M(x=1)) == '{"x":1}'


@pytest.mark.unit
def test_serialize_result_dict():
    assert _serialize_result({"a": 1}) == '{"a": 1}'


@pytest.mark.unit
def test_serialize_result_list():
    assert _serialize_result([1, 2, 3]) == "[1, 2, 3]"


@pytest.mark.unit
def test_serialize_result_other():
    assert _serialize_result(42) == "42"
    assert _serialize_result("hello") == "hello"


@pytest.mark.unit
def test_parse_args_empty_returns_empty_dict():
    step = ToolStep(registry=ToolRegistry())
    assert step._parse_args("", "mytool") == {}


@pytest.mark.unit
def test_parse_args_valid_json():
    step = ToolStep(registry=ToolRegistry())
    assert step._parse_args('{"x": 1}', "mytool") == {"x": 1}


@pytest.mark.unit
def test_parse_args_invalid_raises_value_error():
    step = ToolStep(registry=ToolRegistry())
    with pytest.raises(ValueError, match="Could not parse arguments"):
        step._parse_args("not json at all!!!", "mytool")


@pytest.mark.unit
async def test_run_happy_path_yields_tool_result_event_and_step_result():
    @tool_decorator
    def greet(name: str) -> str:
        return f"hello {name}"

    registry = ToolRegistry(tools=[greet])
    step = ToolStep(registry=registry)
    state: State = _make_state(_make_tool_call("greet", '{"name": "world"}'))

    items = await _collect(step.run(state, None))

    tool_result_events = [i for i in items if isinstance(i, ToolResultEvent)]
    step_results = [i for i in items if isinstance(i, StepResult)]

    assert len(tool_result_events) == 1
    assert "hello world" in tool_result_events[0].content

    assert len(step_results) == 1
    result = step_results[0]
    assert isinstance(result.output, ToolOutput)
    assert result.transition.action == TransitionAction.CONTINUE
    assert result.transition.next_step == "llm"
    assert result.output.results[0]["content"] == "hello world"


@pytest.mark.unit
async def test_run_multiple_tool_calls_all_included_in_output():
    @tool_decorator
    def add(x: int, y: int) -> int:
        return x + y

    registry = ToolRegistry(tools=[add])
    step = ToolStep(registry=registry)
    state = _make_state(
        _make_tool_call("add", '{"x": 1, "y": 2}', call_id="c1"),
        _make_tool_call("add", '{"x": 3, "y": 4}', call_id="c2"),
    )

    items = await _collect(step.run(state, None))

    step_results = [i for i in items if isinstance(i, StepResult)]
    assert len(step_results) == 1
    output: ToolOutput = step_results[0].output
    assert len(output.results) == 2
    contents = {r["content"] for r in output.results}
    assert {"3", "7"} == contents


@pytest.mark.unit
async def test_run_tool_not_found_returns_error_in_content():
    registry = ToolRegistry()  # empty
    step = ToolStep(registry=registry)
    state: State = _make_state(_make_tool_call("missing_tool"))

    items = await _collect(step.run(state, None))

    step_results = [i for i in items if isinstance(i, StepResult)]
    assert len(step_results) == 1
    output: ToolOutput = step_results[0].output
    assert "not found in registry" in output.results[0]["content"]


@pytest.mark.unit
async def test_run_bad_args_returns_error_in_content():
    @tool_decorator
    def noop(x: int) -> int:
        return x

    registry = ToolRegistry(tools=[noop])
    step = ToolStep(registry=registry)
    state: State = _make_state(_make_tool_call("noop", "NOT_VALID_JSON"))

    items = await _collect(step.run(state, None))

    step_results = [i for i in items if isinstance(i, StepResult)]
    assert len(step_results) == 1
    output: ToolOutput = step_results[0].output
    assert "ERROR" in output.results[0]["content"]
    assert "Could not parse" in output.results[0]["content"]


@pytest.mark.unit
async def test_run_tool_exception_returns_error_in_content():
    @tool_decorator
    def boom() -> str:
        raise RuntimeError("kaboom")

    registry = ToolRegistry(tools=[boom])
    step = ToolStep(registry=registry)
    state: State = _make_state(_make_tool_call("boom"))

    items = await _collect(step.run(state, None))

    step_results = [i for i in items if isinstance(i, StepResult)]
    assert len(step_results) == 1
    output: ToolOutput = step_results[0].output
    assert "ERROR" in output.results[0]["content"]
    assert "kaboom" in output.results[0]["content"]


@pytest.mark.unit
async def test_run_clarification_request_yields_clarification_events_and_ends():
    async def _needs_clarification(**_: Any) -> ClarificationNeededOutput:
        return ClarificationNeededOutput(question="Which file?")

    from ant_ai.tools.tool import Tool

    clarify_tool: Tool = Tool._from_function(_needs_clarification, name="clarify_tool")
    registry = ToolRegistry(tools=[clarify_tool])
    step = ToolStep(registry=registry)
    state: State = _make_state(_make_tool_call("clarify_tool"))

    items = await _collect(step.run(state, None))

    clarification_events = [i for i in items if isinstance(i, ClarificationNeededEvent)]
    step_results = [i for i in items if isinstance(i, StepResult)]

    assert len(clarification_events) == 1
    assert clarification_events[0].content == "Which file?"

    assert len(step_results) == 1
    result = step_results[0]
    assert isinstance(result.output, ClarificationNeededOutput)
    assert result.output.question == "Which file?"
    assert result.transition.action == TransitionAction.END


@pytest.mark.unit
async def test_run_injects_ctx_for_tool_that_wants_it():
    received: dict[str, Any] = {}

    @tool_decorator
    def needs_ctx(query: str, ctx: Any | None = None) -> str:
        received["ctx"] = ctx
        return query

    registry = ToolRegistry(tools=[needs_ctx])
    step = ToolStep(registry=registry)
    state: State = _make_state(_make_tool_call("needs_ctx", '{"query": "hi"}'))
    ctx = InvocationContext(session_id="s1", user_id="alice")

    await _collect(step.run(state, ctx))

    assert received["ctx"] is ctx


@pytest.mark.unit
async def test_run_does_not_pass_ctx_to_tool_that_does_not_want_it():
    received: dict[str, Any] = {"called": False}

    @tool_decorator
    def no_ctx(query: str) -> str:
        received["called"] = True
        return query

    registry = ToolRegistry(tools=[no_ctx])
    step = ToolStep(registry=registry)
    state: State = _make_state(_make_tool_call("no_ctx", '{"query": "hi"}'))
    ctx = InvocationContext(session_id="s1", user_id="alice")

    items = await _collect(step.run(state, ctx))

    step_results = [i for i in items if isinstance(i, StepResult)]
    output: ToolOutput = step_results[0].output
    assert received["called"] is True
    assert "ERROR" not in output.results[0]["content"]


@pytest.mark.unit
async def test_run_ctx_excluded_from_observability_span_input(monkeypatch):
    captured_inputs: list[Any] = []

    from ant_ai.observer import obs as obs_module

    original_span = obs_module.span

    class _CapturingSpanCtx:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured_inputs.append(kwargs.get("input"))
            self._inner = original_span(*args, **kwargs)

        async def __aenter__(self):
            return await self._inner.__aenter__()

        async def __aexit__(self, *exc):
            return await self._inner.__aexit__(*exc)

    monkeypatch.setattr(obs_module, "span", _CapturingSpanCtx)

    @tool_decorator
    def needs_ctx(query: str, ctx: Any | None = None) -> str:
        return query

    registry = ToolRegistry(tools=[needs_ctx])
    step = ToolStep(registry=registry)
    state: State = _make_state(_make_tool_call("needs_ctx", '{"query": "hi"}'))
    ctx = InvocationContext(session_id="s1", user_id="alice")

    await _collect(step.run(state, ctx))

    assert captured_inputs == [{"query": "hi"}]
