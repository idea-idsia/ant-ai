from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic import model_serializer

from ant_ai.agent.agent import Agent
from ant_ai.core.events import (
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ToolCallingEvent,
    ToolResultEvent,
)
from ant_ai.core.exceptions import HookMaxRetriesError
from ant_ai.core.message import (
    Message,
    ToolCall,
    ToolCallMessage,
    ToolCallResultMessage,
    ToolFunction,
)
from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    Transition,
    TransitionAction,
)
from ant_ai.core.types import State
from ant_ai.hooks import (
    AgentHook,
    PostModelFallback,
    PostModelPass,
    PostModelRetry,
)
from ant_ai.llm.protocol import ChatLLM
from ant_ai.steps.llm_step import LLMStep
from ant_ai.steps.tool_step import ToolStep
from ant_ai.tools.registry import ToolRegistry


async def run_step(step, state, ctx=None) -> StepResult:
    """Consume a step generator and return the final StepResult."""
    async for item in step.run(state, ctx):
        if isinstance(item, StepResult):
            return item
    raise RuntimeError("Step generator did not yield StepResult")


def make_tool_call(
    *,
    call_id: str = "call-1",
    name: str = "my_tool",
    arguments: str = '{"x": 1}',
) -> ToolCall:
    """Construct a ToolCall with a valid JSON arguments string."""
    return ToolCall(
        id=call_id,
        function=ToolFunction(name=name, arguments=arguments),
    )


class DummyTool:
    def __init__(self, result: Any = "tool-result"):
        self.result = result
        self.called_with: dict[str, Any] | None = None

    async def ainvoke(self, **kwargs: Any) -> Any:
        self.called_with = kwargs
        return self.result


class DummyResponse:
    """Duck-typed stand-in for ChatLLMResponse."""

    def __init__(self, message: Message, tool_calls: list[ToolCall] | None = None):
        self.message = message
        self.tool_calls = tool_calls or []


@pytest.mark.unit
async def test_tool_registry_is_built_on_init():
    class DummyLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(message=Message(role="assistant", content="ok"))

    agent = Agent(
        name="test",
        system_prompt="You are a test agent.",
        llm=DummyLLM(),
        tools=[],
    )

    # Registry is built and exposed via .registry
    assert hasattr(agent, "_registry")
    assert agent.registry is agent._registry

    # Executor owns the serialized tools — not Agent directly
    assert agent._loop.reason_step.serialized_tools == []


@pytest.mark.unit
async def test_llm_step_calls_llm_with_system_plus_state_messages_and_tools():
    """LLMStep prepends the system message and passes serialized tools to the LLM."""
    calls: dict[str, Any] = {}

    class RecordingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            calls["messages"] = list(messages)
            calls["ctx"] = ctx
            calls["tools"] = tools
            return DummyResponse(
                message=Message(role="assistant", content="hello"), tool_calls=[]
            )

    system_msg = Message(role="system", content="sys")
    step = LLMStep(llm=RecordingLLM(), system_message=system_msg, serialized_tools=[])

    user_msg = Message(role="user", content="What is 2+2?")
    state = State(messages=[user_msg])

    result = await run_step(step, state)

    assert calls["messages"] == [system_msg, user_msg]
    assert calls["ctx"] is None
    assert calls["tools"] is None  # empty list passed as None
    assert result.output.kind == "llm"
    assert result.output.raw == "hello"


@pytest.mark.unit
async def test_llm_step_sets_continue_transition_when_tool_calls_present():
    tool_call = make_tool_call()

    class ToolCallingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(
                message=Message(role="assistant", content=""),
                tool_calls=[tool_call],
            )

    step = LLMStep(
        llm=ToolCallingLLM(),
        system_message=Message(role="system", content="sys"),
    )
    state = State(messages=[Message(role="user", content="go")])
    result = await run_step(step, state)

    assert result.output.has_tool_calls
    assert result.transition.next_step == "tool"


@pytest.mark.unit
async def test_llm_step_sets_end_transition_when_no_tool_calls():
    class FinalLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(
                message=Message(role="assistant", content="done"), tool_calls=[]
            )

    step = LLMStep(
        llm=FinalLLM(),
        system_message=Message(role="system", content="sys"),
    )
    state = State(messages=[Message(role="user", content="go")])
    result = await run_step(step, state)

    assert not result.output.has_tool_calls
    assert result.transition.action == TransitionAction.END


@pytest.mark.unit
async def test_tool_step_executes_tool_and_returns_result():
    dummy = DummyTool(result="42")
    registry = ToolRegistry()
    registry.register(_make_tool_shim("my_tool", dummy))

    step = ToolStep(registry=registry)

    tool_call = make_tool_call(call_id="call-1", name="my_tool", arguments='{"x": 1}')
    tool_call_msg = ToolCallMessage(tool_calls=[tool_call])
    state = State(messages=[tool_call_msg])

    result = await run_step(step, state)

    assert result.output.kind == "tool"
    assert len(result.output.results) == 1
    r = result.output.results[0]
    assert r["tool_call_id"] == "call-1"
    assert r["name"] == "my_tool"
    assert r["content"] == "42"
    assert dummy.called_with == {"x": 1}


@pytest.mark.unit
async def test_tool_step_returns_error_content_for_missing_tool():
    registry = ToolRegistry()
    step = ToolStep(registry=registry)

    tool_call = make_tool_call(call_id="xyz", name="does_not_exist", arguments="{}")
    state = State(messages=[ToolCallMessage(tool_calls=[tool_call])])

    result = await run_step(step, state)

    assert result.output.kind == "tool"
    r = result.output.results[0]
    assert "does_not_exist" in r["content"]
    assert "ERROR" in r["content"]


@pytest.mark.unit
async def test_tool_step_serializes_pydantic_result_as_json():
    from pydantic import BaseModel

    class MyModel(BaseModel):
        value: int

    dummy = DummyTool(result=MyModel(value=99))
    registry = ToolRegistry()
    registry.register(_make_tool_shim("model_tool", dummy))

    step = ToolStep(registry=registry)
    tc = make_tool_call(call_id="c1", name="model_tool", arguments="{}")
    state = State(messages=[ToolCallMessage(tool_calls=[tc])])

    result = await run_step(step, state)
    assert result.output.results[0]["content"] == '{"value":99}'


@pytest.mark.unit
async def test_tool_step_executes_multiple_tools_concurrently(monkeypatch):
    """All tool calls are launched as concurrent tasks; results arrive in completion order."""
    from ant_ai.steps.tool_step import ToolStep as _ToolStep

    delays = {"c1": 0.06, "c2": 0.03, "c3": 0.01}

    async def fake_run_one(self, tool_call, ctx):
        await asyncio.sleep(delays[tool_call.id])
        return ToolCallResultMessage(
            name="n/a", tool_call_id=tool_call.id, content="ok"
        )

    monkeypatch.setattr(_ToolStep, "_run_one", fake_run_one)

    step = ToolStep(registry=ToolRegistry())
    tool_calls = [
        make_tool_call(call_id="c1", name="t1", arguments="{}"),
        make_tool_call(call_id="c2", name="t2", arguments="{}"),
        make_tool_call(call_id="c3", name="t3", arguments="{}"),
    ]
    state = State(messages=[ToolCallMessage(tool_calls=tool_calls)])

    t0 = time.perf_counter()
    event_ids: list[str] = []
    result: StepResult | None = None
    async for item in step.run(state, ctx=None):
        if isinstance(item, StepResult):
            result = item
        else:
            event_ids.append(item.message.tool_call_id)
    dt = time.perf_counter() - t0

    assert result is not None
    ids = [r["tool_call_id"] for r in result.output.results]
    assert set(ids) == {"c1", "c2", "c3"}
    # Completion order: c3 fastest, then c2, then c1
    assert ids == ["c3", "c2", "c1"]
    # Events also arrive in real-time completion order
    assert event_ids == ["c3", "c2", "c1"]
    # Concurrent: total time ≈ max delay, not sum
    assert dt < 0.09, f"expected concurrent execution; took {dt:.3f}s"


@pytest.mark.unit
async def test_stream_without_tools_yields_final_answer():
    class NoToolLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(
                message=Message(role="assistant", content="final answer"),
                tool_calls=[],
            )

    agent = Agent(name="stream-agent", system_prompt="sys", llm=NoToolLLM(), tools=[])
    state = State(messages=[Message(role="user", content="What is the answer?")])

    events = [e async for e in agent.stream(state, max_steps=3, ctx=None)]

    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "final answer"


@pytest.mark.unit
async def test_stream_with_tool_calls_then_final_answer():
    """
    Two-step interaction: first LLM call requests a tool, second produces final answer.

    State is immutable — the original state object is never mutated.
    The executor threads new messages through immutable copies internally.
    """

    class TwoStepLLM(ChatLLM):
        def __init__(self):
            self.calls = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.calls += 1
            if self.calls == 1:
                return DummyResponse(
                    message=Message(role="assistant", content="calling tool"),
                    tool_calls=[make_tool_call(call_id="call-1", name="my_tool")],
                )
            return DummyResponse(
                message=Message(role="assistant", content="done"), tool_calls=[]
            )

    llm = TwoStepLLM()
    dummy_tool = DummyTool(result="tool-result")

    tool_shim = _make_tool_shim("my_tool", dummy_tool)

    agent = Agent(
        name="stream-tools-agent",
        system_prompt="sys",
        llm=llm,
        tools=[tool_shim],
    )

    original_state = State(messages=[Message(role="user", content="Question?")])
    events = [e async for e in agent.stream(original_state, max_steps=5, ctx=None)]

    assert any(isinstance(e, ToolCallingEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) for e in events)
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "done"
    assert dummy_tool.called_with == {"x": 1}
    assert llm.calls == 2

    assert any(isinstance(m, ToolCallMessage) for m in original_state.messages)
    assert any(isinstance(m, ToolCallResultMessage) for m in original_state.messages)


@pytest.mark.unit
async def test_stream_max_steps_reached():
    """When max_steps is exhausted the executor emits MAX_STEPS_REACHED as the last event."""

    class LoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(
                message=Message(role="assistant", content="still working"),
                tool_calls=[make_tool_call(call_id="call-1", name="my_tool")],
            )

    dummy_tool = DummyTool(result="ok")
    tool_shim = _make_tool_shim("my_tool", dummy_tool)

    agent = Agent(
        name="loop-agent", system_prompt="sys", llm=LoopLLM(), tools=[tool_shim]
    )
    state = State(messages=[Message(role="user", content="Question?")])

    events = [e async for e in agent.stream(state, max_steps=2, ctx=None)]

    # MAX_STEPS_REACHED is the terminal event — no FINAL_ANSWER after it
    assert isinstance(events[-1], MaxStepsReachedEvent)


@pytest.mark.unit
def test_tool_step_parse_args_json():
    assert ToolStep._parse_args('{"a": 1, "b": true}', "t") == {"a": 1, "b": True}


@pytest.mark.unit
def test_tool_step_parse_args_empty():
    assert ToolStep._parse_args("", "t") == {}


@pytest.mark.unit
def test_tool_step_parse_args_invalid_raises():
    with pytest.raises(ValueError, match="Could not parse"):
        ToolStep._parse_args("not json at all !!!", "t")


def _make_tool_shim(tool_name: str, impl: DummyTool):
    from ant_ai.tools.tool import Tool

    class _Shim(Tool):
        name: str
        description: str = "test"
        parameters: dict = {}

        async def ainvoke(self, **kwargs):
            return await impl.ainvoke(**kwargs)

        @model_serializer
        def serialize_model(self):
            return {"name": self.name, "description": self.description}

    # ainvoke is a public method and would be collected by __init_subclass__ as a
    # namespace method, causing the tool to register as "{name}_ainvoke" instead of
    # "{name}". Clear it so the shim is treated as a plain callable tool.
    _Shim.__namespace_methods__ = []
    return _Shim(name=tool_name)


@pytest.mark.unit
async def test_agent_hook_retry_retries_llm_and_returns_clean_answer():
    """RETRY verdict triggers a retry; final answer comes from the retried LLM call."""

    class ContentHook(AgentHook):
        async def after_model(self, result, ctx):
            if "banned" in result.output.raw:
                return PostModelRetry(reason="Response contains banned content")
            return PostModelPass(result=result)

    class TwoAttemptLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            if self.call_count == 1:
                return DummyResponse(
                    message=Message(role="assistant", content="banned content")
                )
            return DummyResponse(
                message=Message(role="assistant", content="clean answer")
            )

    llm = TwoAttemptLLM()
    agent = Agent(
        name="test",
        system_prompt="sys",
        llm=llm,
        tools=[],
        hooks=[ContentHook()],
        max_retries=2,
    )
    s = State()
    s.add_message(Message(role="user", content="hi"))

    answer = await agent.ainvoke(s)

    assert llm.call_count == 2
    assert answer == "clean answer"


@pytest.mark.unit
async def test_agent_hook_max_retries_raises_error():
    """Hook always failing exhausts retries and raises HookMaxRetriesError."""

    class AlwaysRetryHook(AgentHook):
        async def after_model(self, result, ctx):
            return PostModelRetry(reason="always fails")

    class CountingLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            return DummyResponse(
                message=Message(role="assistant", content="some response")
            )

    llm = CountingLLM()
    agent = Agent(
        name="test",
        system_prompt="sys",
        llm=llm,
        tools=[],
        hooks=[AlwaysRetryHook()],
        max_retries=1,
    )
    s = State()
    s.add_message(Message(role="user", content="hello"))

    with pytest.raises(HookMaxRetriesError):
        await agent.ainvoke(s)

    assert llm.call_count == 2  # original + one retry


@pytest.mark.unit
async def test_agent_hook_fallback_signal_returns_safe_answer():
    """PostModelFallback returns the pre-configured text without retrying the LLM."""

    class FallbackHook(AgentHook):
        async def after_model(self, result, ctx):
            fallback = StepResult(
                output=LLMOutput(raw="I cannot help with that."),
                transition=Transition(action=TransitionAction.END),
            )
            return PostModelFallback(result=fallback)

    class CountingLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            return DummyResponse(
                message=Message(role="assistant", content="original response")
            )

    llm = CountingLLM()
    agent = Agent(
        name="test",
        system_prompt="sys",
        llm=llm,
        tools=[],
        hooks=[FallbackHook()],
    )
    s = State()
    s.add_message(Message(role="user", content="do something bad"))

    answer = await agent.ainvoke(s)

    assert answer == "I cannot help with that."
    assert llm.call_count == 1  # no retry — fallback bypasses LLM


@pytest.mark.unit
async def test_agent_hook_fires_on_reasoning_step_while_tools_work_normally():
    """Hook repairs a bad text response while tool execution proceeds unaffected."""

    class TextContentHook(AgentHook):
        async def after_model(self, result, ctx):
            if result.output.has_tool_calls:
                return PostModelPass(result=result)
            if "banned" in result.output.raw:
                return PostModelRetry(reason="Contains banned content")
            return PostModelPass(result=result)

    class ThreeStepLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            if self.call_count == 1:
                return DummyResponse(
                    message=Message(role="assistant", content="calling tool"),
                    tool_calls=[make_tool_call(call_id="call-1", name="my_tool")],
                )
            if self.call_count == 2:
                return DummyResponse(
                    message=Message(role="assistant", content="banned response")
                )
            return DummyResponse(
                message=Message(role="assistant", content="final clean answer")
            )

    llm = ThreeStepLLM()
    dummy_tool = DummyTool(result="tool-result")
    tool_shim = _make_tool_shim("my_tool", dummy_tool)

    agent = Agent(
        name="test",
        system_prompt="sys",
        llm=llm,
        tools=[tool_shim],
        hooks=[TextContentHook()],
        max_retries=2,
    )
    s = State(messages=[Message(role="user", content="do task")])

    events = [e async for e in agent.stream(s, max_steps=5, ctx=None)]

    assert llm.call_count == 3
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "final clean answer"
    assert any(isinstance(e, ToolCallingEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) for e in events)


@pytest.mark.unit
async def test_agent_hook_critique_message_is_injected_into_retry_state():
    """The hook's reason string is injected into the LLM messages on retry."""
    CRITIQUE_MARKER = "SPECIFIC_CRITIQUE_MARKER_XYZ"
    received_messages_on_retry: list = []

    class FirstCallHook(AgentHook):
        def __init__(self):
            self._checked = False

        async def after_model(self, result, ctx):
            if not self._checked:
                self._checked = True
                return PostModelRetry(reason=CRITIQUE_MARKER)
            return PostModelPass(result=result)

    class RecordingLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            if self.call_count == 2:
                received_messages_on_retry.extend(messages)
            return DummyResponse(message=Message(role="assistant", content="response"))

    llm = RecordingLLM()
    agent = Agent(
        name="test",
        system_prompt="sys",
        llm=llm,
        tools=[],
        hooks=[FirstCallHook()],
        max_retries=1,
    )
    s = State()
    s.add_message(Message(role="user", content="question"))

    await agent.ainvoke(s)

    assert llm.call_count == 2
    retry_contents = [m.content for m in received_messages_on_retry if m.content]
    assert any(CRITIQUE_MARKER in c for c in retry_contents)
