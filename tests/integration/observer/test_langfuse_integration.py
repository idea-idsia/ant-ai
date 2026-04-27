from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from langfuse import Langfuse

from ant_ai.core.types import InvocationContext, State
from ant_ai.observer import obs
from ant_ai.observer.integrations.langfuse import LangfuseSink
from ant_ai.workflow.workflow import END, START, Workflow


def _langfuse_client():
    """Return a configured Langfuse client."""
    from langfuse import get_client

    lf: Langfuse = get_client()

    ok: bool = lf.auth_check()
    if not ok:
        pytest.skip("Langfuse auth_check() failed")

    return lf


def _flush_and_shutdown(lf) -> None:
    lf.flush()
    with contextlib.suppress(Exception):
        lf.shutdown()


def _get_trace(
    lf,
    trace_id: str,
    *,
    initial_wait: float = 10.0,
    retries: int = 5,
    delay: float = 5.0,
    min_obs: int = 0,
):
    assert trace_id, f"expected non-empty trace_id, got {trace_id!r}"

    time.sleep(initial_wait)
    last_exc: Exception | None = None
    last_trace = None

    for _ in range(retries):
        try:
            trace = lf.api.trace.get(trace_id)
            if trace is not None:
                last_trace = trace
                if len(trace.observations) >= min_obs:
                    return trace
        except Exception as exc:
            last_exc = exc
        time.sleep(delay)

    if last_trace is not None:
        return last_trace

    pytest.fail(
        f"Trace {trace_id!r} did not appear in Langfuse after {retries} retries "
        f"(last error: {last_exc!r})"
    )


class _DummyAgent:
    def __init__(self, name: str = "test_agent"):
        self.name = name


def _state_with_message(content: str = "hello world") -> State:
    st = State()

    class _Msg:
        def __init__(self, c):
            self.content = c

    st.messages.append(_Msg(content))
    return st


def _noop_workflow() -> Workflow:
    w = Workflow()

    async def noop(agent, state, ctx):
        async def _gen():
            if False:
                yield

        return _gen()

    w.add_node("node_a", noop)
    w.add_edge(START, "node_a")
    w.add_edge("node_a", END)
    return w


def _two_node_workflow() -> Workflow:
    w = Workflow()

    async def noop(agent, state, ctx):
        async def _gen():
            if False:
                yield

        return _gen()

    w.add_node("node_a", noop)
    w.add_node("node_b", noop)
    w.add_edge(START, "node_a")
    w.add_edge("node_a", "node_b")
    w.add_edge("node_b", END)
    return w


def _failing_workflow() -> Workflow:
    w = Workflow()

    async def boom(agent, state, ctx):
        async def _gen():
            raise RuntimeError("intentional failure")
            if False:
                yield

        return _gen()

    w.add_node("failing_node", boom)
    w.add_edge(START, "failing_node")
    w.add_edge("failing_node", END)
    return w


def _conditional_workflow() -> Workflow:
    w = Workflow()

    async def noop(agent, state, ctx):
        async def _gen():
            await asyncio.sleep(0.05)
            if False:
                yield

        return _gen()

    async def route_to_node_b(agent, state, ctx):
        return "node_b"

    w.add_node("node_a", noop)
    w.add_node("node_b", noop)
    w.add_edge(START, "node_a")
    w.add_conditional_edge("node_a", route_to_node_b)
    w.add_edge("node_b", END)
    return w


@pytest.fixture(autouse=True)
def reset_obs_after():
    yield
    obs.configure(None)


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_single_node_workflow_creates_trace():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    agent = _DummyAgent(name="integration_agent")
    ctx = InvocationContext(session_id="integ-test-1")
    workflow: Workflow = _noop_workflow()

    await workflow.ainvoke(agent, ctx=ctx, state=_state_with_message())
    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None, (
        "trace_id was not captured from workflow.start"
    )

    trace = _get_trace(lf, sink.last_trace_id, min_obs=2)
    assert trace.name == "integration_agent"
    assert trace.input == "hello world"
    assert trace.output == "hello world"


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_single_node_workflow_has_node_span():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    agent = _DummyAgent(name="integration_agent")
    ctx = InvocationContext(session_id="integ-test-1b")
    workflow: Workflow = _noop_workflow()

    await workflow.ainvoke(agent, ctx=ctx, state=_state_with_message())
    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None

    trace = _get_trace(lf, sink.last_trace_id, min_obs=2)
    obs_names = {o.name for o in trace.observations}
    assert "node_a" in obs_names, f"Expected 'node_a' span, got {obs_names}"


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_two_node_workflow_has_both_spans():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    agent = _DummyAgent(name="integration_agent")
    ctx = InvocationContext(session_id="integ-test-2")
    workflow = _two_node_workflow()

    await workflow.ainvoke(agent, ctx=ctx, state=_state_with_message())
    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None

    trace = _get_trace(lf, sink.last_trace_id, min_obs=3)
    obs_names = {o.name for o in trace.observations}
    assert "node_a" in obs_names, f"Expected 'node_a', got {obs_names}"
    assert "node_b" in obs_names, f"Expected 'node_b', got {obs_names}"


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_workflow_trace_and_observations_have_session_id():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    agent = _DummyAgent(name="integration_agent")
    ctx = InvocationContext(session_id="integ-test-session")
    workflow: Workflow = _two_node_workflow()

    await workflow.ainvoke(agent, ctx=ctx, state=_state_with_message())
    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None

    trace = _get_trace(lf, sink.last_trace_id, min_obs=3)

    assert getattr(trace, "session_id", None) == "integ-test-session"

    obs_session_ids = {
        ((getattr(o, "metadata", None) or {}).get("session_id"))
        for o in trace.observations
    }
    assert "integ-test-session" in obs_session_ids, (
        f"Expected observation metadata session_id propagation, got {obs_session_ids}"
    )


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_failing_node_span_has_error_level():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    agent = _DummyAgent(name="integration_agent")
    ctx = InvocationContext(session_id="integ-test-3")
    workflow: Workflow = _failing_workflow()

    with pytest.raises(RuntimeError, match="intentional failure"):
        await workflow.ainvoke(agent, ctx=ctx, state=_state_with_message())

    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None

    trace = _get_trace(lf, sink.last_trace_id, min_obs=2)
    error_obs = [o for o in trace.observations if o.level == "ERROR"]
    assert error_obs, "No ERROR-level observation found in trace"
    assert any("intentional failure" in (o.status_message or "") for o in error_obs)


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_llm_span_input_is_stored():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    messages = [{"role": "user", "content": "What is 2+2?"}]

    await obs.event(
        "workflow.start",
        session_id="integ-test-4",
        agent_name="integration_agent",
        input="What is 2+2?",
        start_at=START,
        max_steps=10,
    )
    await obs.event("node.start", node="my_node", run_step=1)

    async with obs.span("llm", model="gpt-4o-mini", input=messages) as span:
        span.update(output="4")

    await obs.event("node.end", node="my_node", run_step=1)
    await obs.event("workflow.end", steps=1, output="4")

    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None

    trace = _get_trace(lf, sink.last_trace_id, min_obs=3)
    llm_obs = [o for o in trace.observations if o.name == "llm"]
    assert llm_obs, "No 'llm' generation observation found in trace"
    assert llm_obs[0].input is not None, "llm observation has no input"
    assert llm_obs[0].model == "gpt-4o-mini", "llm observation has no model"
    assert str(llm_obs[0].output) == "4"
    assert str(trace.output) == "4"


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_tool_span_input_is_stored():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event(
        "workflow.start",
        session_id="integ-test-5",
        agent_name="integration_agent",
        input="Search for cats",
        start_at=START,
        max_steps=10,
    )
    await obs.event("node.start", node="my_node", run_step=1)

    tool_input = {"query": "cats"}
    tool_output = "Found 42 cat articles"

    async with obs.span(
        "web_search",
        as_type="tool",
        input=tool_input,
        metadata={"tool_call_id": "tool_123"},
    ) as span:
        span.update(output=tool_output)

    await obs.event("node.end", node="my_node", run_step=1)
    await obs.event("workflow.end", steps=1, output=tool_output)

    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None

    trace = _get_trace(lf, sink.last_trace_id, min_obs=3)
    tool_obs = [o for o in trace.observations if o.name == "web_search"]
    assert tool_obs, "No tool observation found in trace"
    assert tool_obs[0].input is not None, "tool observation has no input"
    assert tool_obs[0].output is not None, "tool observation has no output"
    assert trace.output == tool_output


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_conditional_router_appears_as_span():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    agent = _DummyAgent(name="integration_agent")
    ctx = InvocationContext(session_id="integ-test-router")
    workflow = _conditional_workflow()

    await workflow.ainvoke(agent, ctx=ctx, state=_state_with_message())
    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None

    # Expect: agent root + node_a + route_to_node_b router + node_b = 4 observations
    trace = _get_trace(lf, sink.last_trace_id, min_obs=4)
    obs_names = {o.name for o in trace.observations}
    assert "route_to_node_b" in obs_names, (
        f"Expected router span 'route_to_node_b' in trace, got {obs_names}"
    )

    router_obs = next(o for o in trace.observations if o.name == "route_to_node_b")
    assert str(router_obs.input) == "node_a", (
        f"Expected router input 'node_a', got {router_obs.input!r}"
    )
    assert str(router_obs.output) == "node_b", (
        f"Expected router output 'node_b', got {router_obs.output!r}"
    )


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.langfuse
async def test_span_error_marks_observation_as_error():
    lf = _langfuse_client()
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event(
        "workflow.start",
        session_id="integ-test-6",
        agent_name="integration_agent",
        input="test",
        start_at=START,
        max_steps=10,
    )
    await obs.event("node.start", node="my_node", run_step=1)

    with pytest.raises(ValueError, match="llm exploded"):
        async with obs.span(
            "llm", model="gpt-4o", input=[{"role": "user", "content": "test"}]
        ):
            raise ValueError("llm exploded")

    await obs.event("node.end", node="my_node", run_step=1)
    await obs.event("workflow.end", steps=1, output="failed")

    _flush_and_shutdown(lf)

    assert sink.last_trace_id is not None

    trace = _get_trace(lf, sink.last_trace_id, min_obs=3)
    error_obs = [
        o for o in trace.observations if o.level == "ERROR" and o.name == "llm"
    ]
    assert error_obs, "No ERROR-level 'llm' observation found"
    assert "llm exploded" in (error_obs[0].status_message or "")
