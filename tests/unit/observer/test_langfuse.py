from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ant_ai.observer import obs
from ant_ai.observer.integrations.langfuse import LangfuseSink


def make_observation(name: str = "obs") -> MagicMock:
    mock = MagicMock(name=name)
    mock.update = MagicMock()
    mock.end = MagicMock()
    mock.trace_id = f"trace-{name}"
    return mock


def make_cm(obs_mock: MagicMock) -> MagicMock:
    """Context manager mock whose __exit__ ends the wrapped observation."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=obs_mock)

    def _exit(*args):
        obs_mock.end()
        return False

    cm.__exit__ = MagicMock(side_effect=_exit)
    return cm


def make_langfuse_mock(*observations: MagicMock) -> MagicMock:
    """Langfuse client mock.

    If *observations* are provided they are returned in order from
    start_as_current_observation; otherwise the mock auto-generates values.
    """
    lf = MagicMock()
    lf.flush = MagicMock()
    if observations:
        lf.start_as_current_observation.side_effect = [make_cm(o) for o in observations]
    return lf


@pytest.fixture(autouse=True)
def reset_obs():
    yield
    obs.configure(None)


@pytest.mark.unit
async def test_workflow_start_creates_agent_observation():
    root = make_observation("root")
    lf = make_langfuse_mock(root)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event(
        "workflow.start",
        session_id="s1",
        agent_name="my_agent",
        start_at="START",
        max_steps=10,
        input="hello",
    )

    lf.start_as_current_observation.assert_called_once()
    kwargs = lf.start_as_current_observation.call_args.kwargs
    assert kwargs["as_type"] == "agent"
    assert kwargs["name"] == "my_agent"
    assert kwargs["input"] == "hello"
    assert kwargs["metadata"]["start_at"] == "START"
    assert kwargs["metadata"]["max_steps"] == 10
    assert sink.last_trace_id == "trace-root"


@pytest.mark.unit
async def test_node_start_creates_child_span():
    root = make_observation("root")
    node = make_observation("node")
    lf = make_langfuse_mock(root, node)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")
    await obs.event("node.start", node="planner", run_step=1)

    assert lf.start_as_current_observation.call_count == 2
    kwargs = lf.start_as_current_observation.call_args_list[1].kwargs
    assert kwargs["as_type"] == "span"
    assert kwargs["name"] == "planner"
    assert kwargs["metadata"]["run_step"] == 1


@pytest.mark.unit
async def test_node_end_ends_matching_node():
    root = make_observation("root")
    node = make_observation("node")
    lf = make_langfuse_mock(root, node)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")
    await obs.event("node.start", node="planner", run_step=1)
    await obs.event("node.end", node="planner", run_step=1)

    node.end.assert_called_once()


@pytest.mark.unit
async def test_node_error_updates_node_and_closes_workflow():
    root = make_observation("root")
    node = make_observation("node")
    lf = make_langfuse_mock(root, node)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")
    await obs.event("node.start", node="B", run_step=1)

    error = ValueError("boom")
    await obs.exception("node.error", error, node="B", run_step=1)

    node.update.assert_called_once_with(
        level="ERROR",
        status_message="boom",
        metadata={"error_type": "ValueError"},
    )
    node.end.assert_called_once()

    root.update.assert_called_once_with(
        level="ERROR",
        status_message="boom",
        metadata={"error_type": "ValueError"},
    )
    root.end.assert_called_once()
    lf.flush.assert_called_once()


@pytest.mark.unit
async def test_workflow_end_updates_root_output_ends_root_and_flushes():
    root = make_observation("root")
    lf = make_langfuse_mock(root)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")
    await obs.event("workflow.end", output="4", steps=3, finish_reason="done")

    root.update.assert_called_once_with(
        output="4", metadata={"steps": 3, "finish_reason": "done"}
    )
    root.end.assert_called_once()
    lf.flush.assert_called_once()


@pytest.mark.unit
async def test_workflow_end_closes_open_node_then_root():
    root = make_observation("root")
    node = make_observation("node")
    lf = make_langfuse_mock(root, node)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")
    await obs.event("node.start", node="planner", run_step=1)
    await obs.event("workflow.end", output="done")

    root.update.assert_any_call(output="done")
    node.end.assert_called_once()
    root.end.assert_called_once()
    lf.flush.assert_called_once()


@pytest.mark.unit
async def test_workflow_end_without_output_does_not_write_output():
    root = make_observation("root")
    lf = make_langfuse_mock(root)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")
    await obs.event("workflow.end", steps=3)

    assert not any("output" in call.kwargs for call in root.update.call_args_list)
    root.end.assert_called_once()
    lf.flush.assert_called_once()


@pytest.mark.unit
async def test_node_edge_event_does_not_mutate_root_or_current_observation():
    root = make_observation("root")
    node = make_observation("node")
    lf = make_langfuse_mock(root, node)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")
    await obs.event("node.start", node="node_b", run_step=2)
    await obs.event("node.end", node="node_b", run_step=2)
    await obs.event("node.edge.static", src="node_b", dst="END")

    root.update.assert_not_called()
    node.update.assert_not_called()


@pytest.mark.unit
async def test_span_llm_creates_generation_on_current_parent():
    root = make_observation("root")
    gen = make_observation("generation")
    lf = make_langfuse_mock(root, gen)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")

    async with obs.span("llm", input="prompt", model="gpt-test", messages=5):
        pass

    assert lf.start_as_current_observation.call_count == 2
    kwargs = lf.start_as_current_observation.call_args_list[1].kwargs
    assert kwargs["as_type"] == "generation"
    assert kwargs["name"] == "llm"
    assert kwargs["input"] == "prompt"
    assert kwargs["model"] == "gpt-test"
    assert kwargs["metadata"]["messages"] == 5
    gen.end.assert_called_once()


@pytest.mark.unit
async def test_span_tool_creates_tool_observation():
    root = make_observation("root")
    tool = make_observation("tool")
    lf = make_langfuse_mock(root, tool)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")

    async with obs.span("tool", input="cats", tool_name="search"):
        pass

    assert lf.start_as_current_observation.call_count == 2
    kwargs = lf.start_as_current_observation.call_args_list[1].kwargs
    assert kwargs["as_type"] == "tool"
    assert kwargs["name"] == "tool"
    assert kwargs["input"] == "cats"
    assert kwargs["metadata"]["tool_name"] == "search"
    tool.end.assert_called_once()


@pytest.mark.unit
async def test_span_without_parent_uses_client_start_observation():
    leaf = make_observation("leaf")
    lf = make_langfuse_mock(leaf)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    async with obs.span("tool", input="cats", tool_name="search"):
        pass

    lf.start_as_current_observation.assert_called_once()
    kwargs = lf.start_as_current_observation.call_args.kwargs
    assert kwargs["as_type"] == "tool"
    assert kwargs["name"] == "tool"
    assert kwargs["input"] == "cats"
    assert kwargs["metadata"]["tool_name"] == "search"
    leaf.end.assert_called_once()


@pytest.mark.unit
async def test_span_error_updates_level_and_ends():
    root = make_observation("root")
    gen = make_observation("generation")
    lf = make_langfuse_mock(root, gen)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")

    with pytest.raises(RuntimeError, match="llm failed"):
        async with obs.span("llm", input="prompt", messages=3):
            raise RuntimeError("llm failed")

    gen.update.assert_called_once_with(level="ERROR", status_message="llm failed")
    gen.end.assert_called_once()


@pytest.mark.unit
async def test_generic_exception_updates_current_observation_only():
    root = make_observation("root")
    node = make_observation("node")
    lf = make_langfuse_mock(root, node)
    sink = LangfuseSink(langfuse=lf)
    obs.configure(sink)

    await obs.event("workflow.start", session_id="s1", agent_name="a")
    await obs.event("node.start", node="planner", run_step=1)

    error = RuntimeError("bad step")
    await obs.exception("step.error", error, node="planner", run_step=1)

    node.update.assert_called_once_with(
        level="ERROR",
        status_message="step.error: bad step",
        metadata={"error_type": "RuntimeError"},
    )
    node.end.assert_not_called()
    root.end.assert_not_called()


@pytest.mark.unit
def test_langfuse_sink_instantiates():
    lf = MagicMock()
    sink = LangfuseSink(langfuse=lf)
    assert hasattr(sink, "event")
    assert hasattr(sink, "exception")
    assert hasattr(sink, "span")
    assert hasattr(sink, "propagation_headers")
    assert hasattr(sink, "attach_propagation_context")
