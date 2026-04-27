from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from ant_ai.core.types import InvocationContext
from ant_ai.observer import obs
from ant_ai.observer.integrations.log import StructlogSink
from ant_ai.observer.integrations.otel import OTelSink
from ant_ai.workflow.workflow import END, START, Workflow


class SpySink:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.exceptions: list[tuple[str, Exception, dict]] = []

    async def event(self, name: str, **fields) -> None:
        self.events.append((name, fields))

    async def exception(self, name: str, error: Exception, **fields) -> None:
        self.exceptions.append((name, error, fields))

    @asynccontextmanager
    async def span(self, name: str, **attrs):
        yield

    def event_names(self) -> list[str]:
        return [name for name, _ in self.events]

    def exception_names(self) -> list[str]:
        return [name for name, _, _ in self.exceptions]


@pytest.fixture(autouse=True)
def spy_sink():
    spy = SpySink()
    obs.configure(spy)
    yield spy
    obs.configure(None)


@pytest.mark.unit
async def test_no_sink_runs_without_error(agent, seeded_state, noop_workflow):
    obs.configure(None)
    w = noop_workflow
    ctx = InvocationContext(session_id="s1")
    await w.ainvoke(agent, ctx=ctx, state=seeded_state())


@pytest.mark.unit
async def test_event_order_single_node(
    spy_sink: SpySink, agent, seeded_state, noop_workflow
):
    w = noop_workflow
    ctx = InvocationContext(session_id="s1")

    await w.ainvoke(agent, ctx=ctx, state=seeded_state())

    names = spy_sink.event_names()
    assert names[0] == "workflow.start"
    assert "node.start" in names
    assert "node.end" in names
    assert "node.edge.static" in names
    assert names[-1] == "workflow.end"
    assert names.index("node.start") < names.index("node.end")


@pytest.mark.unit
async def test_correct_node_in_event_fields(
    spy_sink: SpySink, agent, seeded_state, noop_workflow
):
    w = noop_workflow
    ctx = InvocationContext(session_id="s1")

    await w.ainvoke(agent, ctx=ctx, state=seeded_state())

    node_starts = [(n, f) for n, f in spy_sink.events if n == "node.start"]
    assert len(node_starts) == 1
    assert node_starts[0][1]["node"] == "A"


@pytest.mark.unit
async def test_node_error_exception_emitted(spy_sink: SpySink, agent, seeded_state):
    w = Workflow()

    async def failing(a, state, ctx):
        async def gen():
            raise ValueError("boom")
            if False:
                yield

        return gen()

    w.add_node("B", failing)
    w.add_edge(START, "B")
    w.add_edge("B", END)

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(ValueError, match="boom"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state())

    assert "node.error" in spy_sink.exception_names()
    node_errors = [(n, e, f) for n, e, f in spy_sink.exceptions if n == "node.error"]
    assert node_errors[0][2]["node"] == "B"


@pytest.mark.unit
async def test_router_edge_event(spy_sink: SpySink, agent, seeded_state):
    w = Workflow()

    async def noop(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    async def router(a, state, ctx):
        return END

    w.add_node("C", noop)
    w.add_edge(START, "C")
    w.add_conditional_edge("C", router)

    ctx = InvocationContext(session_id="s1")

    await w.ainvoke(agent, ctx=ctx, state=seeded_state())

    router_events = [(n, f) for n, f in spy_sink.events if n == "node.edge.router"]
    assert len(router_events) == 1
    assert router_events[0][1]["src"] == "C"
    assert router_events[0][1]["dst"] == END


@pytest.mark.unit
async def test_workflow_start_fields(
    spy_sink: SpySink, agent, seeded_state, noop_workflow
):
    w = noop_workflow
    ctx = InvocationContext(session_id="s1")

    await w.ainvoke(agent, ctx=ctx, state=seeded_state())

    starts = [(n, f) for n, f in spy_sink.events if n == "workflow.start"]
    assert len(starts) == 1
    fields = starts[0][1]
    assert fields["start_at"] == START
    assert "max_steps" in fields


@pytest.mark.unit
async def test_max_steps_event_emitted(spy_sink: SpySink, agent, seeded_state):
    w = Workflow(max_steps=3)

    async def noop(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", noop)
    w.add_edge(START, "A")
    w.add_edge("A", "A")  # infinite loop

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(RuntimeError, match="Max steps exceeded"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state())

    assert "workflow.max_steps" in spy_sink.event_names()


@pytest.mark.unit
def test_structlog_sink_instantiates():
    sink = StructlogSink()
    assert hasattr(sink, "event")
    assert hasattr(sink, "exception")
    assert hasattr(sink, "span")


@pytest.mark.unit
def test_otel_sink_instantiates():
    from unittest.mock import MagicMock

    sink = OTelSink(tracer=MagicMock())
    assert hasattr(sink, "event")
    assert hasattr(sink, "exception")
    assert hasattr(sink, "span")
