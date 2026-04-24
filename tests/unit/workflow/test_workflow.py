from __future__ import annotations

import pytest

from ant_ai.core.events import AgentEvent, Event
from ant_ai.core.types import InvocationContext, State
from ant_ai.workflow.workflow import END, START, Workflow


async def collect_stream(workflow, agent, ctx, start_at=START, state=None):
    events: list[Event] = []
    async for ev in workflow.stream(agent, ctx=ctx, start_at=start_at, state=state):
        if ev.origin.layer != "workflow":
            events.append(ev)
    return events


@pytest.mark.unit
async def test_add_node_rejects_reserved_names():
    w = Workflow()

    async def action(agent, state, ctx):
        async def gen():
            if False:
                yield  # pragma: no cover

        return gen()

    with pytest.raises(ValueError, match="reserved"):
        w.add_node("START", action)
    with pytest.raises(ValueError, match="reserved"):
        w.add_node("end", action)  # upper-cased to END is reserved


@pytest.mark.unit
async def test_add_node_rejects_duplicate():
    w = Workflow()

    async def action(agent, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", action)
    with pytest.raises(ValueError, match="already exists"):
        w.add_node("A", action)


@pytest.mark.unit
async def test_add_node_rejects_non_callable():
    w = Workflow()
    with pytest.raises(TypeError, match="callable"):
        w.add_node("A", action=123)  # type: ignore[arg-type]


@pytest.mark.unit
async def test_add_edge_unknown_node_raises():
    w = Workflow()

    async def action(agent, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", action)

    with pytest.raises(ValueError, match="Unknown node"):
        w.add_edge("B", "A")
    with pytest.raises(ValueError, match="Unknown node"):
        w.add_edge("A", "B")


@pytest.mark.unit
async def test_add_edge_disallows_multiple_outgoing():
    w = Workflow()

    async def action(agent, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", action).add_node("B", action).add_node("C", action)

    w.add_edge("A", "B")
    with pytest.raises(ValueError, match="already has an outgoing edge"):
        w.add_edge("A", "C")


@pytest.mark.unit
async def test_cannot_mix_static_and_conditional_edges():
    w = Workflow()

    async def action(agent, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    async def router(agent, state, ctx):
        return "B"

    w.add_node("A", action).add_node("B", action)

    w.add_edge("A", "B")
    with pytest.raises(ValueError, match="Cannot mix static and conditional edges"):
        w.add_conditional_edge("A", router)

    w2 = Workflow()
    w2.add_node("A", action).add_node("B", action)
    w2.add_conditional_edge("A", router)
    with pytest.raises(ValueError, match="Cannot mix static and conditional edges"):
        w2.add_edge("A", "B")


@pytest.mark.unit
async def test_add_conditional_edge_rejects_non_callable_router():
    w = Workflow()

    async def action(agent, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", action)

    with pytest.raises(TypeError, match="Router must be callable"):
        w.add_conditional_edge("A", router=123)  # type: ignore[arg-type]


@pytest.mark.unit
async def test_validate_graph_requires_start_outgoing(agent, seeded_state):
    w = Workflow()

    async def action(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", action)

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(ValueError, match="START must have an outgoing edge"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))


@pytest.mark.unit
async def test_next_static_edge_ok(agent, seeded_state):
    w = Workflow()

    async def action(a, state, ctx):
        async def gen():
            # no events, no state changes
            if False:
                yield

        return gen()

    w.add_node("A", action)
    w.add_edge(START, "A")
    w.add_edge("A", END)

    ctx = InvocationContext(session_id="s1")

    st = await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))
    assert isinstance(st, State)
    assert st.last_message.content == "s0"


@pytest.mark.unit
async def test_next_conditional_edge_validates_return_non_empty_str_and_known_node(
    agent, seeded_state
):
    w = Workflow()

    async def action(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    async def bad_router_empty(a, state, ctx):
        return ""

    async def bad_router_unknown(a, state, ctx):
        return "NOPE"

    w.add_node("A", action)
    w.add_edge(START, "A")
    w.add_conditional_edge("A", bad_router_empty)

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(RuntimeError, match="Router must return a non-empty str"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))

    w2 = Workflow()
    w2.add_node("A", action)
    w2.add_edge(START, "A")
    w2.add_conditional_edge("A", bad_router_unknown)

    with pytest.raises(RuntimeError, match="Unknown node"):
        await w2.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))


@pytest.mark.unit
async def test_next_no_outgoing_edge_raises(agent, seeded_state):
    w = Workflow()

    async def action(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", action)
    w.add_edge(START, "A")

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(RuntimeError, match=r"No outgoing edge from 'A'"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))


@pytest.mark.unit
async def test_run_node_updates_state_on_return(agent, seeded_state):
    w = Workflow()

    async def node_a(a, state, ctx):
        async def gen():
            if False:
                yield

        # return a new State via "return" path (non-async-iter)
        return seeded_state("s1")

    w.add_node("A", node_a)
    w.add_edge(START, "A")
    w.add_edge("A", END)

    ctx = InvocationContext(session_id="s1")

    final = await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))
    assert final.last_message.content == "s1"


@pytest.mark.unit
async def test_run_node_yields_events_and_updates_state_from_async_iterator(
    agent, seeded_state
):
    w = Workflow()

    async def node_a(a, state, ctx):
        async def gen():
            yield AgentEvent(content="e1")
            yield seeded_state("s1")
            yield AgentEvent(content="e2")
            yield seeded_state("s2")

        return gen()

    w.add_node("A", node_a)
    w.add_edge(START, "A")
    w.add_edge("A", END)

    ctx = InvocationContext(session_id="s1")

    events = await collect_stream(w, agent, ctx, state=seeded_state("s0"))
    assert [e.content for e in events] == ["e1", "e2"]

    final = await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))
    assert final.last_message.content == "s2"


@pytest.mark.unit
async def test_run_node_rejects_invalid_yield_type(agent, seeded_state):
    w = Workflow()

    async def node_a(a, state, ctx):
        async def gen():
            yield 123

        return gen()

    w.add_node("A", node_a)
    w.add_edge(START, "A")
    w.add_edge("A", END)

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(RuntimeError, match=r"Invalid yield from node 'A'"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))


@pytest.mark.unit
async def test_run_node_rejects_invalid_return_type(agent, seeded_state):
    w = Workflow()

    async def node_a(a, state, ctx):
        return 123

    w.add_node("A", node_a)
    w.add_edge(START, "A")
    w.add_edge("A", END)

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(RuntimeError, match=r"Invalid return from node 'A'"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))


@pytest.mark.unit
async def test_max_steps_exceeded_raises(agent, seeded_state):
    w = Workflow(max_steps=3)

    async def node_a(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", node_a)
    w.add_edge(START, "A")
    w.add_edge("A", "A")  # infinite loop

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(RuntimeError, match="Max steps exceeded"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))


@pytest.mark.unit
async def test_start_is_not_executed_as_node(agent, seeded_state):
    w = Workflow()
    called = {"count": 0}

    async def node_a(a, state, ctx):
        called["count"] += 1

        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", node_a)
    w.add_edge(START, "A")
    w.add_edge("A", END)

    ctx = InvocationContext(session_id="s1")

    await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))
    assert called["count"] == 1
