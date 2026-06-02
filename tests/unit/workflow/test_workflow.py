from __future__ import annotations

import pytest

from ant_ai.core.events import AgentEvent, Event
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.workflow.workflow import END, START, Workflow


def _msg(content: str = "seed") -> Message:
    return Message(role="user", content=content)


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

    with pytest.raises(ValueError, match=r"dead end"):
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
    w.add_conditional_edge("A", lambda a, s, c: "A")  # always loops back

    ctx = InvocationContext(session_id="s1")

    with pytest.raises(RuntimeError, match="Max steps exceeded"):
        await w.ainvoke(agent, ctx=ctx, state=seeded_state("s0"))


@pytest.mark.unit
async def test_custom_state_class_is_stored():
    class MyState(State):
        count: int = 0

    w = Workflow(state=MyState)
    assert w.state is MyState


@pytest.mark.unit
async def test_default_workflow_uses_base_state():
    assert Workflow().state is State


@pytest.mark.unit
async def test_ainvoke_returns_custom_state_instance(agent):
    class MyState(State):
        count: int = 0

    async def noop(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w = Workflow(state=MyState)
    w.add_node("A", noop).add_edge(START, "A").add_edge("A", END)

    result = await w.ainvoke(
        agent, ctx=InvocationContext(session_id="s"), state=MyState(messages=[_msg()])
    )
    assert isinstance(result, MyState)


@pytest.mark.unit
async def test_create_state_returns_custom_state_instance():
    class MyState(State):
        count: int = 0

    st = Workflow(state=MyState).create_state()
    assert isinstance(st, MyState)
    assert st.count == 0


@pytest.mark.unit
async def test_node_receives_custom_state_instance(agent):
    class MyState(State):
        count: int = 0

    received: list[type] = []

    async def node(a, state: MyState, ctx):
        received.append(type(state))

        async def gen():
            if False:
                yield

        return gen()

    w = Workflow(state=MyState)
    w.add_node("A", node).add_edge(START, "A").add_edge("A", END)

    await w.ainvoke(
        agent, ctx=InvocationContext(session_id="s"), state=MyState(messages=[_msg()])
    )
    assert received == [MyState]


@pytest.mark.unit
async def test_custom_fields_mutated_in_node_are_returned(agent):
    class MyState(State):
        count: int = 0

    async def node(a, state: MyState, ctx):
        async def gen():
            state.count += 1
            yield state

        return gen()

    w = Workflow(state=MyState)
    w.add_node("A", node).add_edge(START, "A").add_edge("A", END)

    result = await w.ainvoke(
        agent, ctx=InvocationContext(session_id="s"), state=MyState(messages=[_msg()])
    )
    assert result.count == 1


@pytest.mark.unit
async def test_custom_fields_persist_across_nodes(agent):
    class MyState(State):
        count: int = 0

    async def add_one(a, state: MyState, ctx):
        async def gen():
            state.count += 1
            yield state

        return gen()

    async def add_ten(a, state: MyState, ctx):
        async def gen():
            state.count += 10
            yield state

        return gen()

    w = Workflow(state=MyState)
    w.add_node("A", add_one).add_node("B", add_ten)
    w.add_edge(START, "A").add_edge("A", "B").add_edge("B", END)

    result = await w.ainvoke(
        agent, ctx=InvocationContext(session_id="s"), state=MyState(messages=[_msg()])
    )
    assert result.count == 11


@pytest.mark.unit
async def test_router_receives_custom_state_instance(agent):
    class MyState(State):
        count: int = 0

    received: list[type] = []

    async def noop(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    def router(a, state: MyState, ctx):
        received.append(type(state))
        return END

    w = Workflow(state=MyState)
    w.add_node("A", noop).add_edge(START, "A").add_conditional_edge("A", router)

    await w.ainvoke(
        agent, ctx=InvocationContext(session_id="s"), state=MyState(messages=[_msg()])
    )
    assert received == [MyState]


@pytest.mark.unit
async def test_initial_custom_state_passed_to_ainvoke_is_used(agent):
    class MyState(State):
        count: int = 0

    async def noop(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w = Workflow(state=MyState)
    w.add_node("A", noop).add_edge(START, "A").add_edge("A", END)

    result = await w.ainvoke(
        agent,
        ctx=InvocationContext(session_id="s"),
        state=MyState(count=42, messages=[_msg()]),
    )
    assert isinstance(result, MyState)
    assert result.count == 42


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


async def _noop(a, state, ctx):
    async def gen():
        if False:
            yield

    return gen()


@pytest.mark.unit
def test_check_valid_graph_passes():
    w = Workflow()
    w.add_node("A", _noop)
    w.add_edge(START, "A")
    w.add_edge("A", END)
    w.check()  # must not raise


@pytest.mark.unit
def test_check_no_start_edge():
    w = Workflow()
    w.add_node("A", _noop)
    w.add_edge("A", END)
    with pytest.raises(ValueError, match="START must have an outgoing edge"):
        w.check()


@pytest.mark.unit
def test_check_dead_end_node():
    w = Workflow()
    w.add_node("A", _noop)
    w.add_node("B", _noop)
    w.add_edge(START, "A")
    w.add_edge("A", "B")
    # B has no outgoing edge
    with pytest.raises(ValueError, match="dead end"):
        w.check()


@pytest.mark.unit
def test_check_end_unreachable():
    w = Workflow()
    w.add_node("A", _noop)
    w.add_edge(START, "A")
    w.add_edge("A", "A")  # cycle, never reaches END, no conditional edges
    with pytest.raises(ValueError, match="END is unreachable"):
        w.check()


@pytest.mark.unit
def test_check_unreachable_node():
    w = Workflow()
    w.add_node("A", _noop)
    w.add_node("orphan", _noop)
    w.add_edge(START, "A")
    w.add_edge("A", END)
    w.add_edge("orphan", END)  # orphan exists but nothing points to it
    with pytest.raises(ValueError, match="unreachable from START"):
        w.check()


@pytest.mark.unit
def test_check_reports_all_errors_at_once():
    w = Workflow()
    w.add_node("A", _noop)
    w.add_node("B", _noop)
    # no START edge, A has no outgoing edge, B has no outgoing edge, END unreachable
    with pytest.raises(ValueError) as exc_info:
        w.check()
    msg = str(exc_info.value)
    assert "START must have an outgoing edge" in msg
    assert "Node 'A' has no outgoing edge" in msg
    assert "Node 'B' has no outgoing edge" in msg


@pytest.mark.unit
def test_check_conditional_edge_skips_end_reachability():
    w = Workflow()
    w.add_node("A", _noop)
    w.add_edge(START, "A")
    w.add_conditional_edge("A", lambda a, s, c: END)
    w.check()  # conditional edge can return END — must not raise


@pytest.mark.unit
def test_check_conditional_edge_makes_all_nodes_reachable():
    w = Workflow()
    w.add_node("A", _noop)
    w.add_node("B", _noop)
    w.add_edge(START, "A")
    w.add_conditional_edge("A", lambda a, s, c: "B")
    w.add_edge("B", END)
    w.check()  # B is reachable via A's conditional edge — must not raise
