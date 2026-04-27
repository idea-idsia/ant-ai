from __future__ import annotations

import pytest

from ant_ai.workflow.workflow import END, START, Workflow


@pytest.fixture
def noop_workflow() -> Workflow:
    """Simple workflow: START → A → END."""
    w = Workflow()

    async def noop(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    w.add_node("A", noop)
    w.add_edge(START, "A")
    w.add_edge("A", END)
    return w


@pytest.fixture
def conditional_workflow() -> Workflow:
    """Workflow with a conditional edge: START → A → ◇my_router → B or END."""
    w = Workflow()

    async def noop(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    async def noop_b(a, state, ctx):
        async def gen():
            if False:
                yield

        return gen()

    def my_router(a, state, ctx):
        if True:
            return "B"
        return "END"

    w.add_node("A", noop)
    w.add_node("B", noop_b)
    w.add_edge(START, "A")
    w.add_conditional_edge("A", my_router)
    return w
