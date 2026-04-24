from __future__ import annotations

import pytest

from ant_ai.core.types import State
from ant_ai.workflow.workflow import END, START, Workflow


class DummyAgent:
    def __init__(self, name: str = "dummy"):
        self.name = name


class DummyMsg:
    def __init__(self, content: str):
        self.content = content


@pytest.fixture
def agent():
    return DummyAgent()


@pytest.fixture
def seeded_state():
    """Factory fixture: call seeded_state(content) to get a State."""

    def _make(content: str = "seed") -> State:
        st = State()
        st.messages.append(DummyMsg(content))
        return st

    return _make


@pytest.fixture
def make_noop_workflow():
    def _make() -> Workflow:
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

    return _make
