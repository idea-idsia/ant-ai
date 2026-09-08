from __future__ import annotations

from typing import Any

import pytest

from ant_ai.core.message import Message
from ant_ai.memory.protocol import Memory
from ant_ai.topology.activate import RecordingMemory, memory_node_id, record_support
from ant_ai.topology.evaluate import leaked, support_accuracy
from ant_ai.topology.log import RewriteLog
from ant_ai.topology.state import StateGraph

pytestmark = [pytest.mark.unit, pytest.mark.topology]


class FakeMemory(Memory):
    """Returns whatever it was told to, and remembers what it was asked."""

    results: list[Message] = []
    queries: list[str] = []
    written: list[Message] = []

    async def retrieve(
        self, query: str, *, top_k: int = 5, ctx: Any = None, **kwargs: Any
    ) -> list[Message]:
        self.queries.append(query)
        return list(self.results)

    async def update(
        self, messages: list[Message], *, ctx: Any = None, **kw: Any
    ) -> None:
        self.written.extend(messages)


def _memory(**kwargs: Any) -> RecordingMemory:
    inner = FakeMemory(results=[], queries=[], written=[])
    return RecordingMemory(inner=inner, state=StateGraph(), **kwargs)


def test_ids_are_content_addressed_so_a_replay_is_comparable() -> None:
    """A counter would make every replay produce a different graph and every
    cross-run metric meaningless."""
    assert memory_node_id("same") == memory_node_id("same")
    assert memory_node_id("same") != memory_node_id("other")


async def test_a_write_inserts_a_memory_node_and_still_reaches_the_backend() -> None:
    memory = _memory()

    await memory.update([Message(role="user", content="the sky is blue")])

    assert [m.content for m in memory.inner.written] == ["the sky is blue"]
    node = memory.state.nodes[memory_node_id("the sky is blue")]
    assert node.kind == "memory" and node.attrs["content"] == "the sky is blue"


async def test_a_read_leaves_a_support_subgraph_behind() -> None:
    memory = _memory()
    await memory.update([Message(role="user", content="fact")])
    memory.inner.results = [Message(role="user", content="fact")]

    got = await memory.retrieve("what?")

    assert [m.content for m in got] == ["fact"]
    support = memory.state.supports[-1]
    assert support.query == "what?"
    assert support.nodes == (memory_node_id("fact"),)


async def test_the_leakage_guard_drops_evidence_written_after_the_decision() -> None:
    """A backend ranks by similarity and has no notion of when a decision is
    being made."""
    memory = _memory()
    memory.tick(5)
    await memory.update([Message(role="user", content="late")])
    memory.inner.results = [Message(role="user", content="late")]

    memory.tick(2)
    got = await memory.retrieve("what?")

    assert got == []


async def test_the_guard_can_be_turned_off_and_the_leak_is_then_reportable() -> None:
    memory = _memory(leakage_guard=False)
    memory.tick(5)
    await memory.update([Message(role="user", content="late")])
    memory.inner.results = [Message(role="user", content="late")]

    memory.tick(2)
    await memory.retrieve("what?")

    assert leaked(memory.state, memory.state.supports[-1]) == (memory_node_id("late"),)


async def test_reads_and_writes_land_in_the_log_when_there_is_one() -> None:
    """A cascade that begins with a retrieval is only legible if the retrieval
    is in the record next to what it caused."""
    log = RewriteLog()
    memory = _memory(log=log)

    await memory.update([Message(role="user", content="fact")])
    await memory.retrieve("what?")

    assert [e.rewrite.op for e in log.entries] == ["insert", "activate"]


def test_record_support_scores_against_a_gold_set() -> None:
    state = StateGraph()
    record_support(state, kind="memory", id="s1", nodes=("a", "b"), at=0)
    record_support(state, kind="memory", id="s2", nodes=("c",), at=0)

    assert support_accuracy(state, {"s1": {"a", "b"}}) == 1.0
    assert support_accuracy(state, {"s1": {"a"}}) == pytest.approx(2 / 3)


def test_an_unlabelled_activation_is_unmeasured_not_wrong() -> None:
    state = StateGraph()
    record_support(state, kind="memory", id="s1", nodes=("a",), at=0)

    assert support_accuracy(state, {}) == 1.0
