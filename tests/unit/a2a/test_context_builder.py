from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus

from ant_ai.a2a.context_builder import HistoryRequestContextBuilder
from ant_ai.observer import obs


class SpySink:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def event(self, name: str, **fields) -> None:
        self.events.append((name, fields))

    async def exception(self, name: str, error: Exception, **fields) -> None:
        pass

    @asynccontextmanager
    async def span(self, name: str, **attrs):
        yield


@pytest.fixture
def spy_sink():
    spy = SpySink()
    obs.configure(spy)
    yield spy
    obs.configure(None)


def _task(task_id: str, context_id: str = "ctx-1") -> Task:
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
    )


@pytest.mark.unit
async def test_unresolved_reference_is_reported_and_dropped(spy_sink: SpySink):
    """A referenced task id the store cannot return is reported via obs, and the
    resolvable ones are still returned unchanged."""
    store = InMemoryTaskStore()
    ctx = MagicMock()
    await store.save(_task("t-1"), context=ctx)

    builder = HistoryRequestContextBuilder(task_store=store)
    tasks = await builder.collect_all_referenced_tasks(
        context=ctx, task_store=store, initial_ids=["t-1", "t-missing"]
    )

    assert [t.id for t in tasks] == ["t-1"]
    unresolved = [
        f for n, f in spy_sink.events if n == "a2a.referenced_task.unresolved"
    ]
    assert unresolved == [{"task_id": "t-missing"}]


@pytest.mark.unit
async def test_all_references_resolved_reports_nothing(spy_sink: SpySink):
    store = InMemoryTaskStore()
    ctx = MagicMock()
    await store.save(_task("t-1"), context=ctx)
    await store.save(_task("t-2"), context=ctx)

    builder = HistoryRequestContextBuilder(task_store=store)
    tasks = await builder.collect_all_referenced_tasks(
        context=ctx, task_store=store, initial_ids=["t-1", "t-2"]
    )

    assert {t.id for t in tasks} == {"t-1", "t-2"}
    assert "a2a.referenced_task.unresolved" not in [n for n, _ in spy_sink.events]


@pytest.mark.unit
async def test_unresolved_reference_without_sink_does_not_raise():
    obs.configure(None)
    store = InMemoryTaskStore()
    ctx = MagicMock()

    builder = HistoryRequestContextBuilder(task_store=store)
    tasks = await builder.collect_all_referenced_tasks(
        context=ctx, task_store=store, initial_ids=["t-missing"]
    )

    assert tasks == []
