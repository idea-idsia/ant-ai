from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ant_ai.observer.obs import ObservabilitySingleton


@pytest.fixture
def fresh_obs() -> ObservabilitySingleton:
    return ObservabilitySingleton()


@pytest.mark.unit
async def test_event_noop_when_no_sink(fresh_obs: ObservabilitySingleton):
    # Must not raise even though no sink is set
    await fresh_obs.event("any.event", foo="bar")


@pytest.mark.unit
async def test_exception_noop_when_no_sink(fresh_obs: ObservabilitySingleton):
    await fresh_obs.exception("any.error", ValueError("oops"))


@pytest.mark.unit
async def test_span_returns_noop_context_manager_when_no_sink(
    fresh_obs: ObservabilitySingleton,
):
    async with fresh_obs.span("test_span") as s:
        assert hasattr(s, "update")


@pytest.mark.unit
async def test_event_calls_sink_when_configured(fresh_obs: ObservabilitySingleton):
    sink = MagicMock()
    sink.event = AsyncMock(return_value=None)
    fresh_obs.configure(sink)

    await fresh_obs.event("my.event", x=1)

    sink.event.assert_called_once_with("my.event", x=1)


@pytest.mark.unit
async def test_event_suppresses_sink_exception(fresh_obs: ObservabilitySingleton):
    sink = MagicMock()
    sink.event = AsyncMock(side_effect=RuntimeError("sink broke"))
    fresh_obs.configure(sink)

    # Must not propagate
    await fresh_obs.event("any.event")


@pytest.mark.unit
async def test_exception_suppresses_sink_exception(fresh_obs: ObservabilitySingleton):
    sink = MagicMock()
    sink.exception = AsyncMock(side_effect=RuntimeError("sink broke"))
    fresh_obs.configure(sink)

    await fresh_obs.exception("any.error", ValueError("oops"))


@pytest.mark.unit
async def test_bind_merges_fields_into_event(fresh_obs: ObservabilitySingleton):
    received: list[dict] = []

    class CaptureSink:
        async def event(self, name: str, **fields) -> None:
            received.append(fields)

    fresh_obs.configure(CaptureSink())

    with fresh_obs.bind(session_id="s1"):
        await fresh_obs.event("step.start")

    assert received[0]["session_id"] == "s1"


@pytest.mark.unit
async def test_bind_restores_context_after_exit(fresh_obs: ObservabilitySingleton):
    with fresh_obs.bind(key="outer"):
        pass

    assert fresh_obs._ctx.get() == {}


@pytest.mark.unit
async def test_bind_nesting_inner_values_do_not_persist_after_exit(
    fresh_obs: ObservabilitySingleton,
):
    received: list[dict] = []

    class CaptureSink:
        async def event(self, name: str, **fields) -> None:
            received.append(dict(fields))

    fresh_obs.configure(CaptureSink())

    with fresh_obs.bind(outer_key="outer"):
        with fresh_obs.bind(inner_key="inner"):
            await fresh_obs.event("inside_inner")

        await fresh_obs.event("inside_outer")

    assert received[0].get("inner_key") == "inner"
    assert received[0].get("outer_key") == "outer"

    assert "inner_key" not in received[1]
    assert received[1].get("outer_key") == "outer"


@pytest.mark.unit
async def test_bind_tolerates_being_left_from_another_task(
    fresh_obs: ObservabilitySingleton,
):
    """An async generator that entered `bind()` can be closed by a different
    task -- an A2A executor cancelling a stream, or the loop's generator
    finalizer -- which runs the `finally` in another Context. The reset of
    the ContextVar token must not raise there: the context being restored
    belongs to the task that set it, and that task's copy is already gone."""

    async def stream():
        """Hold a binding across a yield, as `BaseAgent.stream` does."""
        with fresh_obs.bind(agent_name="a"):
            yield 1
            yield 2

    async def close_elsewhere() -> dict:
        """Close the generator and report this task's context afterwards."""
        await gen.aclose()
        return fresh_obs._ctx.get()

    gen = stream()
    assert await anext(gen) == 1

    # create_task runs in a COPY of the current context, so the token was
    # made in a different Context from the one the reset runs in. The closing
    # task's copy carried the binding too and must come back clean; the
    # entering task's context is not reachable from there and is left alone.
    seen_after_close = await asyncio.create_task(close_elsewhere())

    assert seen_after_close == {}
    with pytest.raises(StopAsyncIteration):
        await anext(gen)
