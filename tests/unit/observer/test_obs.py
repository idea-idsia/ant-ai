from __future__ import annotations

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
