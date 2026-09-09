from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from ant_ai.observer.composite import CompositeSink


def _make_mock_sink():
    sink = MagicMock()
    sink.event = AsyncMock(return_value=None)
    sink.exception = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _noop_span(name: str, **attrs):
        yield

    sink.span = _noop_span
    return sink


@pytest.mark.unit
async def test_event_fans_out_to_all_sinks():
    s1, s2 = _make_mock_sink(), _make_mock_sink()
    composite = CompositeSink([s1, s2])

    await composite.event("test.event", x=1)

    s1.event.assert_called_once_with("test.event", x=1)
    s2.event.assert_called_once_with("test.event", x=1)


@pytest.mark.unit
async def test_event_continues_if_one_sink_raises():
    s1 = _make_mock_sink()
    s1.event = AsyncMock(side_effect=RuntimeError("sink 1 broken"))
    s2 = _make_mock_sink()
    composite = CompositeSink([s1, s2])

    # Must not raise
    await composite.event("test.event")

    s2.event.assert_called_once()


@pytest.mark.unit
async def test_exception_fans_out_to_all_sinks():
    s1, s2 = _make_mock_sink(), _make_mock_sink()
    composite = CompositeSink([s1, s2])
    err = ValueError("test error")

    await composite.exception("test.error", err, k="v")

    s1.exception.assert_called_once_with("test.error", err, k="v")
    s2.exception.assert_called_once_with("test.error", err, k="v")


@pytest.mark.unit
async def test_exception_continues_if_one_sink_raises():
    s1 = _make_mock_sink()
    s1.exception = AsyncMock(side_effect=RuntimeError("sink 1 broken"))
    s2 = _make_mock_sink()
    composite = CompositeSink([s1, s2])

    # Must not raise
    await composite.exception("test.error", ValueError("oops"))

    s2.exception.assert_called_once()


@pytest.mark.unit
async def test_span_enters_all_sink_spans():
    entered: list[str] = []

    def _tracking_sink(label: str):
        sink = MagicMock()
        sink.event = AsyncMock(return_value=None)
        sink.exception = AsyncMock(return_value=None)

        @asynccontextmanager
        async def _span(name: str, **attrs):
            entered.append(label)
            yield

        sink.span = _span
        return sink

    s1 = _tracking_sink("s1")
    s2 = _tracking_sink("s2")
    composite = CompositeSink([s1, s2])

    async with composite.span("test.span"):
        pass

    assert "s1" in entered
    assert "s2" in entered


@pytest.mark.unit
async def test_span_with_empty_sinks_list_does_not_raise():
    composite = CompositeSink([])
    async with composite.span("test.span"):
        pass


@pytest.mark.unit
async def test_span_yields_handle_whose_update_reaches_every_sink():
    """The yielded object must carry `update()` to all sinks.

    `llm_step` and `tool_step` call `span.update(...)` on whatever `obs.span`
    yields, so a composite that yields nothing breaks both.
    """
    updates: list[tuple[str, dict]] = []

    def _updating_sink(label: str):
        sink = MagicMock()
        sink.event = AsyncMock(return_value=None)
        sink.exception = AsyncMock(return_value=None)

        span_obj = MagicMock()
        span_obj.update = lambda **f: updates.append((label, f))

        @asynccontextmanager
        async def _span(name: str, **attrs):
            yield span_obj

        sink.span = _span
        return sink

    composite = CompositeSink([_updating_sink("s1"), _updating_sink("s2")])

    async with composite.span("llm") as span:
        span.update(output="hello", metadata={"k": 1})

    assert updates == [
        ("s1", {"output": "hello", "metadata": {"k": 1}}),
        ("s2", {"output": "hello", "metadata": {"k": 1}}),
    ]


@pytest.mark.unit
async def test_span_update_continues_if_one_sink_raises():
    seen: list[str] = []

    def _sink(label: str, *, broken: bool):
        sink = MagicMock()
        sink.event = AsyncMock(return_value=None)
        sink.exception = AsyncMock(return_value=None)

        span_obj = MagicMock()
        if broken:
            span_obj.update = MagicMock(side_effect=RuntimeError("span broken"))
        else:
            span_obj.update = lambda **f: seen.append(label)

        @asynccontextmanager
        async def _span(name: str, **attrs):
            yield span_obj

        sink.span = _span
        return sink

    composite = CompositeSink([_sink("s1", broken=True), _sink("s2", broken=False)])

    async with composite.span("llm") as span:
        span.update(output="x")  # must not raise

    assert seen == ["s2"]


@pytest.mark.unit
async def test_span_with_empty_sinks_still_yields_an_updatable_handle():
    composite = CompositeSink([])
    async with composite.span("llm") as span:
        span.update(output="x")  # must not raise
