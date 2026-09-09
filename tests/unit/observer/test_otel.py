from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode

from ant_ai.observer.integrations.otel import OTelSink


def _tracer():
    return TracerProvider().get_tracer("test")


@pytest.mark.unit
async def test_span_yields_handle_with_update():
    """`llm_step`/`tool_step` call `update()`; a raw OTel span has no such method."""
    sink = OTelSink(_tracer())

    async with sink.span("llm", model="gpt-4o") as span:
        assert hasattr(span, "update")
        span.update(output="hello", metadata={"tool_call_count": 1})

    attrs = span.span.attributes
    assert attrs["model"] == "gpt-4o"
    assert attrs["output"] == "hello"


@pytest.mark.unit
async def test_update_skips_none_values():
    sink = OTelSink(_tracer())

    async with sink.span("llm") as span:
        span.update(output="hi", model=None)

    assert "model" not in span.span.attributes
    assert span.span.attributes["output"] == "hi"


@pytest.mark.unit
async def test_update_with_error_level_sets_span_status():
    sink = OTelSink(_tracer())

    async with sink.span("tool") as span:
        span.update(level="ERROR", status_message="tool blew up")

    assert span.span.status.status_code is StatusCode.ERROR
    assert span.span.status.description == "tool blew up"


@pytest.mark.unit
async def test_update_never_raises_into_the_run():
    """A telemetry sink that throws would turn a traced run into a failed one."""

    class _Exploding:
        def set_attribute(self, *_: object) -> None:
            raise RuntimeError("exporter down")

    from ant_ai.observer.integrations.otel import _OTelSpan

    _OTelSpan(_Exploding()).update(output="x")  # must not raise


@pytest.mark.unit
async def test_span_records_exception_and_reraises():
    sink = OTelSink(_tracer())

    with pytest.raises(ValueError):
        async with sink.span("llm") as span:
            raise ValueError("boom")

    assert span.span.status.status_code is StatusCode.ERROR
