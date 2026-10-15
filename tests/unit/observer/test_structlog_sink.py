from __future__ import annotations

import pytest

from ant_ai.core.message import ToolCallMessage
from ant_ai.core.types import State
from ant_ai.observer import obs
from ant_ai.observer.integrations.log import StructlogSink
from ant_ai.steps.tool_step import ToolStep
from ant_ai.tools.registry import ToolRegistry
from ant_ai.tools.tool import tool


@tool
def ping(host: str) -> str:
    """Ping a host."""
    return f"pong {host}"


@pytest.fixture
def structlog_sink():
    previous = obs._sink
    obs.configure(StructlogSink())
    yield
    obs.configure(previous)


@pytest.mark.unit
async def test_span_yields_handle_with_update():
    """`llm_step`/`tool_step` call `update()` on whatever the span yields."""
    sink = StructlogSink()

    async with sink.span("tool", tool="ping") as span:
        assert hasattr(span, "update")
        span.update(output="pong", level="DEFAULT")


@pytest.mark.unit
async def test_tool_call_succeeds_with_structlog_sink(structlog_sink):
    msg = ToolCallMessage.model_validate(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "ping", "arguments": '{"host": "a"}'},
                }
            ],
        }
    )
    step = ToolStep(registry=ToolRegistry([ping]))

    events = [
        e
        async for e in step.run(State(messages=[msg]), None)
        if type(e).__name__ == "ToolResultEvent"
    ]

    assert len(events) == 1
    assert not events[0].is_error, events[0].content
    assert events[0].content == "pong a"
