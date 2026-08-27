from __future__ import annotations

from types import SimpleNamespace

import pytest

from ant_ai.a2a.client import A2AClient
from ant_ai.a2a.config import A2AConfig
from ant_ai.core.events import ContentDeltaEvent, Event, FinalAnswerEvent


def _client(port: int) -> A2AClient:
    return A2AClient(config=A2AConfig(endpoint=f"http://127.0.0.1:{port}/"))


def _stream_chunk(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, reasoning_content=None)
            )
        ]
    )


def _install_streaming_dispatch(scripted_llm, parts: list[str]) -> None:
    async def dispatch(*, messages, stream=False, **_):
        if not stream:
            return scripted_llm.make_text_response("".join(parts))

        async def gen():
            for part in parts:
                yield _stream_chunk(part)

        return gen()

    scripted_llm.install(dispatch)


@pytest.mark.integration
@pytest.mark.a2a
async def test_streamed_deltas_concatenate_to_terminal_event_content(
    streaming_agent_hive, scripted_llm
) -> None:
    """With agent.streaming and A2A stream_artifacts both enabled, the client
    should receive ContentDeltaEvents whose concatenated text matches the
    terminal FinalAnswerEvent's content -- proving the whole path (LLMStep ->
    loop -> A2A artifact translation -> A2AToHVEvent reconstruction) is
    lossless end to end."""
    parts = ["Hel", "lo ", "world"]
    _install_streaming_dispatch(scripted_llm, parts)

    client = _client(streaming_agent_hive["port"])
    events: list[Event] = [
        ev async for ev in client.send_message("ping", context_id="ctx-stream-1")
    ]

    deltas = [e for e in events if isinstance(e, ContentDeltaEvent)]
    assert deltas, "expected at least one ContentDeltaEvent from the streamed run"
    assert "".join(d.delta for d in deltas) == "Hello world"

    final = next(e for e in events if isinstance(e, FinalAnswerEvent))
    assert final.content == "Hello world"


@pytest.mark.integration
@pytest.mark.a2a
async def test_non_streaming_client_unaffected_by_stream_artifacts(
    single_agent_hive, scripted_llm
) -> None:
    """A client talking to an agent/server with streaming disabled (the
    default) must see identical behavior to today: no ContentDeltaEvent, just
    the terminal whole event."""

    async def dispatch(*, messages, **_):
        return scripted_llm.make_text_response("plain response")

    scripted_llm.install(dispatch)

    client = _client(single_agent_hive["port"])
    events: list[Event] = [
        ev async for ev in client.send_message("ping", context_id="ctx-plain-1")
    ]

    assert not any(isinstance(e, ContentDeltaEvent) for e in events)
    final = next(e for e in events if isinstance(e, FinalAnswerEvent))
    assert final.content == "plain response"
