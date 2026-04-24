from __future__ import annotations

import pytest

from ant_ai.a2a.client import A2AClient
from ant_ai.a2a.config import A2AConfig
from ant_ai.core.events import FinalAnswerEvent

RESPONDER_SYS = "You are a responder"


@pytest.mark.integration
@pytest.mark.multi_agent
async def test_responder_returns_final_answer(two_agent_hive, scripted_llm):
    """Responder receives a message and emits a FinalAnswerEvent with no A2A hop."""

    async def dispatch(*, messages, **_):
        return scripted_llm.make_text_response("The answer is 42.")

    scripted_llm.install(dispatch)

    port = two_agent_hive["ports"]["responder"]
    client = A2AClient(config=A2AConfig(endpoint=f"http://127.0.0.1:{port}/"))
    events = [ev async for ev in client.send_message("What is the answer?")]
    await client.aclose()

    final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
    assert final_events, "No FinalAnswerEvent received from responder"
    assert "42" in final_events[-1].content


@pytest.mark.integration
@pytest.mark.multi_agent
async def test_caller_answers_without_delegation(two_agent_hive, scripted_llm):
    """Caller returns a direct answer (no tool call); responder is never reached."""
    responder_called = False

    async def dispatch(*, messages, **_):
        nonlocal responder_called
        sys = messages[0].get("content", "")
        if RESPONDER_SYS in sys:
            responder_called = True
        return scripted_llm.make_text_response("I know this one: 2+2=4.")

    scripted_llm.install(dispatch)

    port = two_agent_hive["ports"]["caller"]
    client = A2AClient(config=A2AConfig(endpoint=f"http://127.0.0.1:{port}/"))
    events = [ev async for ev in client.send_message("What is 2+2?")]
    await client.aclose()

    assert not responder_called, "Responder was unexpectedly called"
    final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
    assert final_events, "No FinalAnswerEvent received from caller"
    assert "4" in final_events[-1].content
