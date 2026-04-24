from __future__ import annotations

import pytest

from ant_ai.a2a.client import A2AClient
from ant_ai.a2a.config import A2AConfig
from ant_ai.core.events import FinalAnswerEvent

CALLER_SYS = "You are a caller"
RESPONDER_SYS = "You are a responder"


@pytest.mark.integration
@pytest.mark.multi_agent
async def test_caller_delegates_to_responder(two_agent_hive, scripted_llm):
    """
    Caller receives a request, decides to call responder via A2AAgentTool,
    responder answers, caller produces a final answer.

    Verifies:
    - responder's server receives and processes the delegated request
    - caller makes exactly 2 LLM calls (tool-call + final answer)
    - caller's FinalAnswerEvent is emitted with content from the responder's reply
    """
    call_log: list[str] = []

    async def dispatch(*, messages, **_):
        sys = messages[0].get("content", "")
        if CALLER_SYS in sys:
            call_log.append("caller")
            if call_log.count("caller") == 1:
                # First call: delegate to responder
                return scripted_llm.make_tool_call_response(
                    "responder", "Please answer: what is the capital of France?"
                )
            # Second call: incorporate responder's answer into final reply
            return scripted_llm.make_text_response("The responder told me: Paris.")
        # Responder's call
        call_log.append("responder")
        return scripted_llm.make_text_response("Paris.")

    scripted_llm.install(dispatch)

    port = two_agent_hive["ports"]["caller"]
    client = A2AClient(config=A2AConfig(endpoint=f"http://127.0.0.1:{port}/"))
    events = [ev async for ev in client.send_message("What is the capital of France?")]
    await client.aclose()

    assert "caller" in call_log, "Caller LLM was never called"
    assert "responder" in call_log, (
        "Responder was never called — A2A hop did not happen"
    )
    assert call_log.count("caller") == 2, (
        f"Expected 2 caller LLM calls (tool-call + final), got {call_log}"
    )

    final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
    assert final_events, "No FinalAnswerEvent received from caller"
    assert "Paris" in final_events[-1].content


@pytest.mark.integration
@pytest.mark.multi_agent
async def test_responder_is_not_called_when_caller_answers_directly(
    two_agent_hive, scripted_llm
):
    """
    When the caller's LLM returns a plain-text answer (no tool call),
    the responder server must not receive any request.
    """
    responder_called = False

    async def dispatch(*, messages, **_):
        nonlocal responder_called
        sys = messages[0].get("content", "")
        if RESPONDER_SYS in sys:
            responder_called = True
            return scripted_llm.make_text_response("I should not be reached.")
        return scripted_llm.make_text_response("42.")

    scripted_llm.install(dispatch)

    port = two_agent_hive["ports"]["caller"]
    client = A2AClient(config=A2AConfig(endpoint=f"http://127.0.0.1:{port}/"))
    await client.send_message("What is 6 times 7?").__aiter__().__anext__()
    # consume fully
    async for _ in client.send_message("What is 6 times 7?"):
        pass
    await client.aclose()

    assert not responder_called, (
        "Responder was called even though no tool call was made"
    )


@pytest.mark.integration
@pytest.mark.multi_agent
async def test_delegation_result_appears_in_caller_context(
    two_agent_hive, scripted_llm
):
    """
    After the responder answers, the caller's second LLM call receives the tool
    result in its message history.  Verify via the final answer content.
    """
    responder_answer = "The speed of light is 299,792,458 m/s."

    async def dispatch(*, messages, **_):
        sys = messages[0].get("content", "")
        if CALLER_SYS in sys:
            # Check if tool result is already in the conversation
            history = " ".join(m.get("content") or "" for m in messages)
            if "299,792,458" in history:
                # Second call: caller sees the responder's result
                return scripted_llm.make_text_response(
                    f"As the responder explained: {responder_answer}"
                )
            return scripted_llm.make_tool_call_response(
                "responder", "What is the speed of light?"
            )
        return scripted_llm.make_text_response(responder_answer)

    scripted_llm.install(dispatch)

    port = two_agent_hive["ports"]["caller"]
    client = A2AClient(config=A2AConfig(endpoint=f"http://127.0.0.1:{port}/"))
    events = [ev async for ev in client.send_message("How fast is light?")]
    await client.aclose()

    final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
    assert final_events, "No FinalAnswerEvent received"
    assert "299,792,458" in final_events[-1].content, (
        "Caller's final answer did not include the responder's result"
    )
