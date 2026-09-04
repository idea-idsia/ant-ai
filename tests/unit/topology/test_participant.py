from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fakes import FakeParticipant

from ant_ai.agent.agent import Agent
from ant_ai.core.events import Event, FinalAnswerEvent, ToolCallingEvent
from ant_ai.core.message import ToolCall, ToolFunction
from ant_ai.topology.participant import (
    Brief,
    Envelope,
    LocalParticipant,
    PeerTool,
    Turn,
)

pytestmark = [pytest.mark.unit, pytest.mark.topology]


class FakeAgent:
    """Duck-typed `BaseAgent`: `LocalParticipant` only needs these members."""

    def __init__(self, name: str, script: list[Event]) -> None:
        self.name = name
        self.description = f"{name} agent"
        self.script = script
        self.tools: list[object] = []

    def add_tool(self, tool) -> None:
        self.tools.append(tool)

    def remove_tool(self, name: str) -> None:
        self.tools = [t for t in self.tools if t.name != name]

    async def stream(
        self, state, *, ctx=None, response_schema=None
    ) -> AsyncIterator[Event]:
        for event in self.script:
            yield event


async def _run(participant: LocalParticipant) -> Turn:
    turn: Turn | None = None
    async for item in participant.act(Brief(round=0, task="go")):
        if isinstance(item, Turn):
            turn = item
    assert turn is not None
    return turn


async def test_act_parses_a_single_pass_payload() -> None:
    """Descriptors arrive with the turn, not from a second LLM call."""
    payload = (
        '{"message": "here is the design", "private": "psst", '
        '"query": "need constraints", "key": "offers API design", "submitted": false}'
    )
    participant = LocalParticipant(
        FakeAgent("arch", [FinalAnswerEvent(content=payload)])
    )

    turn = await _run(participant)

    assert turn.public is not None and turn.public.content == "here is the design"
    assert turn.private is not None and turn.private.content == "psst"
    assert bool(turn.query or turn.key)
    assert turn.query == "need constraints"
    assert turn.submitted is False


async def test_act_degrades_when_the_payload_is_not_json() -> None:
    """A participant ignoring the schema should cost the matcher its
    descriptors, not abort the round."""
    participant = LocalParticipant(
        FakeAgent("arch", [FinalAnswerEvent(content="just prose")])
    )

    turn = await _run(participant)

    assert turn.public is not None and turn.public.content == "just prose"
    assert turn.query == "" and turn.key == ""


async def test_act_reports_which_peers_were_actually_called() -> None:
    call = ToolCall(id="1", function=ToolFunction(name="call_dev", arguments="{}"))
    agent = FakeAgent(
        "arch",
        [ToolCallingEvent(tool_calls=(call,)), FinalAnswerEvent(content="{}")],
    )
    participant = LocalParticipant(agent)
    await participant.bind_peers({"dev": FakeParticipant("dev")})

    turn = await _run(participant)

    assert turn.invoked == ("dev",)


async def test_bind_peers_attaches_and_detaches_on_a_real_agent(stub_llm) -> None:
    """Removal has to reach the loop's serialized tools, not just the registry."""
    agent = Agent(name="dev", system_prompt="you code", llm=stub_llm)
    participant = LocalParticipant(agent)

    await participant.bind_peers({"arch": FakeParticipant("arch")})
    assert "call_arch" in agent.registry
    assert any(
        t["function"]["name"] == "call_arch" for t in agent.registry.to_serialized()
    )
    assert [t.name for t in agent.tools] == ["call_arch"]

    await participant.bind_peers({})
    assert "call_arch" not in agent.registry
    assert agent.tools == []
    assert agent._loop.reason_step.serialized_tools == []


async def test_bind_peers_is_idempotent(stub_llm) -> None:
    agent = Agent(name="dev", system_prompt="you code", llm=stub_llm)
    participant = LocalParticipant(agent)
    peers = {"arch": FakeParticipant("arch")}

    await participant.bind_peers(peers)
    await participant.bind_peers(peers)

    assert [t.name for t in agent.tools] == ["call_arch"]


async def test_profile_comes_from_the_agent_description(stub_llm) -> None:
    agent = Agent(
        name="rev", system_prompt="review", llm=stub_llm, description="reviews code"
    )
    assert LocalParticipant(agent).profile.description == "reviews code"


async def test_peer_tool_calls_through_to_the_target() -> None:
    tool = PeerTool.for_participant(FakeParticipant("dev", message="done"))
    assert await tool.ainvoke(message="please implement") == "done"


async def test_peer_tool_refuses_beyond_max_depth() -> None:
    """Nested peer calls are bounded so 'one graph per round' stays true."""
    tool = PeerTool.for_participant(FakeParticipant("dev"), max_depth=1)

    inner = PeerTool.for_participant(FakeParticipant("rev"), max_depth=1)

    class Nested(FakeParticipant):
        async def act(self, brief, *, ctx=None):
            from ant_ai.topology import participant as mod

            assert mod._peer_depth.get() == 1
            yield Turn(participant=self.name)

    tool._target = Nested("dev")
    await tool.ainvoke(message="go")

    # At depth 1 with max_depth 1, a further hop is refused rather than recursing.
    from ant_ai.topology import participant as mod

    token = mod._peer_depth.set(1)
    try:
        assert "maximum call depth" in await inner.ainvoke(message="go")
    finally:
        mod._peer_depth.reset(token)


async def test_act_parses_addressing_and_reactions() -> None:
    """The paper's model in one forward pass: whom each message is for, and what
    the turn did with each message it was handed."""
    payload = (
        '{"message": "split five ways", '
        '"messages": [{"to": ["dev"], "content": "chunk 1"}, '
        '{"to": ["ops"], "content": "chunk 2"}], '
        '"reactions": {"e1": "wait"}, "reroute": {"e2": ["dev"]}}'
    )
    inbox = (
        Envelope(sender="ops", content="not ready"),
        Envelope(sender="qa", content="for someone else"),
    )
    participant = LocalParticipant(
        FakeAgent("arch", [FinalAnswerEvent(content=payload)])
    )

    turn: Turn | None = None
    async for item in participant.act(Brief(round=0, task="go", inbox=inbox)):
        if isinstance(item, Turn):
            turn = item
    assert turn is not None

    addressed = {e.content: e.recipients for e in turn.outputs if e.recipients}
    assert addressed == {"chunk 1": ("dev",), "chunk 2": ("ops",)}
    assert turn.public is not None and turn.public.content == "split five ways"
    # Tags are per-brief handles; the record keeps ids.
    assert turn.reactions[inbox[0].id] == "wait"
    assert turn.reactions[inbox[1].id] == "reroute"
    assert turn.rerouted[inbox[1].id] == ("dev",)
    assert turn.reaction_for("never delivered") == "consume"


async def test_a_brief_tags_its_inbox_so_a_model_can_answer_about_it() -> None:
    inbox = (
        Envelope(sender="ops", content="one"),
        Envelope(sender="qa", content="two"),
    )
    brief = Brief(round=0, task="go", inbox=inbox)

    assert brief.tags == {inbox[0].id: "e1", inbox[1].id: "e2"}
    assert brief.resolve("e2") == inbox[1].id
    assert brief.resolve(inbox[0].id) == inbox[0].id
    assert brief.resolve("e9") is None
    assert "[e1] ops: one" in brief.as_prompt()
