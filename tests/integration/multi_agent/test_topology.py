"""End-to-end adaptive topology over a real Colony, real Agents and real ReAct loops.

Lives beside the other multi-agent integration tests so it can reuse
`scripted_llm`, which patches the LiteLLM transport: everything above it — the
agent loop, structured output, tool binding, the ensemble — is the real thing,
and no network is touched.
"""

from __future__ import annotations

import json

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from conftest import build_single_node_workflow

from ant_ai.a2a.colony import Colony
from ant_ai.agent.agent import Agent
from ant_ai.core.events import TopologyEvent
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.topology.builtins import DyTopo

pytestmark = [pytest.mark.integration, pytest.mark.multi_agent, pytest.mark.topology]

ARCHITECT_SYS = "You are the architect"
DEVELOPER_SYS = "You are the developer"
REVIEWER_SYS = "You are the reviewer"

# Round-by-round descriptors. The architect frames the work, then goes quiet;
# the developer and reviewer converge. Nothing prunes anyone explicitly — the
# scores simply stop matching.
SCRIPT: dict[str, list[tuple[str, str]]] = {
    "architect": [
        ("need the edge cases", "module design and API surface"),
        ("need nothing further", "nothing further to add"),
    ],
    "developer": [
        ("need the API surface", "implementation status, nothing yet"),
        ("need review of quoting", "a parser implementation"),
    ],
    "reviewer": [
        ("need code to review", "correctness critique"),
        ("need code to review", "sign-off on the quoting fix"),
    ],
}

VECTORS = {
    "module design and API surface": [1.0, 0.0, 0.0],
    "need the API surface": [1.0, 0.0, 0.0],
    "a parser implementation": [0.0, 1.0, 0.0],
    "need code to review": [0.0, 1.0, 0.0],
    "correctness critique": [0.0, 0.0, 1.0],
    "need review of quoting": [0.0, 0.0, 1.0],
    "sign-off on the quoting fix": [0.0, 0.0, 1.0],
}


class ScriptedEmbedder:
    """Fixed vectors, so the test states the similarity structure it wants."""

    model_id = "scripted"

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [VECTORS.get(t, [0.0, 0.0, 0.0]) for t in texts]


def _card(name: str, port: int) -> AgentCard:
    return AgentCard(
        name=name,
        description=f"The {name}.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=f"http://127.0.0.1:{port}/")
        ],
        skills=[],
    )


@pytest.fixture
def colony() -> Colony:
    colony = Colony()
    for i, (name, system) in enumerate(
        (
            ("architect", ARCHITECT_SYS),
            ("developer", DEVELOPER_SYS),
            ("reviewer", REVIEWER_SYS),
        )
    ):
        colony.agent(
            name,
            agent=Agent(
                name=name,
                system_prompt=system,
                description=f"The {name}.",
                llm=LiteLLMChat(model="gpt-4o-mini"),
            ),
            workflow=build_single_node_workflow(),
            card=_card(name, 9101 + i),
        )
    return colony


@pytest.fixture
def scripted_rounds(scripted_llm):
    """Answer every agent with a valid single-pass TurnPayload."""
    calls: dict[str, int] = {}

    async def dispatch(*, messages, **_):
        system = messages[0].get("content", "")
        name = (
            "architect"
            if ARCHITECT_SYS in system
            else "developer"
            if DEVELOPER_SYS in system
            else "reviewer"
        )
        index = calls.get(name, 0)
        calls[name] = index + 1
        query, key = SCRIPT[name][min(index, len(SCRIPT[name]) - 1)]
        return scripted_llm.make_text_response(
            json.dumps(
                {
                    "message": f"{name} round {index}",
                    "private": f"{name} detail {index}",
                    "query": query,
                    "key": key,
                    "submitted": False,
                }
            )
        )

    scripted_llm.install(dispatch)
    return calls


async def test_topology_changes_between_rounds(colony, scripted_rounds) -> None:
    """The feature working, stated as an assertion: the wiring is not the same
    twice. Under a static topology every round would be identical.

    Deliberately built with a bare `colony.ensemble()`. A workflow-driven turn
    carries no response schema, so its query/key come back empty and `Semantic`
    scores unchanging AgentCard text — the same links every round. This asserts
    the default resolves that for you: with the wrong default the run still
    completes and the wirings are simply all identical, which is a feature that
    silently is not there."""
    colony.topology(DyTopo(embedder=ScriptedEmbedder(), tau=0.5, k_in=2, max_rounds=3))

    ensemble = colony.ensemble()
    assert all(p.workflow is None for p in ensemble.participants.values())

    events = [e async for e in ensemble.stream("Build a CSV parser")]
    topologies = [e for e in events if isinstance(e, TopologyEvent)]

    assert len(topologies) >= 2
    wirings = [
        sorted((link.src, link.dst) for link in event.links) for event in topologies
    ]
    assert wirings[0] != wirings[-1], f"topology never adapted: {wirings}"


async def test_every_link_explains_itself(colony, scripted_rounds) -> None:
    colony.topology(DyTopo(embedder=ScriptedEmbedder(), tau=0.5, max_rounds=2))

    events = [
        e
        async for e in colony.ensemble(use_workflows=False).stream("Build a CSV parser")
    ]
    links = [link for e in events if isinstance(e, TopologyEvent) for link in e.links]

    assert links
    assert all(link.reason and "sim=" in link.reason for link in links)


async def test_peer_tools_match_in_neighbours(colony, scripted_rounds) -> None:
    """In visibility mode an agent's address book *is* the topology: after
    materialisation its attached peer tools are exactly its in-neighbours."""
    from ant_ai.topology.materialise import VisibilityMaterialiser

    colony.topology(DyTopo(embedder=ScriptedEmbedder(), tau=0.5, k_in=2, max_rounds=2))
    ensemble = colony.ensemble(
        use_workflows=False, materialiser=VisibilityMaterialiser()
    )

    events = [e async for e in ensemble.stream("Build a CSV parser")]
    last = [e for e in events if isinstance(e, TopologyEvent)][-1]
    assert last.links, "no reachability was granted, so the check would be vacuous"

    for name, participant in ensemble.participants.items():
        expected = {link.src for link in last.links if link.dst == name}
        attached = {
            tool.name.removeprefix("call_")
            for tool in participant.agent.tools
            if tool.name and tool.name.startswith("call_")
        }
        assert attached == expected, f"{name}: {attached} != {expected}"


async def test_the_run_records_which_method_produced_it(
    colony, scripted_rounds
) -> None:
    colony.topology(DyTopo(embedder=ScriptedEmbedder(), tau=0.5, max_rounds=2))
    ensemble = colony.ensemble(use_workflows=False)

    await ensemble.ainvoke("Build a CSV parser")

    assert ensemble.provenance["strategy"] == "dytopo"
    assert ensemble.provenance["embedder"] == "scripted"
    assert json.dumps(ensemble.provenance)
    assert json.loads(ensemble.graph.model_dump_json())["edges"]
