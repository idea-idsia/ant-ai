"""The configurations that used to run to completion and mean nothing.

Each test names the artefact the old behaviour produced, so that a change which
re-opens one of these paths fails with the reason rather than a diff.
"""

from __future__ import annotations

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentInterface

from ant_ai.a2a.colony import Colony
from ant_ai.agent.agent import Agent
from ant_ai.topology.materialise import VisibilityMaterialiser
from ant_ai.topology.problem import TopologyConfigurationError
from ant_ai.workflow.workflow import END, START, Workflow

pytestmark = [pytest.mark.unit, pytest.mark.topology, pytest.mark.a2a]


def _card(name: str, port: int) -> AgentCard:
    return AgentCard(
        name=name,
        description=f"{name} agent",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=f"http://{name}:{port}/")
        ],
        skills=[],
    )


def _runnable_workflow() -> Workflow:
    """START -> run -> END. A colony's workflows have to be able to run.

    A bare `Workflow()` has no edges and `Workflow.check()` rejects it, which is
    now its own preflight error (E004) rather than a run in which every
    activation fails.
    """

    async def _run(agent, state, ctx):
        yield state

    wf = Workflow()
    wf.add_node("run", _run)
    wf.add_edge(START, "run")
    wf.add_edge("run", END)
    return wf


@pytest.fixture
def colony(stub_llm) -> Colony:
    colony = Colony()
    for i, name in enumerate(("architect", "developer", "reviewer")):
        colony.agent(
            name,
            agent=Agent(name=name, system_prompt="work", llm=stub_llm),
            workflow=_runnable_workflow(),
            card=_card(name, 9001 + i),
        )
    return colony.collab("architect", "developer", mutual=True)


def test_workflow_driven_turns_under_dig_are_refused(colony: Colony) -> None:
    """Used to run: no reactions meant every delivery recorded as consumed, no
    submits meant Early Termination could never fire, and the three structural
    detectors fired on defaults instead — an orphaned-event storm on a
    collaboration nobody had observed."""
    colony.evolve("dig")

    with pytest.raises(TopologyConfigurationError, match="E001"):
        colony.ensemble(use_workflows=True)


def test_remote_participants_under_dytopo_are_refused(colony: Colony) -> None:
    """Used to run: A2A carries no response schema, so every participant fell
    back to static card text, the matrix was identical every round, and the
    topology never moved while reporting itself adaptive."""
    colony.evolve("dytopo")

    with pytest.raises(TopologyConfigurationError, match="E001"):
        colony.ensemble(local=False)


def test_remote_participants_under_visibility_are_refused(colony: Colony) -> None:
    """Used to warn only: a decided topology binding peer tools on agents in
    another process constrains nothing at all."""
    colony.evolve("dytopo")

    with pytest.raises(TopologyConfigurationError, match="E003"):
        colony.ensemble(local=False, materialiser=VisibilityMaterialiser())


def test_the_guided_path_builds_clean(colony: Colony) -> None:
    """The whole point: the common case is one string and no warnings."""
    colony.evolve("dytopo|dig")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ensemble = colony.ensemble()

    assert [type(s).__name__ for s in ensemble.pipeline.stages] == [
        "Semantic",
        "TopK",
        "Heal",
    ]
    assert ensemble.provenance["strategy"] == "dytopo|dig"


def test_a_repair_only_strategy_routes_over_declared_edges(colony: Colony) -> None:
    """`DigToHeal` writes no links. Its plan used to be empty, so unaddressed
    messages reached nobody; the declared `collab()` edges now stand."""
    colony.evolve("dig")

    ensemble = colony.ensemble()

    assert ensemble.pipeline.writes_links is False
    assert ensemble.seed, "the declared edges are the standing topology"
