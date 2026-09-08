from __future__ import annotations

import warnings

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from fakes import FakeEmbedder

from ant_ai.a2a.colony import Colony
from ant_ai.agent.agent import Agent
from ant_ai.topology.builtins.dig import DigToHeal
from ant_ai.topology.builtins.dytopo import DyTopo
from ant_ai.topology.graph import Link
from ant_ai.topology.materialise import DeliveryMaterialiser, VisibilityMaterialiser
from ant_ai.topology.participant import A2AParticipant, LocalParticipant
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
    for i, name in enumerate(("codegen", "testgen")):
        colony.agent(
            name,
            agent=Agent(name=name, system_prompt="work", llm=stub_llm),
            workflow=_runnable_workflow(),
            card=_card(name, 9001 + i),
        )
    return colony.collab("codegen", "testgen")


def test_a_colony_without_a_declared_topology_behaves_as_before(colony: Colony) -> None:
    ensemble = colony.ensemble()

    # No stage writes links, so the declared edges govern every round — which is
    # exactly the pre-topology behaviour.
    assert ensemble.pipeline.stages == []
    assert isinstance(ensemble.pipeline.materialiser, VisibilityMaterialiser)
    # collab(codegen -> testgen) means testgen offers, so the link reverses.
    assert ensemble.seed == (
        Link(src="testgen", dst="codegen", reason="declared via Colony.collab()"),
    )


def test_local_participants_are_built_from_the_specs(colony: Colony) -> None:
    ensemble = colony.ensemble()

    assert set(ensemble.participants) == {"codegen", "testgen"}
    assert all(isinstance(p, LocalParticipant) for p in ensemble.participants.values())
    assert ensemble.participants["codegen"].profile.description == "A base agent."


def test_remote_participants_are_built_when_local_is_false(colony: Colony) -> None:
    ensemble = colony.ensemble(local=False)

    assert all(isinstance(p, A2AParticipant) for p in ensemble.participants.values())
    assert ensemble.participants["testgen"].profile.description.startswith(
        "testgen agent"
    )


def test_a_strategy_supplies_stages_materialiser_and_provenance(colony: Colony) -> None:
    colony.evolve(DyTopo(embedder=FakeEmbedder({}), tau=0.4, max_rounds=4))

    ensemble = colony.ensemble()

    assert [type(s).__name__ for s in ensemble.pipeline.stages] == ["Semantic", "TopK"]
    assert isinstance(ensemble.pipeline.materialiser, DeliveryMaterialiser)
    assert ensemble.pipeline.max_rounds == 4
    assert ensemble.provenance["strategy"] == "dytopo"
    assert ensemble.provenance["tau"] == 0.4


def test_a_composed_strategy_reaches_the_colony(colony: Colony) -> None:
    """The payoff of `|` at the level a user writes: routing and repair layered
    without either strategy knowing about the other."""
    colony.evolve(DyTopo(embedder=FakeEmbedder({})) | DigToHeal())

    ensemble = colony.ensemble()

    assert [type(s).__name__ for s in ensemble.pipeline.stages] == [
        "Semantic",
        "TopK",
        "Heal",
    ]
    assert ensemble.provenance["strategy"] == "dytopo|dig"


def test_an_explicit_materialiser_overrides_the_strategy(colony: Colony) -> None:
    colony.evolve(DyTopo(embedder=FakeEmbedder({})))

    ensemble = colony.ensemble(materialiser=VisibilityMaterialiser())

    assert isinstance(ensemble.pipeline.materialiser, VisibilityMaterialiser)


def test_ad_hoc_detectors_are_appended_as_one_heal_stage(colony: Colony) -> None:
    from ant_ai.topology.builtins.dig import EarlyTermination

    colony.evolve(DyTopo(embedder=FakeEmbedder({})), detectors=[EarlyTermination()])

    stages = colony.ensemble().pipeline.stages

    assert [type(s).__name__ for s in stages] == ["Semantic", "TopK", "Heal"]
    assert [d.pattern for d in stages[-1].detectors] == ["ET"]


def test_collab_and_asgi_are_untouched_by_the_topology_layer(colony: Colony) -> None:
    assert colony._edges == {
        "codegen": {"testgen": colony._edges["codegen"]["testgen"]}
    }
    assert colony.get_agent_host("codegen") == ("codegen", 9001)


def test_use_workflows_is_off_by_default_and_explicit_when_on(colony: Colony) -> None:
    """`Workflow.stream` takes no response schema, so a workflow-driven
    participant cannot emit query/key descriptors, declare reactions or submit.

    That used to be inferred from the pipeline. It is now simply the default,
    because a flag whose value depends on what is in the stage list is a flag
    nobody can predict the meaning of."""
    assert colony.ensemble().participants["codegen"].workflow is None
    assert (
        colony.ensemble(use_workflows=True).participants["codegen"].workflow is not None
    )


def test_a_descriptor_driven_strategy_invokes_agents_directly(colony: Colony) -> None:
    """A workflow-driven turn emits no query/key, so a semantic matcher would
    score unchanging card text and the topology would be a fixed baseline
    wearing an adaptive label."""
    colony.evolve(DyTopo(embedder=FakeEmbedder({})))

    assert colony.ensemble().participants["codegen"].workflow is None


def test_a_repair_strategy_also_invokes_agents_directly(colony: Colony) -> None:
    """Not only matchers. Every symptom DIG looks for — a submit, a message left
    waiting, a reroute — is declared in a structured turn."""
    colony.evolve(DigToHeal())

    assert colony.ensemble().participants["codegen"].workflow is None


def test_ad_hoc_detectors_are_checked_like_any_other_stage(colony: Colony) -> None:
    """The `Heal` stage `Colony.evolve(detectors=...)` appends is a stage like
    any other, so it needs structured turns like any other."""
    from ant_ai.topology.builtins.dig import EarlyTermination
    from ant_ai.topology.builtins.shapes import Baseline

    colony.evolve(Baseline(), detectors=[EarlyTermination()])

    with pytest.raises(TopologyConfigurationError, match="E001"):
        colony.ensemble(use_workflows=True)


def test_a_strategy_that_reads_nothing_from_a_turn_may_run_the_workflow(
    colony: Colony,
) -> None:
    """Faithful to how a colony serves a request, and allowed wherever nothing
    depends on what a turn declares."""
    from ant_ai.topology.builtins.shapes import Baseline

    colony.evolve(Baseline())

    assert (
        colony.ensemble(use_workflows=True).participants["codegen"].workflow is not None
    )


def test_forcing_workflows_under_a_structured_strategy_is_rejected(
    colony: Colony,
) -> None:
    """Previously honoured, and the run then produced a matcher scoring static
    text or detectors reading defaults — a plausible result with nothing behind
    it. Stating the combination now names what would have gone wrong."""
    colony.evolve(DyTopo(embedder=FakeEmbedder({})))

    with pytest.raises(TopologyConfigurationError, match="E001") as excinfo:
        colony.ensemble(use_workflows=True)

    assert "Semantic" in str(excinfo.value)
    assert [p.code for p in excinfo.value.problems] == ["E001"]


def test_remote_participants_under_visibility_are_rejected(colony: Colony) -> None:
    """There is no A2A operation for attaching a tool to an agent in another
    process, so a topology materialised as peer tools constrains nothing. It was
    a warning; a topology that provably does nothing should not run."""
    colony.evolve(DyTopo(embedder=FakeEmbedder({})))

    with pytest.raises(TopologyConfigurationError, match="E003"):
        colony.ensemble(local=False, materialiser=VisibilityMaterialiser())


def test_remote_participants_cannot_carry_a_structured_strategy(
    colony: Colony,
) -> None:
    """A2A carries no response schema, so a remote run is unstructured however
    the colony is configured — which `use_workflows` never controlled."""
    colony.evolve(DyTopo(embedder=FakeEmbedder({})))

    with pytest.raises(TopologyConfigurationError, match="E001") as excinfo:
        colony.ensemble(local=False)

    assert "remote (A2A)" in str(excinfo.value)


def test_a_remote_run_with_a_shape_strategy_is_allowed(colony: Colony) -> None:
    """Delivery routes messages, which works over the wire, and a fixed shape
    reads nothing a structured turn carries.

    It still warns that nothing can submit: a remote run has no structured turn
    to declare one in, so the round budget is what ends it. Allowed, and said
    out loud rather than discovered from the bill."""
    from ant_ai.topology.builtins.dytopo import RandomTopology

    colony.evolve(RandomTopology())

    with pytest.warns(RuntimeWarning, match="W001"):
        ensemble = colony.ensemble(local=False)

    assert len(ensemble.participants) == 2


def test_a_colony_with_no_strategy_is_not_warned_about(colony: Colony) -> None:
    """Nothing decides a topology, so remote agents stay wired as their servers
    wired them — the pre-topology behaviour, not a silent failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        colony.ensemble(local=False)
