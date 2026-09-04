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
from ant_ai.workflow.workflow import Workflow

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


@pytest.fixture
def colony(stub_llm) -> Colony:
    colony = Colony()
    for i, name in enumerate(("codegen", "testgen")):
        colony.agent(
            name,
            agent=Agent(name=name, system_prompt="work", llm=stub_llm),
            workflow=Workflow(),
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
    colony.topology(DyTopo(embedder=FakeEmbedder({}), tau=0.4, max_rounds=4))

    ensemble = colony.ensemble()

    assert [type(s).__name__ for s in ensemble.pipeline.stages] == ["Semantic", "TopK"]
    assert isinstance(ensemble.pipeline.materialiser, DeliveryMaterialiser)
    assert ensemble.pipeline.max_rounds == 4
    assert ensemble.provenance["strategy"] == "dytopo"
    assert ensemble.provenance["tau"] == 0.4


def test_a_composed_strategy_reaches_the_colony(colony: Colony) -> None:
    """The payoff of `|` at the level a user writes: routing and repair layered
    without either strategy knowing about the other."""
    colony.topology(DyTopo(embedder=FakeEmbedder({})) | DigToHeal())

    ensemble = colony.ensemble()

    assert [type(s).__name__ for s in ensemble.pipeline.stages] == [
        "Semantic",
        "TopK",
        "Heal",
    ]
    assert ensemble.provenance["strategy"] == "dytopo|dig"


def test_an_explicit_materialiser_overrides_the_strategy(colony: Colony) -> None:
    colony.topology(DyTopo(embedder=FakeEmbedder({})))

    ensemble = colony.ensemble(materialiser=VisibilityMaterialiser())

    assert isinstance(ensemble.pipeline.materialiser, VisibilityMaterialiser)


def test_ad_hoc_detectors_are_appended_as_one_heal_stage(colony: Colony) -> None:
    from ant_ai.topology.builtins.dig import EarlyTermination

    colony.topology(DyTopo(embedder=FakeEmbedder({})), detectors=[EarlyTermination()])

    stages = colony.ensemble().pipeline.stages

    assert [type(s).__name__ for s in stages] == ["Semantic", "TopK", "Heal"]
    assert [d.pattern for d in stages[-1].detectors] == ["ET"]


def test_collab_and_asgi_are_untouched_by_the_topology_layer(colony: Colony) -> None:
    assert colony._edges == {
        "codegen": {"testgen": colony._edges["codegen"]["testgen"]}
    }
    assert colony.get_agent_host("codegen") == ("codegen", 9001)


def test_use_workflows_controls_descriptor_elicitation(colony: Colony) -> None:
    """`Workflow.stream` takes no response schema, so a workflow-driven
    participant cannot emit query/key descriptors. The flag makes that a
    choice rather than a silent fallback to AgentCard text."""
    assert colony.ensemble().participants["codegen"].workflow is not None
    assert colony.ensemble(use_workflows=False).participants["codegen"].workflow is None


def test_a_descriptor_driven_strategy_invokes_agents_directly_by_default(
    colony: Colony,
) -> None:
    """The default has to follow the strategy. A workflow-driven turn emits no
    query/key, so a semantic matcher would score unchanging card text and the
    topology would be static — `colony.ensemble()` would look adaptive and be a
    fixed baseline."""
    colony.topology(DyTopo(embedder=FakeEmbedder({})))

    assert colony.ensemble().participants["codegen"].workflow is None


def test_a_repair_strategy_also_invokes_agents_directly_by_default(
    colony: Colony,
) -> None:
    """Not only matchers. Every symptom DIG looks for — a submit, a message left
    waiting, a reroute — is declared in a structured turn, so under a workflow
    every detector would report a healthy run."""
    colony.topology(DigToHeal())

    assert colony.ensemble().participants["codegen"].workflow is None


def test_ad_hoc_detectors_also_decide_the_default(colony: Colony) -> None:
    """The `Heal` stage `Colony.topology(detectors=...)` appends is a stage like
    any other, so it settles the question the same way."""
    from ant_ai.topology.builtins.dig import EarlyTermination
    from ant_ai.topology.builtins.shapes import Baseline

    colony.topology(Baseline(), detectors=[EarlyTermination()])

    assert colony.ensemble().participants["codegen"].workflow is None


def test_a_strategy_that_reads_nothing_from_a_turn_still_runs_the_workflow(
    colony: Colony,
) -> None:
    """Faithful to how a colony serves a request, which is the right default
    wherever nothing depends on what a turn declares."""
    from ant_ai.topology.builtins.shapes import Baseline

    colony.topology(Baseline())

    assert colony.ensemble().participants["codegen"].workflow is not None


def test_an_explicit_choice_is_honoured_over_the_strategys_needs(
    colony: Colony,
) -> None:
    """Stating it yourself still wins — the fallback is reported at run time
    rather than forbidden here."""
    colony.topology(DyTopo(embedder=FakeEmbedder({})))

    assert (
        colony.ensemble(use_workflows=True).participants["codegen"].workflow is not None
    )


def test_remote_participants_under_visibility_are_warned_about(colony: Colony) -> None:
    """There is no A2A operation for attaching a tool to an agent in another
    process, so a topology materialised as peer tools constrains nothing."""
    colony.topology(DyTopo(embedder=FakeEmbedder({})))

    with pytest.warns(RuntimeWarning, match="cannot be rebound"):
        colony.ensemble(local=False, materialiser=VisibilityMaterialiser())


def test_remote_participants_under_delivery_are_not_warned_about(
    colony: Colony,
) -> None:
    """Delivery routes their messages, which works over the wire."""
    colony.topology(DyTopo(embedder=FakeEmbedder({})))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        colony.ensemble(local=False)


def test_a_colony_with_no_strategy_is_not_warned_about(colony: Colony) -> None:
    """Nothing decides a topology, so remote agents stay wired as their servers
    wired them — the pre-topology behaviour, not a silent failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        colony.ensemble(local=False)
