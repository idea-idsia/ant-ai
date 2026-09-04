"""DIG's repair loop over a real Colony, real Agents and real ReAct loops.

The unit tests drive detectors against a hand-built `InteractionGraph`, and the
example drives them through scripted stand-ins. Neither answers the question this
file exists for: does a *model-driven* run produce a record with the symptom in
it, and does repairing that record change what the run does?

`scripted_llm` patches the LiteLLM transport, so the agent loop, structured
output and the ensemble are all the real thing and no network is touched. What
the script fixes is the model's choices, not the framework's.
"""

from __future__ import annotations

import json

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from conftest import build_single_node_workflow

from ant_ai.a2a.colony import Colony
from ant_ai.agent.agent import Agent
from ant_ai.core.events import HealingEvent
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.topology.builtins import DigToHeal
from ant_ai.topology.materialise import DeliveryMaterialiser
from ant_ai.topology.schedule import BufferScheduler
from ant_ai.topology.strategy import Pipeline, TopologyStrategy

pytestmark = [pytest.mark.integration, pytest.mark.multi_agent, pytest.mark.topology]

COORDINATOR_SYS = "You are the coordinator"
WORKER_SYS = "You are the worker"
AUDITOR_SYS = "You are the auditor"

TASK = "Count how often each integer occurs in the array."
PREMATURE = "Final answer: counted, based on the worker's partial."
COMPLETE = "Final answer: counted, with the auditor's discrepancy resolved."

# One turn per round per agent, as a `TurnPayload`. The coordinator calls the
# task done in round 1 while the auditor's opening message is still sitting
# unread — the Early Termination the paper's detector exists to catch.
SCRIPT: dict[str, list[dict]] = {
    "coordinator": [
        {
            "message": "Splitting the array.",
            "messages": [{"to": ["worker"], "content": "Count elements 0-4999."}],
        },
        {"message": PREMATURE, "submitted": True},
        {"message": COMPLETE, "submitted": True},
    ],
    "worker": [
        {
            "message": "Counting my slice.",
            "messages": [
                {"to": ["coordinator"], "content": "Partial: 0 appears 250x."}
            ],
        },
        {"message": "Nothing further from me."},
    ],
    "auditor": [
        # Addressed to nobody: the auditor does not yet know who is coordinating.
        # With no routing stage under it, that is a message nothing is routed to.
        {"message": "I see a discrepancy in the slice boundaries."},
        {"message": "Still waiting to be told who owns the boundaries."},
    ],
}


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
            ("coordinator", COORDINATOR_SYS),
            ("worker", WORKER_SYS),
            ("auditor", AUDITOR_SYS),
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
            card=_card(name, 9201 + i),
        )
    return colony


@pytest.fixture
def scripted_rounds(scripted_llm):
    """Answer each agent with its next scripted `TurnPayload`."""

    calls: dict[str, int] = {}

    async def dispatch(*, messages, **_):
        system = messages[0].get("content", "")
        name = (
            "coordinator"
            if COORDINATOR_SYS in system
            else "worker"
            if WORKER_SYS in system
            else "auditor"
        )
        index = calls.get(name, 0)
        calls[name] = index + 1
        turns = SCRIPT[name]
        return scripted_llm.make_text_response(
            json.dumps(turns[min(index, len(turns) - 1)])
        )

    scripted_llm.install(dispatch)
    return calls


class Unsupervised(TopologyStrategy):
    """DIG's timing and delivery with no detectors — the control condition.

    Isolates the repair loop: everything else about the run, including which
    agent activates when and how messages are routed, is identical.
    """

    def build(self) -> Pipeline:
        return Pipeline(
            scheduler=BufferScheduler(), materialiser=DeliveryMaterialiser()
        )


async def test_an_unsupervised_run_ends_on_the_premature_answer(
    colony, scripted_rounds
) -> None:
    """The failure, first: the coordinator submits while the auditor's message
    has reached nobody, and nothing stops the run ending there."""
    colony.topology(Unsupervised(max_rounds=4))

    ensemble = colony.ensemble(use_workflows=False)
    answer = await ensemble.ainvoke(TASK)

    assert PREMATURE in answer
    assert ensemble.findings == []
    # Two rounds: the split, then the submit that ended it.
    assert scripted_rounds["coordinator"] == 2


async def test_early_termination_is_detected_in_a_model_driven_run(
    colony, scripted_rounds
) -> None:
    """The same script under DIG: the submit is caught, handed back, and the run
    keeps going long enough to produce a different answer."""
    colony.topology(DigToHeal(max_rounds=4))

    ensemble = colony.ensemble()
    events = [e async for e in ensemble.stream(TASK)]

    healings = [e for e in events if isinstance(e, HealingEvent)]
    assert "ET" in {e.pattern for e in healings}, [e.pattern for e in healings]
    assert "reroute" in {i for e in healings for i in e.interventions}
    assert scripted_rounds["coordinator"] > 2, "the premature submit ended the run"


async def test_the_auditors_unrouted_message_is_found(colony, scripted_rounds) -> None:
    """The other half of the same record: with no routing stage under it, a
    message addressed to nobody reaches nobody, and only the graph knows."""
    colony.topology(DigToHeal(max_rounds=4))

    ensemble = colony.ensemble()
    await ensemble.ainvoke(TASK)

    patterns = {f.pattern for f in ensemble.findings}
    assert "OE" in patterns, patterns
    assert any("auditor" in f.explanation for f in ensemble.findings)


async def test_dig_invokes_agents_directly_by_default(colony) -> None:
    """The default that makes the two tests above possible. Under a workflow the
    same script would yield one plain public message per turn: nothing addressed,
    nothing submitted, and every detector reporting a healthy run."""
    colony.topology(DigToHeal())

    assert all(p.workflow is None for p in colony.ensemble().participants.values())
