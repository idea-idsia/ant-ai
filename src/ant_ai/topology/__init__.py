"""Adaptive multi-agent topology: who can reach whom, decided per round.

The interface lives here; the strategies that use it live in
`ant_ai.topology.builtins`. That split is deliberate — a reader should be able to
tell a seam from a use of one without opening the file.

A strategy is an ordered list of `Stage`s, each transforming the plan for the next
round. Two published methods motivate that shape: DyTopo decides reachability
*before* a round from self-declared descriptors, DIG repairs collaboration *after*
one from the causal graph. Different inputs, same output — so they compose:

    from ant_ai.topology.builtins import DigToHeal, DyTopo

    strategy = DyTopo(embedder=embedder) | DigToHeal()
"""

from ant_ai.topology.graph import InteractionGraph, Link
from ant_ai.topology.heal import Detector, Heal
from ant_ai.topology.materialise import TopologyMaterialiser
from ant_ai.topology.participant import (
    Brief,
    Envelope,
    LocalParticipant,
    Participant,
    Turn,
)
from ant_ai.topology.plan import (
    Finding,
    Intervention,
    RoundPlan,
    RunContext,
    Stage,
)
from ant_ai.topology.runtime import Ensemble
from ant_ai.topology.schedule import Scheduler
from ant_ai.topology.strategy import Halt, HaltPolicy, Pipeline, TopologyStrategy

__all__ = [
    # run one
    "Ensemble",
    "TopologyStrategy",
    "Pipeline",
    # extend: the six seams
    "Stage",
    "Detector",
    "Scheduler",
    "TopologyMaterialiser",
    "HaltPolicy",
    "Participant",
    # the vocabulary a stage or detector is written against
    "RoundPlan",
    "RunContext",
    "InteractionGraph",
    "Link",
    "Turn",
    "Envelope",
    "Brief",
    "Finding",
    "Intervention",
    # configure
    "Halt",
    "Heal",
    "LocalParticipant",
]
