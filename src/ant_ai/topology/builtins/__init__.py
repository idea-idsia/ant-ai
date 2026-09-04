"""Shipped topology strategies — one module per published method.

The convention for adding one: a new module here, containing whatever `Stage`s
and `Detector`s the paper introduces plus a `TopologyStrategy` subclass that
assembles them in `build()`. Nothing in the core needs to change, and the strategy
registers itself by `name` for config-driven ablations.

Kept out of `ant_ai.topology` so that the interface and the implementations of it
are distinguishable at a glance: everything in the parent package is a seam,
everything here is a use of one.
"""

from ant_ai.topology.builtins.dig import (
    CrossLineageAggregation,
    Deadlock,
    DigToHeal,
    EarlyTermination,
    ExcessiveRerouting,
    JudgeHealing,
    LLMJudge,
    MissingCompletion,
    OrphanedEvent,
    RepeatedSubproblem,
    dig_detectors,
)
from ant_ai.topology.builtins.dytopo import (
    DyTopo,
    RandomScores,
    RandomTopology,
    Semantic,
    TopK,
)
from ant_ai.topology.builtins.shapes import Baseline, Static, chain, mesh, star

__all__ = [
    # shapes
    "Baseline",
    "Static",
    "chain",
    "star",
    "mesh",
    # dytopo (arXiv:2602.06039)
    "DyTopo",
    "RandomTopology",
    "Semantic",
    "TopK",
    "RandomScores",
    # dig to heal (arXiv:2603.00309)
    "DigToHeal",
    "JudgeHealing",
    "dig_detectors",
    "EarlyTermination",
    "MissingCompletion",
    "OrphanedEvent",
    "Deadlock",
    "ExcessiveRerouting",
    "CrossLineageAggregation",
    "RepeatedSubproblem",
    "LLMJudge",
]
