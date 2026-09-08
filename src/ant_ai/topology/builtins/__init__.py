"""Shipped topology strategies — one module per published method.

The convention for adding one: a new module here, containing whatever `Stage`s
and `Detector`s the paper introduces plus a `EvolutionStrategy` subclass that
assembles them in `build()`. Nothing in the core needs to change, and the strategy
registers itself by `name` for config-driven use — `colony.evolve("dytopo")`.

Kept out of `ant_ai.topology` so that the interface and the implementations of it
are distinguishable at a glance: everything in the parent package is a seam,
everything here is a use of one.

Exported here are the strategies themselves. Their parts — the individual
detectors, the scoring and sparsifying stages — are importable from the module
that defines them (`ant_ai.topology.builtins.dig`,
`ant_ai.topology.builtins.dytopo`) and are what you reach for to assemble a
variant rather than to run one.
"""

from ant_ai.topology.builtins.dig import DigToHeal, JudgeHealing, dig_detectors
from ant_ai.topology.builtins.dytopo import DyTopo, RandomTopology
from ant_ai.topology.builtins.shapes import Baseline, Static, chain, mesh, star

__all__ = [
    # published methods
    "DyTopo",  # arXiv:2602.06039
    "DigToHeal",  # arXiv:2603.00309
    # their controls
    "RandomTopology",
    "JudgeHealing",
    "Baseline",
    # fixed shapes
    "Static",
    "chain",
    "star",
    "mesh",
    # assembling a variant
    "dig_detectors",
]
