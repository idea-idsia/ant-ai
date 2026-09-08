"""Adaptive multi-agent topology, and the rest of what a self-evolving run edits.

Most use needs nothing from this module. Declare a strategy on a colony by name
and run it:

    colony.evolve("dytopo|dig")
    async for event in colony.ensemble().stream(task):
        ...

What is exported here is the rest of that path: the runtime, the record it
produces, and the two configuration knobs — `Halt` and `Heal` — that a strategy
is most often adjusted with.

**Writing a strategy** means implementing one of the seams, which live in their
own modules so that a seam is distinguishable from a use of one:

| Seam | Module |
| --- | --- |
| `Stage` | `ant_ai.topology.plan` |
| `Detector` | `ant_ai.topology.heal` |
| `Scheduler` | `ant_ai.topology.schedule` |
| `TopologyMaterialiser` | `ant_ai.topology.materialise` |
| `HaltPolicy` | `ant_ai.topology.strategy` |
| `Participant` | `ant_ai.topology.participant` |

The published methods built on them are in `ant_ai.topology.builtins`.

**What a stage may change** is a typed `Rewrite` (`ant_ai.topology.rewrite`) —
the operator set from arXiv:2608.18104: insert, delete, feature_update and merge
on nodes; link, unlink, rewire and edge_feature_update on edges; activate for
read-only selection. Reachability is the communication-edge case of that
vocabulary, which is why `with_links` still exists and still means what it did;
everything else a method might evolve — a memory, a tool, a skill, a workflow —
is the same operators applied to a different node kind, and lands in the same
`RewriteLog`.

| What | Module |
| --- | --- |
| The operators | `ant_ai.topology.rewrite` |
| The persistent graph they apply to | `ant_ai.topology.state` |
| The trail, and rollback | `ant_ai.topology.log` |
| Recorded read-only selection | `ant_ai.topology.activate` |
| A stage on a slower clock | `ant_ai.topology.cadence` |
| Graph-aware metrics | `ant_ai.topology.evaluate` |
"""

from ant_ai.topology.activate import RecordingMemory, record_support
from ant_ai.topology.cadence import Every
from ant_ai.topology.evaluate import leaked, locality, support_accuracy
from ant_ai.topology.graph import InteractionGraph, Link
from ant_ai.topology.heal import Heal
from ant_ai.topology.log import LogEntry, RewriteLog
from ant_ai.topology.participant import LocalParticipant
from ant_ai.topology.problem import Problem, TopologyConfigurationError
from ant_ai.topology.rewrite import EdgeRef, NodeRef, Rewrite
from ant_ai.topology.runtime import Ensemble, RunReport
from ant_ai.topology.state import (
    Schema,
    SchemaViolation,
    StateGraph,
    SupportSubgraph,
)
from ant_ai.topology.strategy import (
    EvolutionStrategy,
    Halt,
    Pipeline,
)

__all__ = [
    # run one
    "Ensemble",
    "RunReport",
    "EvolutionStrategy",
    "EvolutionStrategy",
    "Pipeline",
    # inspect one
    "InteractionGraph",
    "Link",
    "StateGraph",
    "SupportSubgraph",
    "RewriteLog",
    "LogEntry",
    # edit one
    "Rewrite",
    "NodeRef",
    "EdgeRef",
    # configure one
    "Halt",
    "Heal",
    "Every",
    "LocalParticipant",
    "RecordingMemory",
    "record_support",
    "Schema",
    # measure one
    "support_accuracy",
    "leaked",
    "locality",
    # when it cannot run
    "TopologyConfigurationError",
    "SchemaViolation",
    "Problem",
]
