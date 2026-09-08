"""Graph-aware measurement, for the questions an answer string cannot settle.

`RunReport` already says the layer's failure mode out loud: a run that completes
and looks fine. These are the same argument applied to what evolves — a memory
that retrieved the wrong evidence, a retrieval that saw the future, a local edit
that quietly moved something on the other side of the graph. None of it shows up
in the output, and all of it shows up here.

Three of the paper's five protocols are computable from what the layer already
records; the other two — their V-C deletion checking and V-E open challenges —
are `StateGraph.purge` and future work respectively.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ant_ai.topology.log import RewriteLog
from ant_ai.topology.state import StateGraph, SupportSubgraph

__all__ = ["leaked", "locality", "support_accuracy"]


def support_accuracy(state: StateGraph, gold: Mapping[str, Iterable[str]]) -> float:
    """Mean support-subgraph accuracy over the activations a gold set names (M1).

    `gold` maps a support id to the nodes an acceptable answer could have been
    grounded in. Activations with no gold entry are skipped rather than scored
    zero: an unlabelled retrieval is unmeasured, not wrong, and averaging it in
    as a failure is how a metric ends up reporting the labelling effort.

    Returns 1.0 when nothing is labelled, which is the honest reading of "no
    evidence of a problem" and the reason callers should check
    `len(state.supports)` alongside it.
    """
    scored = [s.dice(gold[s.id]) for s in state.supports if s.id in gold]
    return sum(scored) / len(scored) if scored else 1.0


def leaked(state: StateGraph, support: SupportSubgraph) -> tuple[str, ...]:
    """Nodes an activation used that did not exist yet when it ran (V-B).

    The temporal protocol as a query rather than as a discipline. A chronological
    split prevents leakage between train and test; this catches the other kind,
    where a component's own state graph hands a decision evidence written after
    the decision was made — which no split can prevent, because it happens
    inside one run.
    """
    return tuple(
        sorted(
            node_id
            for node_id in support.nodes
            if (node := state.nodes.get(node_id)) is not None
            and not node.alive_at(support.at)
        )
    )


def locality(log: RewriteLog, *, since: int = 0, probes: Iterable[str] = ()) -> float:
    """Fraction of probe components no edit since *since* could have reached (M2).

    Counterfactual locality, read structurally: an edit is local with respect to
    a probe when the probe is not in the edit's affected scope, so the probe's
    behaviour cannot have changed for a reason this edit is responsible for.

    That is weaker than the paper's behavioural probe — it bounds influence
    rather than measuring it, and a component inside the scope has not
    necessarily changed. It is also computable without re-running anything,
    which is what makes it usable as a regression check on a strategy that is
    supposed to make small edits and turns out not to.
    """
    watched = set(probes)
    if not watched:
        return 1.0
    reached: set[str] = set()
    for entry in log.since(since):
        if entry.applied:
            reached.update(entry.affected)
    return len(watched - reached) / len(watched)
