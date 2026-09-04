"""Projecting an `InteractionGraph` onto the paper's real-time figure.

The DIG page animates a run as three panels sharing one time axis: the
interaction graph, the same graph with the supervisor's markup stripped out,
and a Gantt of activation durations. This module turns the framework's own
record into the coordinates that figure needs, and nothing else — no plotting,
no run loop — so it works the same on a live `Ensemble` and on a graph reloaded
from `model_dump_json()`.

The mapping, in one place:

| Figure | Record |
| --- | --- |
| lane | participant name, with `supervisor` on top |
| x | seconds since the first activation started |
| circle | an `Activation`, coloured by DIG's problem-reducing test |
| square | an `Envelope` |
| grey edge | `generates` — the turn that produced the message |
| green edge | `delivers` with action `consume` |
| dashed edge | an `intervenes` edge: reroute, discard, inject |
| orange span | a round a participant did not activate in: DIG's *wait* |

The one liberty taken is that an activation still running has no end yet, so it
is drawn out to *now* and marked live. That is the whole point of watching it
happen rather than reading it afterwards.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime
from typing import Any

from ant_ai.topology.builtins.dig import is_problem_reducing
from ant_ai.topology.graph import Activation, InteractionGraph
from ant_ai.topology.participant import Envelope
from ant_ai.topology.plan import SUPERVISOR

__all__ = ["project"]

CLIP = 320
"""Characters of message content kept for the hover box."""

_REACTIONS = ("consume", "wait", "discard", "reroute")
"""Reactions a recipient can declare, drawn in the recipient's own terms."""


def project(
    graph: InteractionGraph,
    *,
    names: list[str],
    now: datetime | None = None,
    live: bool = True,
) -> dict[str, Any]:
    """The whole figure as plain data: lanes, nodes, edges, bars, stats.

    `live` says whether the run is still going, which is the only thing the
    projection cannot read off the record: a participant that has not activated
    in the round now running is pending, not waiting, and drawing it as waiting
    would report a stall that has not happened.
    """
    now = now or datetime.now(UTC)
    lanes = [*names, SUPERVISOR]
    lane_of = {name: i for i, name in enumerate(lanes)}

    starts = [a.started_at for a in graph.activations.values()]
    if not starts:
        return _empty(lanes)

    t0 = min(starts)

    def sec(moment: datetime) -> float:
        return (moment - t0).total_seconds()

    now_s = max(sec(now), 0.0)

    # -- activations ------------------------------------------------------
    spans: dict[str, tuple[float, float, Activation]] = {}
    for activation in graph.activations.values():
        begin = sec(activation.started_at)
        end = sec(activation.ended_at) if activation.ended_at else now_s
        spans[activation.id] = (begin, max(end, begin), activation)

    rounds = sorted({a.round for a in graph.activations.values()})
    window: dict[int, tuple[float, float]] = {}
    for rnd in rounds:
        in_round = [s for s in spans.values() if s[2].round == rnd]
        window[rnd] = (min(s[0] for s in in_round), max(s[1] for s in in_round))
    last_round = rounds[-1] if rounds else 0

    # -- message x placement ---------------------------------------------
    #
    # An envelope carries no timestamp — it is a value, not an occurrence — so
    # it is placed where it came into being: the moment its generating turn
    # ended. A supervisor's message has no generating turn at all, so it lands
    # just after the round that provoked it.
    labels = {mid: f"e{i + 1}" for i, mid in enumerate(graph.messages)}
    message_x: dict[str, float] = {}
    for mid, envelope in graph.messages.items():
        generator = graph.generator_of(mid)
        if generator in spans:
            message_x[mid] = spans[generator][1]
        else:
            message_x[mid] = window.get(envelope.round, (now_s, now_s))[1] + 0.35

    consumed = graph.consumed()
    delivered = graph.delivered()
    # Orphaned per *contribution*, as `OrphanedEvent` counts it: a turn whose
    # addressed message was delivered has been heard, whatever became of the
    # copy it left for the record. Marking those red would put a warning on
    # every public message in a healthy run.
    orphaned = {m.id for m in graph.undelivered()}
    recipients: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "delivers":
            target = spans.get(edge.dst)
            if target is not None:
                recipients.setdefault(edge.src, []).append(
                    f"{target[2].participant}@r{target[2].round}"
                )

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []
    sys_marks: dict[int, dict[str, Any]] = {}

    for aid, (begin, end, activation) in spans.items():
        lane = lane_of.get(activation.participant, lane_of[SUPERVISOR])
        live = activation.ended_at is None
        outputs = graph.outputs_of(aid)
        inputs = graph.inputs_of(aid)
        nodes.append(
            {
                "kind": "activation",
                "x": begin,
                "y": lane,
                "state": "error"
                if activation.error
                else "live"
                if live
                else _classify(graph, aid, outputs),
                "hover": _activation_hover(
                    graph, activation, inputs, outputs, labels, end - begin, live
                ),
            }
        )
        bars.append(
            {
                "x0": begin,
                "x1": end,
                "y": lane,
                "live": live,
                "hover": f"{activation.participant} · round {activation.round} · "
                f"{end - begin:.2f}s",
            }
        )

    for mid, envelope in graph.messages.items():
        lane = lane_of.get(envelope.sender, lane_of[SUPERVISOR])
        nodes.append(
            {
                "kind": "event",
                "x": message_x[mid],
                "y": lane,
                "state": "terminal"
                if envelope.terminal
                else "supervisor"
                if envelope.sender == SUPERVISOR
                else "orphan"
                if mid in orphaned
                else "event",
                "label": labels[mid],
                "hover": _message_hover(
                    graph, envelope, labels, recipients.get(mid, ()), mid in consumed
                ),
            }
        )

    # -- edges ------------------------------------------------------------
    for edge in graph.edges:
        if edge.kind == "generates":
            span = spans.get(edge.src)
            if span is None or edge.dst not in message_x:
                continue
            lane = lane_of.get(span[2].participant, lane_of[SUPERVISOR])
            edges.append(
                _segment(
                    "generate",
                    (span[0], lane),
                    (message_x[edge.dst], lane),
                    f"{span[2].participant} generated {labels.get(edge.dst, '')}",
                )
            )
        elif edge.kind == "delivers":
            span = spans.get(edge.dst)
            if span is None or edge.src not in message_x:
                continue
            sender = graph.messages[edge.src].sender
            # The recipient's own reaction, in its own vocabulary: consume,
            # wait, discard, reroute. A delivery edge that says `consume` when
            # the agent said `wait` would draw a run that did not happen.
            reaction = edge.action if edge.action in _REACTIONS else "deliver"
            edges.append(
                _segment(
                    reaction,
                    (message_x[edge.src], lane_of.get(sender, lane_of[SUPERVISOR])),
                    (span[0], lane_of.get(span[2].participant, lane_of[SUPERVISOR])),
                    f"{reaction} — {span[2].participant} on "
                    f"{labels.get(edge.src, '')} from {sender}",
                )
            )
        elif edge.kind == "intervenes":
            mark = sys_marks.setdefault(
                edge.round,
                {
                    "x": window.get(edge.round, (now_s, now_s))[1] + 0.35,
                    "y": lane_of[SUPERVISOR],
                    "actions": [],
                },
            )
            mark["actions"].append(f"{edge.action}: {edge.reason or '—'}")
            if edge.action == "emit":
                continue  # the notice itself is a node on the supervisor lane
            x_src = message_x.get(edge.src)
            if x_src is None:
                continue
            sender = graph.messages[edge.src].sender
            y_src = lane_of.get(sender, lane_of[SUPERVISOR])
            if edge.action == "inject":
                # The supervisor reaching down to rewrite a message in flight.
                edges.append(
                    _segment(
                        "inject",
                        (mark["x"], lane_of[SUPERVISOR]),
                        (x_src, y_src),
                        f"inject into {labels.get(edge.src, '')} — {edge.reason}",
                    )
                )
            elif edge.action == "reroute":
                edges.append(
                    _segment(
                        "reroute",
                        (x_src, y_src),
                        (mark["x"], lane_of.get(edge.dst, lane_of[SUPERVISOR])),
                        f"reroute {labels.get(edge.src, '')} to {edge.dst} "
                        f"— {edge.reason}",
                    )
                )
            elif edge.action == "discard":
                edges.append(
                    _segment(
                        "discard",
                        (x_src, y_src),
                        (x_src + 0.6, y_src),
                        f"discard {labels.get(edge.src, '')} — {edge.reason}",
                    )
                )

    # -- wait spans -------------------------------------------------------
    #
    # Only for rounds that have finished: a participant that has not activated
    # *yet* in the round now running is not waiting, it is pending.
    acted = {(a.participant, a.round) for a in graph.activations.values()}
    for rnd in rounds:
        if live and rnd == last_round:
            continue
        begin, end = window[rnd]
        for name in names:
            if (name, rnd) in acted:
                continue
            edges.append(
                _segment(
                    "wait",
                    (begin, lane_of[name]),
                    (end, lane_of[name]),
                    f"wait — {name} did not activate in round {rnd}",
                )
            )

    busy = sum(end - begin for begin, end, _ in spans.values())
    return {
        "lanes": lanes,
        "now": now_s,
        "nodes": nodes,
        "edges": edges,
        "bars": bars,
        "sys": [
            {
                "x": mark["x"],
                "y": mark["y"],
                "hover": f"<b>supervisor · round {rnd}</b><br>"
                + "<br>".join(mark["actions"]),
            }
            for rnd, mark in sorted(sys_marks.items())
        ],
        "stats": {
            "round": last_round,
            "rounds": len(rounds),
            "activations": len(graph.activations),
            "events": len(graph.messages),
            "consumed": len(consumed),
            "outstanding": len(graph.unsettled()),
            "undelivered": len(graph.undelivered()),
            "elapsed": now_s,
            "busy": busy,
        },
    }


def _empty(lanes: list[str]) -> dict[str, Any]:
    return {
        "lanes": lanes,
        "now": 0.0,
        "nodes": [],
        "edges": [],
        "bars": [],
        "sys": [],
        "stats": {
            "round": 0,
            "rounds": 0,
            "activations": 0,
            "events": 0,
            "consumed": 0,
            "outstanding": 0,
            "undelivered": 0,
            "elapsed": 0.0,
            "busy": 0.0,
        },
    }


def _segment(
    action: str, src: tuple[float, int], dst: tuple[float, int], hover: str
) -> dict[str, Any]:
    return {
        "action": action,
        "x": [src[0], dst[0]],
        "y": [src[1], dst[1]],
        "hover": hover,
    }


def _classify(
    graph: InteractionGraph, activation_id: str, outputs: tuple[str, ...]
) -> str:
    """DIG's own discriminator, and the figure's node colour.

    Not a property of the record — `is_problem_reducing` is the paper's reading
    of it — which is why it is imported from the strategy rather than reinvented
    here.
    """
    if not outputs:
        return "quiet"
    return "reducing" if is_problem_reducing(graph, activation_id) else "expanding"


def _activation_hover(
    graph: InteractionGraph,
    activation: Activation,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    labels: dict[str, str],
    duration: float,
    live: bool,
) -> str:
    phase = "RUNNING" if live else "RESPOND" if outputs else "WAIT"
    kind = _classify(graph, activation.id, outputs)
    lines = [
        f"<b>{activation.participant}</b> ({phase}) · round {activation.round}",
        f"<b>Classification:</b> {_LABEL[kind]}",
        f"<b>Duration:</b> {duration:.2f}s",
    ]
    if inputs:
        lines.append("<b>Decisions:</b>")
        lines += [f"  {labels.get(mid, mid[:6])}: <b>consume</b>" for mid in inputs]
    else:
        lines.append("<b>Decisions:</b> none — activated with an empty buffer")
    if outputs:
        lines.append(
            "<b>Generated:</b> " + ", ".join(labels.get(m, m[:6]) for m in outputs)
        )
    if activation.error:
        lines.append(f"<b>Error:</b> {activation.error}")
    return "<br>".join(lines)


_LABEL = {
    "reducing": "Problem-Reducing",
    "expanding": "Problem-Expanding",
    "quiet": "No output — waiting",
    "live": "still running",
    "error": "failed",
}


def _message_hover(
    graph: InteractionGraph,
    envelope: Envelope,
    labels: dict[str, str],
    to: tuple[str, ...] | list[str],
    consumed: bool,
) -> str:
    lines = [
        f"<b>{labels.get(envelope.id, '')}</b> — {envelope.sender} "
        f"({envelope.visibility}, round {envelope.round})",
        "<b>Addressed to:</b> "
        + (", ".join(envelope.recipients) if envelope.recipients else "nobody named"),
        f"<b>Delivered to:</b> {', '.join(to) if to else 'nobody yet'}",
        f"<b>Consumed:</b> {'yes' if consumed else 'not yet'}",
    ]
    if envelope.terminal:
        lines.append("<b>Terminal:</b> this message claims the task is done")
    if envelope.parents:
        lines.append(
            "<b>Derived from:</b> "
            + ", ".join(labels.get(p, p[:6]) for p in envelope.parents)
        )
    history = [
        f"{e.action} ({e.reason})"
        for e in graph.edges
        if e.kind == "intervenes" and e.src == envelope.id
    ]
    if history:
        lines.append("<b>Interventions:</b> " + ", ".join(history))
    lines.append("<b>Content:</b>")
    body = " ".join(envelope.content.split())
    if len(body) > CLIP:
        body = body[: CLIP - 1] + "…"
    lines += [f"  {line}" for line in textwrap.wrap(body, 58)]
    return "<br>".join(lines)
