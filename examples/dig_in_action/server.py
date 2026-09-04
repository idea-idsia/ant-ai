"""A live view of a run that has not finished yet.

The interesting property of DIG's figure is that it is drawn *during* the run:
the time axis grows, activations appear as they start, and a repair shows up at
the moment the supervisor makes it. So the page is not rendered — it is fed. A
background task drives `Ensemble.stream`, the graph it mutates is projected on
demand, and browsers hold an SSE connection that hands them the current
projection a few times a second.

Nothing here is specific to the example's scripted cast. `Session` takes a
factory that returns a fresh `(scenario, ensemble)` pair, so pointing this at a
colony of real agents is a change of one function:

    from ant_ai.a2a import Colony
    from ant_ai.topology.builtins import DigToHeal, DyTopo

    def factory(*, heal):
        colony = build_colony()
        colony.topology(DyTopo(embedder=embedder) | DigToHeal())
        return scenario, colony.ensemble()
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ant_ai.core.events import CompletedEvent, Event, HealingEvent, TopologyEvent
from examples.dig_in_action.projection import project
from examples.dig_in_action.scenario import Scenario, build_ensemble, build_scenario

__all__ = ["Session", "create_app"]

VIEWER = Path(__file__).parent / "viewer.html"

type Factory = Callable[..., tuple[Scenario, Any]]


class Session:
    """One run at a time, restartable, observable while it is going."""

    def __init__(self, factory: Factory, *, heal: bool = True) -> None:
        self._factory = factory
        self._task: asyncio.Task[None] | None = None
        self.heal = heal
        self.scenario, self.ensemble = factory(heal=heal)
        self.status = "idle"
        self.answer = ""
        self.findings: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []
        self.started = datetime.now(UTC)

    async def start(self, *, heal: bool) -> None:
        """Abandon whatever is running and begin again, from a fresh cast.

        Fresh, not reset: participants carry state, so reusing them would leak
        one condition's history into the next — the same reason a comparison
        loop rebuilds them per run.
        """
        await self.stop()
        self.heal = heal
        self.scenario, self.ensemble = self._factory(heal=heal)
        self.status = "running"
        self.answer = ""
        self.findings = []
        self.log = []
        self.started = datetime.now(UTC)
        self._task = asyncio.create_task(self._drive())

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _drive(self) -> None:
        try:
            async for event in self.ensemble.stream(self.scenario.task):
                self._record(event)
            self.status = "finished"
        except asyncio.CancelledError:
            self.status = "cancelled"
            raise
        except Exception as exc:  # pragma: no cover - surfaced in the page
            self.status = f"error: {exc}"

    def _record(self, event: Event) -> None:
        elapsed = (datetime.now(UTC) - self.started).total_seconds()
        if isinstance(event, HealingEvent):
            self.findings.append(
                {
                    "round": event.round,
                    "pattern": event.pattern,
                    "detector": event.detector,
                    "explanation": event.content,
                    "interventions": list(event.interventions),
                    "at": elapsed,
                }
            )
        if isinstance(event, CompletedEvent):
            self.answer = event.content
        detail = event.content
        if isinstance(event, TopologyEvent):
            detail = f"round {event.round}: {len(event.links)} links"
        self.log.append({"at": elapsed, "kind": event.kind, "content": detail})
        del self.log[:-200]

    def _clock(self) -> datetime:
        """Now, or the moment the run stopped.

        A finished run has a finished figure: letting the clock keep going
        would stretch the time axis to the right forever and quietly turn "time
        so far" into "how long you have been looking at it".
        """
        if self.status == "running":
            return datetime.now(UTC)
        ends = [
            a.ended_at or a.started_at for a in self.ensemble.graph.activations.values()
        ]
        return max(ends) if ends else self.started

    def snapshot(self) -> dict[str, Any]:
        return {
            "task": self.scenario.task,
            "heal": self.heal,
            "status": self.status,
            "provenance": self.ensemble.provenance,
            "max_rounds": self.ensemble.pipeline.max_rounds,
            "figure": project(
                self.ensemble.graph,
                names=self.scenario.order,
                now=self._clock(),
                live=self.status == "running",
            ),
            "findings": self.findings,
            "log": self.log[-25:],
            "answer": self.answer,
            "verdict": self.scenario.verdict(),
        }


def default_factory(
    *,
    think: float = 1.2,
    patience: int = 4,
    max_rounds: int = 8,
    repeated_subproblem: bool = False,
) -> Factory:
    def factory(*, heal: bool) -> tuple[Scenario, Any]:
        scenario = build_scenario(think=think, patience=patience)
        return scenario, build_ensemble(
            scenario,
            heal=heal,
            max_rounds=max_rounds,
            repeated_subproblem=repeated_subproblem,
        )

    return factory


def create_app(
    factory: Factory, *, heal: bool = True, interval: float = 0.25
) -> FastAPI:
    session = Session(factory, heal=heal)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # The run starts with the server: opening the page mid-run is normal —
        # a late viewer gets the graph so far, not an empty canvas.
        await session.start(heal=heal)
        yield
        await session.stop()

    app = FastAPI(title="DIG in action", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return VIEWER.read_text(encoding="utf-8")

    @app.get("/api/snapshot")
    async def snapshot() -> JSONResponse:
        return JSONResponse(session.snapshot())

    @app.post("/api/run")
    async def run(heal: bool = True) -> JSONResponse:
        await session.start(heal=heal)
        return JSONResponse({"status": session.status, "heal": session.heal})

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def pump() -> AsyncIterator[str]:
            last = ""
            while True:
                payload = json.dumps(session.snapshot())
                # A finished run stops changing; keeping the connection quiet
                # then costs nothing and leaves the last frame on screen.
                if payload != last:
                    last = payload
                    yield f"data: {payload}\n\n"
                else:
                    yield ": idle\n\n"
                await asyncio.sleep(interval)

        return StreamingResponse(
            pump(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
