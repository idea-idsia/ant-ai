from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from ant_ai.core.events import Event, FinalAnswerEvent
from ant_ai.core.types import InvocationContext
from ant_ai.topology.participant import (
    Brief,
    Envelope,
    Participant,
    ParticipantProfile,
    PeerTool,
    Turn,
)


class FakeEmbedder:
    """Deterministic embedder: fixed vectors per exact string, else a default.

    Lets a test state the similarity structure it wants instead of hoping a
    real encoder produces it.
    """

    def __init__(
        self, vectors: dict[str, list[float]], default: list[float] | None = None
    ):
        self.vectors = vectors
        self.default = default or [0.0, 0.0, 1.0]
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "fake-embedder"

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors.get(t, self.default) for t in texts]


class FakeParticipant:
    """Scripted participant. No LLM, no network."""

    def __init__(
        self,
        name: str,
        *,
        query: str = "",
        key: str = "",
        message: str = "",
        submitted: bool = False,
        invoked: tuple[str, ...] = (),
        raises: str | None = None,
        description: str = "",
    ) -> None:
        self._name = name
        self.query = query
        self.key = key
        self.message = message or f"{name} says something"
        self.submitted = submitted
        self.invoked = invoked
        self.raises = raises
        self.description = description
        self.bound: list[frozenset[str]] = []
        self.briefs: list[Brief] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> ParticipantProfile:
        return ParticipantProfile(name=self._name, description=self.description)

    def as_tool(self) -> Any:
        return PeerTool.for_participant(self)

    async def bind_peers(self, peers: Mapping[str, Participant]) -> bool:
        self.bound.append(frozenset(peers))
        return True

    def _make_turn(self, round: int = 0) -> Turn:
        return Turn(
            participant=self._name,
            outputs=(
                Envelope(
                    sender=self._name,
                    content=self.message,
                    visibility="public",
                    round=round,
                ),
                Envelope(
                    sender=self._name,
                    content=f"private from {self._name}",
                    visibility="private",
                    round=round,
                ),
            ),
            query=self.query,
            key=self.key,
            invoked=self.invoked,
            submitted=self.submitted,
        )

    async def act(
        self, brief: Brief, *, ctx: InvocationContext | None = None
    ) -> AsyncIterator[Event | Turn]:
        self.briefs.append(brief)
        if self.raises:
            raise RuntimeError(self.raises)
        yield FinalAnswerEvent(content=self.message)
        yield self._make_turn(brief.round)


class ScriptedParticipant:
    """A participant whose behaviour is a per-round script.

    `FakeParticipant` says the same thing forever, which is enough for routing
    tests but cannot express the *shapes* the DIG detectors look for — an agent
    that submits early, one that falls silent, one that keeps talking. Each
    entry is `(message, submitted)`; `None` means the turn produces no message
    at all, which is what exhausts reachable work.
    """

    def __init__(
        self,
        name: str,
        script: list[tuple[str, bool] | None],
        *,
        description: str = "",
        private: bool = True,
    ) -> None:
        self._name = name
        self.script = script
        self.description = description
        self.private = private
        self.briefs: list[Brief] = []
        self.bound: list[frozenset[str]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> ParticipantProfile:
        return ParticipantProfile(name=self._name, description=self.description)

    def as_tool(self) -> Any:
        return PeerTool.for_participant(self)

    async def bind_peers(self, peers: Mapping[str, Participant]) -> bool:
        self.bound.append(frozenset(peers))
        return True

    async def act(
        self, brief: Brief, *, ctx: InvocationContext | None = None
    ) -> AsyncIterator[Event | Turn]:
        self.briefs.append(brief)
        entry = (
            self.script[brief.round]
            if brief.round < len(self.script)
            else self.script[-1]
        )
        if entry is None:
            yield Turn(participant=self._name)
            return
        message, submitted = entry
        yield FinalAnswerEvent(content=message)
        outputs = [
            Envelope(
                sender=self._name,
                content=message,
                visibility="public",
                round=brief.round,
            )
        ]
        if self.private:
            outputs.append(
                Envelope(
                    sender=self._name,
                    content=f"private: {message}",
                    visibility="private",
                    round=brief.round,
                )
            )
        yield Turn(participant=self._name, outputs=tuple(outputs), submitted=submitted)


class StubLLM:
    """Placeholder `ChatLLM`; tests here never reach a model call."""

    async def ainvoke(self, messages, **kwargs):  # pragma: no cover
        raise AssertionError("StubLLM should not be called")

    def invoke(self, messages, **kwargs):  # pragma: no cover
        raise AssertionError("StubLLM should not be called")
