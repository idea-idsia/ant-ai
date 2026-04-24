from collections.abc import AsyncIterator
from typing import Protocol

from ant_ai.core.events import Event
from ant_ai.core.result import StepResult
from ant_ai.core.types import InvocationContext, State


class Step(Protocol):
    """Minimal interface every executor step must satisfy.

    Implementations should be `BaseModel` subclasses so they can be declared
    as Pydantic fields on the executor and benefit from validation,
    serialization, and introspection for free.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier used in logs and hook dispatch."""
        ...

    def run(
        self,
        state: State,
        ctx: InvocationContext | None,
    ) -> AsyncIterator[Event | StepResult]:
        """Stream events as they happen, then yield the final `StepResult`.

        Args:
            state: Current agent state, including the conversation history.
            ctx: Invocation context, or None if not available.

        Returns:
            An async iterator of `Event` items followed by a single `StepResult`.
        """
        ...
