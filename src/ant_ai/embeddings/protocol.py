from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Interface for a model that turns text into dense vectors.

    A sibling of `ChatLLM` rather than an extension of it: embedding and chat
    backends are rarely the same object, and widening `ChatLLM` would force
    every implementation — test doubles included — to grow a method it cannot
    honour.
    """

    @property
    def model_id(self) -> str:
        """Identifier of the underlying model.

        Recorded in run provenance so a trace says which encoder produced it.
        The embedder instance itself is not serialisable; this string is.
        """
        ...

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts*, returning one vector per input, in order.

        Args:
            texts: Strings to embed. May be empty.

        Returns:
            One vector per input string, in the same order.
        """
        ...
