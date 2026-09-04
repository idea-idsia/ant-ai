from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

_MISSING = (
    "SentenceTransformerEmbedder requires the 'sentence-transformers' package. "
    "Install it with the topology extra: pip install 'ant-ai[topology]'"
)


class SentenceTransformerEmbedder(BaseModel):
    """`Embedder` backed by a local sentence-transformers model.

    Defaults to `all-MiniLM-L6-v2` (384-d), the encoder used by DyTopo
    (arXiv:2602.06039), so semantic matching reproduces the paper and runs
    offline.

    The model is loaded lazily on first use rather than at construction: it is
    ~90 MB, and loading it eagerly would stall the first round of every run
    that never ends up embedding anything.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name or local path.",
    )

    _encoder: Any = PrivateAttr(default=None)
    _lock: asyncio.Lock | None = PrivateAttr(default=None)

    @property
    def model_id(self) -> str:
        return self.model

    def _load(self) -> Any:
        try:
            from sentence_transformers import (  # ty: ignore[unresolved-import]
                SentenceTransformer,
            )
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(_MISSING) from exc
        return SentenceTransformer(self.model)

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._lock is None:
            self._lock = asyncio.Lock()

        if self._encoder is None:
            async with self._lock:
                if self._encoder is None:
                    self._encoder = await asyncio.to_thread(self._load)

        # Encoding is CPU-bound; keep it off the event loop.
        vectors = await asyncio.to_thread(
            self._encoder.encode, texts, convert_to_numpy=True
        )
        return [[float(x) for x in vec] for vec in vectors]
