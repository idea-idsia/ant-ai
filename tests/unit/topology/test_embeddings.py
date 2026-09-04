from __future__ import annotations

import builtins

import pytest

from ant_ai.embeddings import Embedder
from ant_ai.embeddings.backends.sentence_transformer import SentenceTransformerEmbedder

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def test_importing_topology_does_not_require_the_extra() -> None:
    import ant_ai.topology  # noqa: F401


async def test_missing_extra_raises_and_names_it(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"ant-ai\[topology\]"):
        await SentenceTransformerEmbedder().aembed(["hello"])


async def test_empty_input_never_loads_the_model() -> None:
    """~90 MB of model must not load for a call that has nothing to embed."""
    embedder = SentenceTransformerEmbedder()

    assert await embedder.aembed([]) == []
    assert embedder._encoder is None


def test_backend_satisfies_the_protocol() -> None:
    assert isinstance(SentenceTransformerEmbedder(), Embedder)
    assert SentenceTransformerEmbedder(model="x").model_id == "x"
