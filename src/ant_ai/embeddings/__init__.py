"""Text embedding backends, and the protocol they satisfy."""

from ant_ai.embeddings.protocol import Embedder

__all__ = ["Embedder", "default_embedder"]


def default_embedder() -> Embedder:
    """The encoder a semantic matcher uses when none was chosen.

    `all-MiniLM-L6-v2` via sentence-transformers: local, offline, and the
    encoder DyTopo (arXiv:2602.06039) itself used, so the default reproduces the
    paper rather than quietly substituting for it.

    A function rather than a module-level instance so that importing this
    package never constructs a model, and so the missing-dependency error is
    raised where it can be acted on.

    Raises:
        ImportError: If the optional `topology` extra is not installed.
    """
    from ant_ai.embeddings.backends.sentence_transformer import (
        SentenceTransformerEmbedder,
    )

    return SentenceTransformerEmbedder()
