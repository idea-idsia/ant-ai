"""`ant`: a minimal coding agent in the terminal, built on ant-ai.

Needs the `cli` extra (`pip install "ant-ai[cli]"`) for the terminal output.
"""

try:
    from ant_ai.cli.app import Session, main
except ImportError as e:  # pragma: no cover - depends on the install
    raise ImportError(
        'The terminal agent needs the cli extra: pip install "ant-ai[cli]"'
    ) from e

__all__ = ["Session", "main"]
