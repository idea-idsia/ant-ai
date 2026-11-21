from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter

from ant_ai.cli import config
from ant_ai.core.message import AnyMessage, BaseMessage

# Loaded by their `kind`, so each message comes back as its own class (tool calls, ids).
_MESSAGES: TypeAdapter[list[AnyMessage]] = TypeAdapter(list[AnyMessage])


def root() -> Path:
    """Where conversations are kept: `~/.ant/sessions` (see `config.home`)."""
    return config.home() / "sessions"


def folder(workspace: Path, base: Path | None = None) -> Path:
    """The folder of the conversations of `workspace`.

    Args:
        workspace: The folder the agent works in.
        base: Where all conversations are kept, instead of `root()`.

    Returns:
        One folder per workspace: its path made readable, then a short hash of
        the path, since two paths can read the same (`a_b`, `a-b`).
    """
    path = str(workspace.resolve())
    slug = re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-")
    digest = hashlib.sha256(path.encode()).hexdigest()[:8]
    return (base or root()) / f"{slug}-{digest}"


def new_file(workspace: Path, base: Path | None = None) -> Path:
    """The file of a new conversation of `workspace` (see `folder`), not created yet."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return folder(workspace, base) / f"{stamp}.json"


def latest(workspace: Path, base: Path | None = None) -> Path | None:
    """The most recent conversation saved for `workspace`, if any."""
    files = sorted(folder(workspace, base).glob("*.json"))
    return files[-1] if files else None


def save(path: Path, messages: list[BaseMessage]) -> None:
    """Write the conversation to `path`, replacing it whole.

    The file is readable by its owner only: it holds prompts and tool output.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps([m.model_dump(mode="json") for m in messages], indent=1)
    # A temporary file of its own (mode 600): two `ant -c` in one folder can
    # save at once, and a reader never sees half a file.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load(path: Path) -> list[BaseMessage]:
    """The messages of a saved conversation, each as its own class."""
    return list(_MESSAGES.validate_json(path.read_bytes()))
