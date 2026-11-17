from __future__ import annotations

import json
import re
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
    """One folder per workspace, named after its path."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(workspace.resolve())).strip("-")
    return (base or root()) / slug


def new_file(workspace: Path, base: Path | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return folder(workspace, base) / f"{stamp}.json"


def latest(workspace: Path, base: Path | None = None) -> Path | None:
    """The most recent conversation saved for `workspace`, if any."""
    files = sorted(folder(workspace, base).glob("*.json"))
    return files[-1] if files else None


def save(path: Path, messages: list[BaseMessage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([m.model_dump(mode="json") for m in messages], indent=1))
    tmp.replace(path)


def load(path: Path) -> list[BaseMessage]:
    return list(_MESSAGES.validate_json(path.read_bytes()))
