from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """The terminal agent's own folder: `$ANT_HOME`, by default `~/.ant`."""
    return Path(os.environ.get("ANT_HOME", Path.home() / ".ant"))


def load_env(path: Path | None = None) -> list[str]:
    """Read `KEY=value` lines from `~/.ant/env` into the environment.

    Variables already set win over the file, so a one-off `ANT_MODEL=... ant`
    still works. Blank lines and `#` comments are skipped.

    Returns:
        Warnings for the user, e.g. a file that others can read.
    """
    path = path or home() / "env"
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    if os.name == "posix" and path.stat().st_mode & 0o077:
        return [f"{path} can be read by other users: chmod 600 {path}"]
    return []
