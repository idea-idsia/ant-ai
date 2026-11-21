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

    Args:
        path: The file to read, instead of `home() / "env"`.

    Returns:
        Warnings for the user, e.g. a line without a name or a file that
        others can read.
    """
    path = path or home() / "env"
    if not path.is_file():
        return []
    warnings: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if not key:
            warnings.append(f"{path}:{number}: no variable name, line skipped")
            continue
        os.environ.setdefault(key, value.strip("'\""))
    if os.name == "posix" and path.stat().st_mode & 0o077:
        warnings.append(f"{path} can be read by other users: chmod 600 {path}")
    return warnings
