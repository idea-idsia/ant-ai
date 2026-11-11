from __future__ import annotations

import os
import uuid
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path`, creating its parent directories, so a reader never sees half a file.

    The text goes to a temporary file next to the target, which then replaces it:
    a crash leaves the old file or the new one, never a mix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
