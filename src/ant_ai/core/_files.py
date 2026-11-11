from __future__ import annotations

import os
import uuid
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path`, creating its parent directories, so a reader never sees half a file.

    The text goes to a temporary file next to the target, which then replaces it:
    a crash, even of the machine, leaves the old file or the new one, never a mix.
    The new file keeps the permissions of the old one. A symlink is followed, so
    the file it points to is replaced, not the link.
    """
    path: Path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode: int | None = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        mode = None
    tmp: Path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # Created with the old permissions, so a private file is never readable by others.
        fd: int = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else mode
        )
        with open(fd, "w", encoding="utf-8") as f:
            if mode is not None:
                os.fchmod(f.fileno(), mode)
            f.write(text)
            f.flush()
            # On disk before the rename, or a power cut can leave an empty file.
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
