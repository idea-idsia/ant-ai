from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

# Threads of one process take turns; `flock` adds the other processes.
_DIR_LOCK: threading.Lock = threading.Lock()


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path`, creating its parent directories, so a reader never sees half a file.

    The text goes to a temporary file next to the target, which then replaces it:
    a crash, even of the machine, leaves the old file or the new one, never a mix.
    A new file is readable by its owner only; an existing one keeps its
    permissions. A symlink is followed, so the file it points to is replaced,
    not the link.
    """
    path: Path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode: int = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        mode = 0o600
    tmp: Path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # Created with the final permissions, so a private file is never readable by others.
        fd: int = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with open(fd, "w", encoding="utf-8") as f:
            # `os.open` applies the umask; set the mode exactly.
            os.fchmod(f.fileno(), mode)
            f.write(text)
            f.flush()
            # On disk before the rename, or a power cut can leave an empty file.
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # The rename is in the directory: sync it too, or a power cut can undo it.
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def lock_dir(path: Path) -> Generator[None]:
    """Hold an exclusive lock on the directory of `path`, creating it if needed.

    The lock is taken by every thread and, on POSIX, every process that calls
    this for a file in the same directory, so a check of the file and its
    replacement happen as one step. It is advisory: a process that writes
    without it, such as an editor, is not stopped. On Windows only the threads
    of this process are coordinated.
    """
    directory: Path = path.resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    with _DIR_LOCK:
        if os.name != "posix":
            yield
            return
        import fcntl

        fd: int = os.open(directory, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # Closing the descriptor releases the lock.


def _fsync_dir(directory: Path) -> None:
    """Flush the entries of `directory` to disk, where the platform allows it."""
    if os.name != "posix":
        return
    fd: int = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
