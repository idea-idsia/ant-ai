from __future__ import annotations

import os
from pathlib import Path

import pytest

from ant_ai.core._files import write_atomic


def test_write_creates_the_file_and_its_directories(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "notes.md"
    write_atomic(path, "hello\n")
    assert path.read_text(encoding="utf-8") == "hello\n"


def test_write_replaces_the_file_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("old\n", encoding="utf-8")
    write_atomic(path, "new\n")
    assert path.read_text(encoding="utf-8") == "new\n"
    assert [p.name for p in tmp_path.iterdir()] == ["notes.md"]


def test_a_failed_replace_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "notes.md"
    path.write_text("old\n", encoding="utf-8")

    def crash(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("ant_ai.core._files.os.replace", crash)
    with pytest.raises(OSError, match="No space left on device"):
        write_atomic(path, "new\n")
    assert path.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["notes.md"]


def test_the_text_is_on_disk_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    fsync, replace = os.fsync, os.replace
    monkeypatch.setattr(
        "ant_ai.core._files.os.fsync", lambda fd: (calls.append("fsync"), fsync(fd))
    )
    monkeypatch.setattr(
        "ant_ai.core._files.os.replace",
        lambda src, dst: (calls.append("replace"), replace(src, dst)),
    )
    write_atomic(tmp_path / "notes.md", "hello\n")
    assert calls == ["fsync", "replace"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_a_write_keeps_the_permissions_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o600)
    write_atomic(path, "new\n")
    assert path.stat().st_mode & 0o777 == 0o600


def test_a_symlinked_file_is_written_through_the_link(tmp_path: Path) -> None:
    target = tmp_path / "real.md"
    link = tmp_path / "notes.md"
    link.symlink_to(target)
    write_atomic(link, "hello\n")
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "hello\n"
