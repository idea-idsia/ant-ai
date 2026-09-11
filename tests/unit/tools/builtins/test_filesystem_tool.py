from __future__ import annotations

from pathlib import Path

import pytest

from ant_ai.tools import ToolError
from ant_ai.tools.builtins.filesystem_tool import FilesystemTool


def test_read_file_returns_content(workspace: Path, fs_tool: FilesystemTool) -> None:
    (workspace / "hello.txt").write_text("hello world", encoding="utf-8")
    assert fs_tool.read_file("hello.txt") == "hello world"


def test_read_file_not_found_raises(workspace: Path, fs_tool: FilesystemTool) -> None:
    with pytest.raises(ToolError, match="missing.txt"):
        fs_tool.read_file("missing.txt")


def test_read_file_path_traversal_raises(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    with pytest.raises(ToolError, match="escapes"):
        fs_tool.read_file("../../etc/passwd")


def test_read_file_absolute_path_raises(fs_tool: FilesystemTool) -> None:
    with pytest.raises(ToolError, match="escapes"):
        fs_tool.read_file("/etc/passwd")


def test_write_file_creates_file(workspace: Path, fs_tool: FilesystemTool) -> None:
    fs_tool.write_file("output.txt", "content")
    assert (workspace / "output.txt").read_text(encoding="utf-8") == "content"


def test_write_file_reports_byte_count(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    result = fs_tool.write_file("f.txt", "abc")
    assert "3" in result


def test_write_file_creates_parent_directories(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    fs_tool.write_file("a/b/c/file.txt", "nested")
    assert (workspace / "a" / "b" / "c" / "file.txt").exists()


def test_write_file_overwrites_existing(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "f.txt").write_text("old", encoding="utf-8")
    fs_tool.write_file("f.txt", "new")
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "new"


def test_write_file_path_traversal_raises(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    with pytest.raises(ToolError, match="escapes"):
        fs_tool.write_file("../../evil.txt", "bad")


def test_list_dir_returns_entries(workspace: Path, fs_tool: FilesystemTool) -> None:
    (workspace / "a.txt").touch()
    (workspace / "b.txt").touch()
    result = fs_tool.list_dir("")
    assert "a.txt" in result and "b.txt" in result


def test_list_dir_empty_directory(workspace: Path, fs_tool: FilesystemTool) -> None:
    assert fs_tool.list_dir("") == []


def test_list_dir_subdirectory(workspace: Path, fs_tool: FilesystemTool) -> None:
    (workspace / "sub").mkdir()
    (workspace / "sub" / "x.py").touch()
    assert fs_tool.list_dir("sub") == ["x.py"]


def test_list_dir_not_found_raises(workspace: Path, fs_tool: FilesystemTool) -> None:
    with pytest.raises(ToolError, match="nonexistent"):
        fs_tool.list_dir("nonexistent")


def test_list_dir_path_traversal_raises(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    with pytest.raises(ToolError, match="escapes"):
        fs_tool.list_dir("../../etc")


def test_search_returns_matching_lines(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "code.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    result = fs_tool.search("def foo")
    assert "code.py" in result and "def foo" in result


def test_search_includes_line_numbers(workspace: Path, fs_tool: FilesystemTool) -> None:
    (workspace / "f.py").write_text("line1\nTARGET\nline3\n", encoding="utf-8")
    assert ":2:" in fs_tool.search("TARGET")


def test_search_no_matches(workspace: Path, fs_tool: FilesystemTool) -> None:
    (workspace / "f.txt").write_text("nothing here", encoding="utf-8")
    assert fs_tool.search("ZZZNOMATCH") == "No matches found"


def test_search_invalid_regex_raises(workspace: Path, fs_tool: FilesystemTool) -> None:
    with pytest.raises(ToolError, match="invalid regex"):
        fs_tool.search("[invalid(regex")


def test_search_scoped_to_subdirectory(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "sub").mkdir()
    (workspace / "sub" / "a.py").write_text("TARGET", encoding="utf-8")
    (workspace / "b.py").write_text("TARGET", encoding="utf-8")
    result = fs_tool.search("TARGET", path="sub")
    assert "sub/a.py" in result and "b.py" not in result


def test_search_path_traversal_raises(workspace: Path, fs_tool: FilesystemTool) -> None:
    with pytest.raises(ToolError, match="escapes"):
        fs_tool.search("root", path="../../")
