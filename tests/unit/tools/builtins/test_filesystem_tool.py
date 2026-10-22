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


@pytest.fixture()
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory: Path = tmp_path_factory.mktemp("outside")
    (directory / "secret.txt").write_text("SECRET", encoding="utf-8")
    return directory


def test_list_dir_skips_symlinks_out_of_workspace(
    workspace: Path, outside: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "linkdir").symlink_to(outside)
    (workspace / "linkfile.txt").symlink_to(outside / "secret.txt")
    assert fs_tool.list_dir() == []


def test_search_skips_symlinks_out_of_workspace(
    workspace: Path, outside: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "linkdir").symlink_to(outside)
    (workspace / "linkfile.txt").symlink_to(outside / "secret.txt")
    assert fs_tool.search("SECRET") == "No matches found"


def test_symlinks_inside_workspace_are_kept(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "real").mkdir()
    (workspace / "real" / "a.txt").write_text("TARGET", encoding="utf-8")
    (workspace / "alias").symlink_to(workspace / "real")
    (workspace / "link.txt").symlink_to(workspace / "real" / "a.txt")
    assert "alias/a.txt" in fs_tool.list_dir()
    assert "link.txt:1: TARGET" in fs_tool.search("TARGET")


@pytest.fixture()
def ten_lines(workspace: Path) -> Path:
    path = workspace / "ten.txt"
    path.write_text("".join(f"line {i}\n" for i in range(10)), encoding="utf-8")
    return path


def test_read_file_without_range_returns_the_whole_file(
    ten_lines: Path, fs_tool: FilesystemTool
) -> None:
    assert fs_tool.read_file("ten.txt") == ten_lines.read_text(encoding="utf-8")


def test_read_file_with_offset_and_limit_returns_those_lines(
    ten_lines: Path, fs_tool: FilesystemTool
) -> None:
    assert fs_tool.read_file("ten.txt", offset=3, limit=2) == "line 3\nline 4\n"
    assert fs_tool.read_file("ten.txt", offset=8) == "line 8\nline 9\n"
    assert fs_tool.read_file("ten.txt", limit=1) == "line 0\n"


def test_read_file_limit_past_the_end_stops_at_the_end(
    ten_lines: Path, fs_tool: FilesystemTool
) -> None:
    assert fs_tool.read_file("ten.txt", offset=9, limit=1000) == "line 9\n"


@pytest.mark.parametrize(
    ("offset", "limit", "match"),
    [(-1, None, "offset"), (0, 0, "limit"), (10, None, "past the end")],
)
def test_read_file_rejects_a_bad_range(
    ten_lines: Path,
    fs_tool: FilesystemTool,
    offset: int,
    limit: int | None,
    match: str,
) -> None:
    with pytest.raises(ToolError, match=match):
        fs_tool.read_file("ten.txt", offset=offset, limit=limit)


def test_edit_file_replaces_the_one_occurrence(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    assert "1 occurrence" in fs_tool.edit_file("calc.py", "a - b", "a + b")
    assert (workspace / "calc.py").read_text(encoding="utf-8") == (
        "def add(a, b):\n    return a + b\n"
    )


def test_edit_file_refuses_an_ambiguous_edit_unless_replace_all(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "x.txt").write_text("a\na\n", encoding="utf-8")
    with pytest.raises(ToolError, match="2 times"):
        fs_tool.edit_file("x.txt", "a", "b")
    assert (workspace / "x.txt").read_text(encoding="utf-8") == "a\na\n"
    assert "2 occurrences" in fs_tool.edit_file("x.txt", "a", "b", replace_all=True)
    assert (workspace / "x.txt").read_text(encoding="utf-8") == "b\nb\n"


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [("missing", "x", "not found"), ("", "x", "empty"), ("a", "a", "the same")],
)
def test_edit_file_rejects_an_edit_it_cannot_make(
    workspace: Path, fs_tool: FilesystemTool, old: str, new: str, match: str
) -> None:
    (workspace / "x.txt").write_text("a\n", encoding="utf-8")
    with pytest.raises(ToolError, match=match):
        fs_tool.edit_file("x.txt", old, new)
    assert (workspace / "x.txt").read_text(encoding="utf-8") == "a\n"


def test_edit_file_keeps_windows_line_endings(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    fs_tool.edit_file("crlf.txt", "two", "three")
    assert (workspace / "crlf.txt").read_bytes() == b"one\r\nthree\r\n"


def test_edit_file_missing_file_and_traversal_raise(fs_tool: FilesystemTool) -> None:
    with pytest.raises(ToolError, match="missing.txt"):
        fs_tool.edit_file("missing.txt", "a", "b")
    with pytest.raises(ToolError, match="escapes"):
        fs_tool.edit_file("../outside.txt", "a", "b")


def test_edit_file_is_exposed_to_the_model(fs_tool: FilesystemTool) -> None:
    names = [t.name for t in fs_tool._expand_namespace()]
    assert "FilesystemTool_edit_file" in names
