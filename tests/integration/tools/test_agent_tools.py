from __future__ import annotations

from pathlib import Path

import pytest

from ant_ai.tools.builtins.filesystem_tool import FilesystemTool
from ant_ai.tools.builtins.shell_tool import ShellTool


def test_filesystem_tool_is_a_namespace_tool() -> None:
    assert FilesystemTool().is_namespace is True


def test_filesystem_tool_expands_to_expected_tool_names() -> None:
    tools = FilesystemTool()._expand_namespace()
    names = {t.name for t in tools}
    assert {
        "FilesystemTool_read_file",
        "FilesystemTool_write_file",
        "FilesystemTool_list_dir",
        "FilesystemTool_search",
    }.issubset(names)


def test_filesystem_tool_read_file_schema() -> None:
    tools = FilesystemTool()._expand_namespace()
    read = next(t for t in tools if t.name == "FilesystemTool_read_file")
    assert read.parameters is not None
    props = read.parameters["properties"]
    assert "path" in props
    assert props["path"]["type"] == "string"
    assert read.parameters.get("required") == ["path"]


def test_filesystem_tool_write_file_schema() -> None:
    tools = FilesystemTool()._expand_namespace()
    write = next(t for t in tools if t.name == "FilesystemTool_write_file")
    assert write.parameters is not None
    props = write.parameters["properties"]
    assert "path" in props and "content" in props
    assert set(write.parameters.get("required", [])) == {"path", "content"}


def test_filesystem_tool_list_dir_schema_has_optional_path() -> None:
    tools = FilesystemTool()._expand_namespace()
    list_dir = next(t for t in tools if t.name == "FilesystemTool_list_dir")
    assert list_dir.parameters is not None
    props = list_dir.parameters["properties"]
    assert "path" in props
    # path has a default ("") so it must NOT appear in required
    assert "path" not in list_dir.parameters.get("required", [])


def test_filesystem_tool_search_schema() -> None:
    tools = FilesystemTool()._expand_namespace()
    search = next(t for t in tools if t.name == "FilesystemTool_search")
    assert search.parameters is not None
    props = search.parameters["properties"]
    assert "pattern" in props
    assert search.parameters.get("required") == ["pattern"]


def test_filesystem_tool_serializes_to_openai_format() -> None:
    tools = FilesystemTool()._expand_namespace()
    for tool in tools:
        dumped = tool.model_dump()
        assert dumped["type"] == "function"
        assert "name" in dumped["function"]
        assert "parameters" in dumped["function"]
        assert dumped["function"]["parameters"]["type"] == "object"


@pytest.mark.unit
async def test_filesystem_tool_ainvoke_write_and_read(fs_tool: FilesystemTool) -> None:
    tools = {t.name: t for t in fs_tool._expand_namespace()}

    write_result = await tools["FilesystemTool_write_file"].ainvoke(
        path="hello.txt", content="integration test"
    )
    assert "hello.txt" in write_result

    read_result = await tools["FilesystemTool_read_file"].ainvoke(path="hello.txt")
    assert read_result == "integration test"


@pytest.mark.unit
async def test_filesystem_tool_ainvoke_list_dir(
    workspace: Path, fs_tool: FilesystemTool
) -> None:
    (workspace / "a.py").touch()
    (workspace / "b.py").touch()
    tools = {t.name: t for t in fs_tool._expand_namespace()}
    result = await tools["FilesystemTool_list_dir"].ainvoke(path="")
    assert "a.py" in result and "b.py" in result


def test_shell_tool_is_a_namespace_tool() -> None:
    assert ShellTool().is_namespace is True


def test_shell_tool_expands_to_run_tool() -> None:
    tools = ShellTool()._expand_namespace()
    names = {t.name for t in tools}
    assert {"ShellTool_run"}.issubset(names)


def test_shell_tool_run_schema_has_command_as_only_required_parameter() -> None:
    tools = ShellTool()._expand_namespace()
    run = next(t for t in tools if t.name == "ShellTool_run")
    assert run.parameters is not None
    props = run.parameters["properties"]
    assert "command" in props
    assert props["command"]["type"] == "string"
    assert run.parameters.get("required") == ["command"]


def test_shell_tool_serializes_to_openai_format() -> None:
    tools = ShellTool()._expand_namespace()
    run = next(t for t in tools if t.name == "ShellTool_run")
    dumped = run.model_dump()
    assert dumped["type"] == "function"
    assert dumped["function"]["name"] == "ShellTool_run"
    assert "parameters" in dumped["function"]


@pytest.mark.unit
async def test_shell_tool_ainvoke_run(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path))
    tools = {t.name: t for t in tool._expand_namespace()}
    result = await tools["ShellTool_run"].ainvoke(command="echo integration")
    assert result["stdout"].strip() == "integration"
    assert result["returncode"] == 0


@pytest.mark.unit
async def test_shell_tool_ainvoke_blocked_command_returns_error(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path), blocked_commands=[r"\bsudo\b"])
    tools = {t.name: t for t in tool._expand_namespace()}
    result = await tools["ShellTool_run"].ainvoke(command="sudo echo hi")
    assert result["returncode"] == -1
    assert "blocked" in result["stderr"].lower()
