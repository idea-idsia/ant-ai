from __future__ import annotations

from pathlib import Path

import pytest

from ant_ai.tools.builtins.shell_tool import ShellTool


@pytest.fixture()
def bash_tool(tmp_path: Path) -> ShellTool:
    return ShellTool(cwd=str(tmp_path))


def test_run_returns_stdout(bash_tool: ShellTool) -> None:
    result = bash_tool.run("echo hello")
    assert result["stdout"].strip() == "hello"
    assert result["returncode"] == 0


def test_run_returns_stderr(bash_tool: ShellTool) -> None:
    result = bash_tool.run("echo error >&2")
    assert "error" in result["stderr"]


def test_run_returns_nonzero_returncode_on_failure(bash_tool: ShellTool) -> None:
    result = bash_tool.run("exit 42")
    assert result["returncode"] == 42


def test_run_result_has_expected_keys(bash_tool: ShellTool) -> None:
    result = bash_tool.run("echo hello")
    assert set(result.keys()) == {"stdout", "stderr", "returncode"}


def test_run_timeout_returns_error(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path), timeout=1)
    result = tool.run("sleep 10")
    assert result["returncode"] == -1
    assert "timed out" in result["stderr"]


def test_run_invalid_cwd_returns_error() -> None:
    result = ShellTool(cwd="/nonexistent_path_xyz").run("echo hi")
    assert result["returncode"] == -1


def test_blocked_command_is_rejected(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path), blocked_commands=[r"\bsudo\b"])
    result = tool.run("sudo apt install curl")
    assert result["returncode"] == -1
    assert "blocked" in result["stderr"].lower()


def test_blocked_command_non_matching_runs_normally(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path), blocked_commands=[r"\bsudo\b"])
    result = tool.run("echo hello")
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"


def test_no_blocked_commands_runs_normally(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path))
    result = tool.run("echo hello")
    assert "blocked" not in result["stderr"].lower()


@pytest.mark.parametrize(
    "pattern,command",
    [
        (r"\bsudo\b", "sudo rm -rf /"),
        (r"\bdd\b", "dd if=/dev/zero of=/dev/sda"),
        (r"rm\s+-rf", "rm -rf /"),
    ],
)
def test_blocked_patterns_are_matched(
    tmp_path: Path, pattern: str, command: str
) -> None:
    tool = ShellTool(cwd=str(tmp_path), blocked_commands=[pattern])
    result = tool.run(command)
    assert result["returncode"] == -1
    assert "blocked" in result["stderr"].lower()


def test_allowed_command_passes_through(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path), allowed_commands=["echo"])
    result = tool.run("echo hello")
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"


def test_command_not_in_allowlist_is_rejected(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path), allowed_commands=["echo"])
    result = tool.run("ls /tmp")
    assert result["returncode"] == -1
    assert "allowed" in result["stderr"].lower()


def test_no_allowed_commands_permits_all(tmp_path: Path) -> None:
    tool = ShellTool(cwd=str(tmp_path))
    result = tool.run("ls /tmp")
    assert "allowed" not in result["stderr"].lower()


def test_blocked_takes_priority_over_allowed(tmp_path: Path) -> None:
    tool = ShellTool(
        cwd=str(tmp_path),
        allowed_commands=["sudo"],
        blocked_commands=[r"\bsudo\b"],
    )
    result = tool.run("sudo echo hi")
    assert result["returncode"] == -1
    assert "blocked" in result["stderr"].lower()
