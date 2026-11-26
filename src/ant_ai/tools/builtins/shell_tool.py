from __future__ import annotations

import re
import subprocess
from re import Pattern
from subprocess import CompletedProcess

from ant_ai.tools.tool import Tool

# `;`, `&&`, `||`, `|`, `|&`, newline and `&`, but not the `&` of redirections like `2>&1`.
_COMMAND_SEPARATOR: Pattern[str] = re.compile(r"&&|\|\||\|&?|;|\n|(?<![<>])&(?!>)")
_COMMAND_SUBSTITUTION: Pattern[str] = re.compile(r"\$\(|`|<\(|>\(")


class ShellTool(Tool):
    """
    Execute shell commands in the agent workspace.

    Example:
        ```python
        from ant_ai.agent import Agent
        from ant_ai.tools.builtins.shell_tool import ShellTool

        shell = ShellTool(
            cwd="/workspace",
            timeout=30,
            blocked_commands=["rm\\s+-rf\\b", "shutdown", "reboot"],
        )
        agent = Agent(tools=[shell], ...)
        ```

        The agent can now call:

            - ShellTool_run(command="python3 main.py")
            - ShellTool_run(command="pytest tests/ -q")

    Notes:
        Commands are filtered before execution against `allowed_commands` (allowlist)
        and `blocked_commands` (denylist), both expressed as regex patterns.
        With an allowlist, a chained command (`;`, `&&`, `||`, `|`, `&`, newline) runs
        only if every command in the chain matches a pattern, and command
        substitution (`$(...)`, backticks) is refused. Each command must match a
        pattern from its start (`re.match`): end the pattern with `\\b` (e.g.
        `ls\\b`) so that it does not also allow longer names (`lsof`).
        Policy violations, timeouts, and runtime errors are returned as a dict
        with a non-zero `returncode` and the reason in `stderr`, so the agent
        receives a recoverable error rather than an exception.
    """

    cwd: str = "/workspace"
    timeout: int = 60
    allowed_commands: list[str] | None = None
    blocked_commands: list[str] | None = None

    def run(self, command: str) -> dict:
        """Execute a shell command in the workspace. Prefer running scripts by filename (e.g. `python3 hello.py`) over passing file contents inline.

        Args:
            command: The shell command to execute.

        Returns:
            A dict with keys: stdout, stderr, returncode.
        """
        violation: str | None = self._policy_violation(command)
        if violation is not None:
            return _failure(violation)
        return self._execute(command)

    def _policy_violation(self, command: str) -> str | None:
        """Return why `command` is refused by the allow/block lists, or None if it may run."""
        if self.allowed_commands and not _is_allowed(command, self.allowed_commands):
            return "Command not allowed by policy"
        if self.blocked_commands and _any_pattern(self.blocked_commands).search(
            command
        ):
            return "Command is blocked by policy"
        return None

    def _execute(self, command: str) -> dict:
        """Run `command` in `cwd`, turning failures into a non-zero result."""
        try:
            result: CompletedProcess[str] = subprocess.run(
                command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=True,
            )
        except subprocess.TimeoutExpired:
            return _failure(f"Command timed out after {self.timeout} seconds")
        except FileNotFoundError:
            return _failure(f"Working directory not found: {self.cwd}")
        except Exception as e:
            return _failure(str(e))
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }


def _any_pattern(patterns: list[str]) -> Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


def _is_allowed(command: str, patterns: list[str]) -> bool:
    """Whether every command in the chain matches a pattern, with no command substitution."""
    allowed: Pattern[str] = _any_pattern(patterns)
    parts: list[str] = [
        p.strip() for p in _COMMAND_SEPARATOR.split(command) if p.strip()
    ]
    return (
        not _COMMAND_SUBSTITUTION.search(command)
        and bool(parts)
        and all(allowed.match(part) for part in parts)
    )


def _failure(stderr: str) -> dict:
    return {"stdout": "", "stderr": stderr, "returncode": -1}
