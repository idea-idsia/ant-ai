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
        substitution (`$(...)`, backticks) is refused. Patterns are searched, not
        matched in full: anchor them (e.g. `^ls\\b`) to match the command name only.
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
        if self.allowed_commands:
            allowed: Pattern[str] = re.compile(
                "|".join(f"(?:{p})" for p in self.allowed_commands)
            )
            parts: list[str] = [
                p.strip() for p in _COMMAND_SEPARATOR.split(command) if p.strip()
            ]
            if (
                _COMMAND_SUBSTITUTION.search(command)
                or not parts
                or not all(allowed.search(part) for part in parts)
            ):
                return {
                    "stdout": "",
                    "stderr": "Command not allowed by policy",
                    "returncode": -1,
                }

        if self.blocked_commands:
            blocked: Pattern[str] = re.compile(
                "|".join(f"(?:{p})" for p in self.blocked_commands)
            )
            if blocked.search(command):
                return {
                    "stdout": "",
                    "stderr": "Command is blocked by policy",
                    "returncode": -1,
                }

        try:
            result: CompletedProcess[str] = subprocess.run(
                command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=True,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout} seconds",
                "returncode": -1,
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": f"Working directory not found: {self.cwd}",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }
