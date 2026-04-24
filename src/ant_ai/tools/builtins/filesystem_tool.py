from __future__ import annotations

import re
from pathlib import Path
from re import Pattern

from ant_ai.tools.tool import Tool


class FilesystemTool(Tool):
    """
    Tools for reading and writing files within the agent workspace.

    Example:
        ```python
        from ant_ai.agent import Agent
        from ant_ai.tools.builtins.filesystem_tool import FilesystemTool

        fs = FilesystemTool(workspace_root="/workspace")
        agent = Agent(tools=[fs], ...)
        ```

        The agent can now call:

            - FilesystemTool_read_file(path="notes.txt")
            - FilesystemTool_write_file(path="output.txt", content="hello")
            - FilesystemTool_list_dir(path="src/")
            - FilesystemTool_search(pattern="TODO", path="src/")

    Notes:
        All paths are resolved relative to `workspace_root` and sandboxed within it.
        Attempts to escape the workspace (e.g. "../etc/passwd", "/absolute/path")
        are caught by `_resolve` and returned to the caller as an error string
        instead of raising — so the agent receives a recoverable error message
        rather than an exception.
    """

    workspace_root: Path = Path()

    def __init__(self, workspace_root: str = "/workspace", **data):
        super().__init__(**data)
        self.workspace_root = Path(workspace_root).resolve()

    def read_file(self, path: str) -> str:
        """Read the full contents of a file. Path is relative to /workspace."""
        try:
            return self._resolve(path).read_text(encoding="utf-8")
        except ValueError as e:
            return f"Error: {e}"
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except Exception as e:
            return f"Error: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file, overwriting if it exists. Creates parent directories as needed. Path is relative to /workspace."""
        try:
            target: Path = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Written {len(content)} bytes to {path}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def list_dir(self, path: str = "") -> list[str]:
        """List files and directories at the given path. Path is relative to /workspace. Defaults to workspace root."""
        try:
            target: Path = self._resolve(path) if path else self.workspace_root
            return [entry.name for entry in sorted(target.iterdir())]
        except ValueError as e:
            return [f"Error: {e}"]
        except FileNotFoundError:
            return [f"Error: directory not found: {path}"]
        except Exception as e:
            return [f"Error: {e}"]

    def search(self, pattern: str, path: str = "") -> str:
        """Search for a regex pattern recursively within path. Path is relative to /workspace. Returns matching lines in the format 'file:lineno: content'."""
        try:
            root: Path = self._resolve(path) if path else self.workspace_root
            regex: Pattern[str] = re.compile(pattern)
            results: list[str] = []
            for file in sorted(root.rglob("*")):
                if not file.is_file():
                    continue
                try:
                    for lineno, line in enumerate(
                        file.read_text(encoding="utf-8", errors="replace").splitlines(),
                        start=1,
                    ):
                        if regex.search(line):
                            rel = file.relative_to(self.workspace_root)
                            results.append(f"{rel}:{lineno}: {line}")
                except Exception:
                    continue
            return "\n".join(results) if results else "No matches found"
        except ValueError as e:
            return f"Error: {e}"
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"
        except Exception as e:
            return f"Error: {e}"

    def _resolve(self, path: str) -> Path:
        """Resolve path relative to base, ensuring it stays within base."""
        resolved = (self.workspace_root / path).resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError(f"Path '{path}' escapes the workspace boundary")
        return resolved
