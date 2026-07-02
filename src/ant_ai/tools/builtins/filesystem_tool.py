from __future__ import annotations

import os
import re
from pathlib import Path
from re import Pattern

import pathspec
from pydantic import PrivateAttr

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
    _ignore_spec: pathspec.PathSpec | None = PrivateAttr(default=None)

    def __init__(self, workspace_root: str = "/workspace", **data):
        super().__init__(**data)
        self.workspace_root = Path(workspace_root).resolve()
        self._ignore_spec = self._load_ignore_spec()

    def _load_ignore_spec(self) -> pathspec.PathSpec | None:
        gitignore: Path = self.workspace_root / ".gitignore"
        if not gitignore.is_file():
            return None
        lines: list[str] = gitignore.read_text(encoding="utf-8").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)

    def _is_ignored(self, entry: Path, is_dir: bool) -> bool:
        if self._ignore_spec is None:
            return False
        rel: str = entry.relative_to(self.workspace_root).as_posix()
        return self._ignore_spec.match_file(f"{rel}/" if is_dir else rel)

    def read_file(self, path: str) -> str:
        """Read the full contents of a file. Path is relative to /workspace."""
        try:
            return self._resolve(path).read_text(encoding="utf-8")
        except ValueError as e:
            return f"Error: {e}"
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except OSError as e:
            return f"Error: {e.strerror or e}: {path}"
        except Exception as e:
            return f"Error: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file, overwriting if it exists. Creates parent directories as needed. Path is relative to /workspace."""
        try:
            target: Path = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Written {len(content.encode('utf-8'))} bytes to {path}"
        except ValueError as e:
            return f"Error: {e}"
        except OSError as e:
            return f"Error: {e.strerror or e}: {path}"
        except Exception as e:
            return f"Error: {e}"

    def list_dir(self, path: str = ".", depth: int = 4) -> list[str]:
        """List files and directories recursively at the given path, up to `depth` levels deep. Path is relative to /workspace. Defaults to workspace root."""
        try:
            target: Path = self._resolve(path)
            results: list[str] = []
            self._walk(target, target, depth, results)
            return sorted(results)
        except ValueError as e:
            return [f"Error: {e}"]
        except FileNotFoundError:
            return [f"Error: directory not found: {path}"]
        except OSError as e:
            return [f"Error: {e.strerror or e}: {path}"]
        except Exception as e:
            return [f"Error: {e}"]

    def _walk(self, base: Path, current: Path, depth: int, results: list[str]) -> None:
        """Collect paths (relative to `base`) of entries under `current`, recursing up to `depth` levels."""
        for entry in current.iterdir():
            is_dir = entry.is_dir()
            if self._is_ignored(entry, is_dir):
                continue
            rel = str(entry.relative_to(base))
            results.append(f"{rel}/" if is_dir else rel)
            if is_dir and depth > 0:
                self._walk(base, entry, depth - 1, results)

    def search(self, pattern: str, path: str = ".", depth: int = 4) -> str:
        """Search for a regex pattern recursively within path, up to `depth` levels deep. Path is relative to /workspace. Returns matching lines in the format 'file:lineno: content'."""
        try:
            root: Path = self._resolve(path)
            regex: Pattern[str] = re.compile(pattern)
            results: list[str] = []
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames.sort()
                dirnames[:] = [
                    d for d in dirnames if not self._is_ignored(Path(dirpath, d), True)
                ]
                if len(Path(dirpath).relative_to(root).parts) >= depth:
                    dirnames.clear()
                for filename in sorted(filenames):
                    file: Path = Path(dirpath) / filename
                    if not file.is_file() or self._is_ignored(file, False):
                        continue
                    try:
                        for lineno, line in enumerate(
                            file.read_text(
                                encoding="utf-8", errors="replace"
                            ).splitlines(),
                            start=1,
                        ):
                            if regex.search(line):
                                rel: Path = file.relative_to(self.workspace_root)
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
        resolved: Path = (self.workspace_root / path).resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError(f"Path '{path}' escapes the workspace boundary")
        return resolved
