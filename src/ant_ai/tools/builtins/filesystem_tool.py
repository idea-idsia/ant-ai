from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from re import Pattern

import pathspec
from pydantic import PrivateAttr

from ant_ai.tools.tool import Tool, ToolError

# The tool calls of one step run in parallel: edits and writes take turns, so an
# edit never writes back a file that another call changed after it was read.
_WRITE_LOCK: threading.Lock = threading.Lock()


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
            - FilesystemTool_read_file(path="big.log", offset=201, limit=50)
            - FilesystemTool_write_file(path="output.txt", content="hello")
            - FilesystemTool_edit_file(path="app.py", old_string="a - b", new_string="a + b")
            - FilesystemTool_list_dir(path="src/")
            - FilesystemTool_search(pattern="TODO", path="src/")

    Notes:
        All paths are resolved relative to `workspace_root` and sandboxed within it.
        Failures -- a missing file, an invalid regex, an attempt to escape the
        workspace (e.g. "../etc/passwd", "/absolute/path") -- raise `ToolError`,
        which `ToolStep` turns into a recoverable `ERROR: ...` message for the
        agent and marks as an error.
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

    def read_file(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        """Read a file. Path is relative to /workspace. Without `offset` and `limit` the whole file is returned; for a long file, pass `offset` (first line, counting from 1 as `search` does) and `limit` (number of lines) to read only that part."""
        if offset < 1:
            raise ToolError("offset must be 1 or more")
        if limit is not None and limit < 1:
            raise ToolError("limit must be 1 or more")
        text: str = self._read(path)
        if offset == 1 and limit is None:
            return text
        lines: list[str] = text.splitlines(keepends=True)
        if offset > max(len(lines), 1):
            raise ToolError(
                f"offset {offset} is past the end of {path} ({len(lines)} lines)"
            )
        start: int = offset - 1
        end: int | None = None if limit is None else start + limit
        return "".join(lines[start:end])

    def edit_file(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace `old_string` with `new_string` in a file, leaving the rest unchanged. Path is relative to /workspace. `old_string` must appear exactly once, so include enough surrounding text to make it unique; set `replace_all` to replace every occurrence."""
        if not old_string:
            raise ToolError("old_string is empty")
        if old_string == new_string:
            raise ToolError("old_string and new_string are the same")
        with _WRITE_LOCK:
            # newline="": keep the file's own line endings on the way in and out.
            text: str = self._read(path, newline="")
            # The model writes "\n" even for a file with "\r\n": match either.
            file_eol: str = "\r\n" if "\r\n" in text else "\n"
            old_lines: list[str] = old_string.replace("\r\n", "\n").split("\n")
            pattern: Pattern[str] = re.compile(r"\r?\n".join(map(re.escape, old_lines)))
            new_lines: list[str] = new_string.replace("\r\n", "\n").split("\n")

            def replace(match: re.Match[str]) -> str:
                found: str = match.group()
                eol: str = (
                    "\r\n" if "\r\n" in found else "\n" if "\n" in found else file_eol
                )
                return eol.join(new_lines)

            count: int = len(pattern.findall(text))
            if count == 0:
                raise ToolError(f"old_string not found in {path}")
            if count > 1 and not replace_all:
                raise ToolError(
                    f"old_string appears {count} times in {path}: add surrounding text to make it unique, or set replace_all"
                )
            try:
                self._resolve(path).write_text(
                    pattern.sub(replace, text, count=0 if replace_all else 1),
                    encoding="utf-8",
                    newline="",
                )
            except OSError as e:
                raise ToolError(f"{e.strerror or e}: {path}") from e
        replaced: int = count if replace_all else 1
        return f"Replaced {replaced} occurrence{'' if replaced == 1 else 's'} in {path}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file, overwriting if it exists. Creates parent directories as needed. Path is relative to /workspace."""
        try:
            target: Path = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with _WRITE_LOCK:
                target.write_text(content, encoding="utf-8")
            return f"Written {len(content.encode('utf-8'))} bytes to {path}"
        except ValueError as e:
            raise ToolError(str(e)) from e
        except OSError as e:
            raise ToolError(f"{e.strerror or e}: {path}") from e

    def list_dir(self, path: str = ".", depth: int = 4) -> list[str]:
        """List files and directories recursively at the given path, up to `depth` levels deep. Path is relative to /workspace. Defaults to workspace root."""
        try:
            target: Path = self._resolve(path)
            results: list[str] = []
            self._walk(target, target, depth, results)
            return sorted(results)
        except ValueError as e:
            raise ToolError(str(e)) from e
        except FileNotFoundError as e:
            raise ToolError(f"directory not found: {path}") from e
        except OSError as e:
            raise ToolError(f"{e.strerror or e}: {path}") from e

    def _walk(self, base: Path, current: Path, depth: int, results: list[str]) -> None:
        """Collect paths (relative to `base`) of entries under `current`, recursing up to `depth` levels."""
        for entry in current.iterdir():
            if not self._is_inside(entry):
                continue
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
                    if (
                        not file.is_file()
                        or not self._is_inside(file)
                        or self._is_ignored(file, False)
                    ):
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
            raise ToolError(str(e)) from e
        except re.error as e:
            raise ToolError(f"invalid regex pattern: {e}") from e

    def _read(self, path: str, newline: str | None = None) -> str:
        """Read a workspace file as text, turning failures into `ToolError`."""
        try:
            return self._resolve(path).read_text(encoding="utf-8", newline=newline)
        except ValueError as e:
            raise ToolError(str(e)) from e
        except FileNotFoundError as e:
            raise ToolError(f"file not found: {path}") from e
        except OSError as e:
            raise ToolError(f"{e.strerror or e}: {path}") from e

    def _resolve(self, path: str) -> Path:
        """Resolve path relative to base, ensuring it stays within base."""
        resolved: Path = (self.workspace_root / path).resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError(f"Path '{path}' escapes the workspace boundary")
        return resolved

    def _is_inside(self, entry: Path) -> bool:
        """Whether `entry`, with symlinks followed, stays within the workspace."""
        return entry.resolve().is_relative_to(self.workspace_root)
