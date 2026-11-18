from __future__ import annotations

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.status import Status
from rich.table import Table
from rich.text import Text

from ant_ai.core.events import (
    BaseEvent,
    ClarificationNeededEvent,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ReasoningEvent,
    ToolCallingEvent,
    ToolResultEvent,
)
from ant_ai.core.message import ToolCall

# The tool that `Agent(planning=True)` adds: its calls are shown as a checklist.
PLAN_TOOL = "PlanningTool_write_todos"

# The ant-ai logo (docs/assets/ant_icon_dark.png) in braille, 16 columns wide.
LOGO = """\
           ⢀⠞⠉⠉⠉
          ⣴⡟⣷⣦⣀
  ⢀⣀⢀  ⢀⣤⣼⡿⠿⠟⠛⠉
⣠⣾⣿⠏⣮⢻⣷⢿⣿⣿⡟ ⡀
⠉⠻⠟⠼⠟⠃⢁⡾⠈⠉⠿⠛⢳⡀
 ⣰⠏   ⡾⠁    ⠈⢷⡀"""

ACCENT = "#6c5a8e"  # muted dark violet
ARG_CHARS = 50
ANSWER_CHARS = 70
PLAN_MARKS: dict[str, tuple[str, str]] = {
    "completed": ("✓", "dim"),
    "in_progress": ("▸", f"bold {ACCENT}"),
    "pending": ("○", ""),
}


def model_label(model: str) -> str:
    """`anthropic/claude-sonnet-5-5` -> `claude-sonnet-5-5 · anthropic`, easier to read."""
    provider, _, name = model.rpartition("/")
    return f"{name} · {provider}" if provider else name


def _count(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else plural or word + 's'}"


def main_argument(call: ToolCall) -> str:
    """The argument that says what a call is about: a path, a command, a pattern."""
    try:
        args: dict[str, Any] = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return call.function.arguments or ""
    for key in ("path", "command", "pattern", "task"):
        if isinstance(args.get(key), str):
            value: str = args[key]
            break
    else:
        value = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
    value = value.splitlines()[0] if value else ""
    return value if len(value) <= ARG_CHARS else value[: ARG_CHARS - 1] + "…"


def result_summary(name: str, content: str) -> tuple[str, bool]:
    """A few words on a tool result, and whether it went well."""
    try:
        data: Any = json.loads(content)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and "returncode" in data:
        out = (data.get("stdout") or "") + (data.get("stderr") or "")
        last = next(
            (line.strip() for line in reversed(out.splitlines()) if line.strip()), ""
        )
        code = data["returncode"]
        return f"exit {code}" + (f" · {last[:60]}" if last else ""), code == 0
    if isinstance(data, list):
        return _count(len(data), "entry", "entries"), True
    if name == "write_file":
        return "written", True
    lines = content.splitlines()
    if len(lines) == 1 and len(lines[0]) <= ANSWER_CHARS:
        return lines[0], True
    return _count(len(lines), "line"), True


class Renderer:
    """Prints a run compactly: the model's short reason for each step, one line
    per tool call with its result, then the answer.

    With `verbose`, the model's reasoning and the first lines of each tool
    result are printed too.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        spinner: bool = True,
        verbose: bool = False,
    ):
        self.console = console or Console(highlight=False)
        self.spinner = spinner
        self.verbose = verbose
        self._status: Status | None = None
        self._pending: dict[str, ToolCall] = {}
        self._under_reason = False
        self.steps = 0
        self.tool_calls = 0
        self._started = 0.0

    def banner(self, model: str, workspace: str) -> None:
        info = Text()
        info.append("ant-ai\n", style=f"bold {ACCENT}")
        info.append("model ", style="dim")
        info.append(f"{model_label(model)}\n", style="bold")
        info.append(workspace, style="dim")
        grid = Table.grid(padding=(0, 3))
        grid.add_row(Text(LOGO, style=ACCENT), Padding(info, (2, 0, 0, 0)))
        self.console.print(grid)
        self.console.print()

    def user(self, text: str) -> None:
        self.console.print(Text(f"› {text}", style="bold"))
        self.console.print()

    def note(self, text: str) -> None:
        self.console.print(Text(text, style="dim"))

    def partial(self, text: str) -> None:
        """A piece of text still being written; printed once complete."""

    @contextmanager
    def turn(self) -> Generator[None]:
        """Wrap one user turn: spinner while waiting, summary at the end."""
        self.steps = self.tool_calls = 0
        self._pending.clear()
        self._started = time.monotonic()
        if self.spinner:
            self._status = self.console.status("thinking", spinner="dots")
            self._status.start()
        try:
            yield
        finally:
            if self._status is not None:
                self._status.stop()
                self._status = None
            elapsed = time.monotonic() - self._started
            self.note(
                f"{_count(self.steps, 'step')} · {_count(self.tool_calls, 'tool')} · {elapsed:.1f}s"
            )
            self.console.print()

    @contextmanager
    def paused(self) -> Generator[None]:
        """Stop the spinner while the user answers a question."""
        if self._status is not None:
            self._status.stop()
        try:
            yield
        finally:
            if self._status is not None:
                self._status.start()

    def event(self, event: BaseEvent) -> None:
        match event:
            case ReasoningEvent() if self.verbose:
                for line in event.content.strip().splitlines():
                    self.console.print(Text(f"  {line}", style="dim italic"))
            case ToolCallingEvent():
                self.steps += 1
                reason = event.content.strip()
                self._under_reason = bool(reason)
                if reason:
                    self._bullet(Text(reason))
                for call in event.tool_calls:
                    self.tool_calls += 1
                    self._pending[call.id] = call
                self._working()
            case ToolResultEvent():
                self._result(event)
            case FinalAnswerEvent():
                self.steps += 1
                self.console.print()
                self.console.print(Markdown(event.content or "(no answer)"))
            case MaxStepsReachedEvent():
                self.console.print(Text("stopped: step limit reached", style="yellow"))
            case ClarificationNeededEvent():
                self.console.print(Text(f"? {event.content}", style="bold yellow"))

    def _bullet(self, text: Text) -> None:
        """A top-level line: a dot, then text wrapped under itself."""
        self._row(Text("● ", style=ACCENT), text)

    def _row(self, mark: Text, text: Text) -> None:
        """`mark` then `text`, with wrapped lines aligned under the text."""
        grid = Table.grid()
        grid.add_column(width=len(mark.plain), no_wrap=True)
        grid.add_column()
        grid.add_row(mark, text)
        self.console.print(grid)

    def _working(self) -> None:
        if self._status is None:
            return
        if self._pending:
            names = ", ".join(c.function.name for c in self._pending.values())
            self._status.update(f"running {names}")
        else:
            self._status.update("thinking")

    def _result(self, event: ToolResultEvent) -> None:
        call = self._pending.pop(event.tool_call_id, None)
        line = Text()
        line.append("plan" if event.name == PLAN_TOOL else event.name, style="bold")
        if call is not None and event.name != PLAN_TOOL:
            line.append(f"  {main_argument(call)}")
        if event.is_error:
            message = event.content.removeprefix("ERROR: ").splitlines()[0]
            line.append(f"  ✗ {message}", style="red")
        elif event.name != PLAN_TOOL:
            summary, ok = result_summary(event.name, event.content)
            line.append(f"  {summary}", style="dim" if ok else "yellow")
        if self._under_reason:
            self._row(Text("  ⎿ ", style="dim"), line)
        else:
            self._bullet(line)
        if event.name == PLAN_TOOL and call is not None and not event.is_error:
            self._plan(call)
        elif self.verbose and not event.is_error:
            for preview in event.content.splitlines()[:4]:
                self.console.print(Text(f"      {preview[:100]}", style="dim"))
        self._working()

    def _plan(self, call: ToolCall) -> None:
        """Print the plan written by the planning tool as a checklist."""
        try:
            todos = json.loads(call.function.arguments or "{}").get("todos", [])
        except json.JSONDecodeError:
            return
        for todo in todos:
            status = todo.get("status", "pending")
            mark, style = PLAN_MARKS.get(status, PLAN_MARKS["pending"])
            self.console.print(
                Text(f"      {mark} {todo.get('content', '')}", style=style)
            )
