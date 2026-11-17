"""`ant --tui`: the ACP client of `ant` as a Textual app.

The conversation sits on the left, with every tool call folded on one line.
The panel on the right follows the run: the agents and their state, the plan,
the messages between the agents, and how much work each step did.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode
from textual.worker import Worker

from ant_ai.cli.acp_client import RemoteSession
from ant_ai.cli.render import PLAN_MARKS, PLAN_TOOL, main_argument, result_summary
from ant_ai.core.events import (
    BaseEvent,
    ClarificationNeededEvent,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ToolCallingEvent,
    ToolResultEvent,
)
from ant_ai.core.message import ToolCall

RESULT_LINES = 40
RESULT_CHARS = 4000
LINE_CHARS = 300

CSS = """
Screen { background: #17181c; color: #d4d4d8; }
Header { background: #22242a; color: #d4d4d8; }
Footer { background: #22242a; }
#main { height: 1fr; }
#left { width: 3fr; padding: 0 1; }
#side { width: 1.4fr; min-width: 50; }
#top { height: 11; }
#conversation { height: 1fr; scrollbar-size-vertical: 1; }
#statusbar { height: 1; margin: 1 0 0 1; }
#status { width: 1fr; }
#walker { width: auto; }
.live { margin: 1 0 0 0; color: #a9adb6; }
#ask { margin: 0; border: round #6c5a8e; background: #17181c; }
.panel { border: round #3a3d45; border-title-color: #a99bc9; border-title-style: bold; padding: 0 1; background: #1b1c21; }
#agents, #activity { height: 100%; }
#agents { width: 5fr; }
#activity { width: 4fr; }
#plan { height: auto; max-height: 10; }
#messages { height: 1fr; }
#stats, #tools { height: auto; }
#context { height: 5; }
.user { text-style: bold; color: #ffffff; margin: 1 0 0 0; }
.reason { margin: 1 0 0 0; }
.note { color: #8a8f99; }
.answer { margin: 1 0 0 0; }
.summary { color: #8a8f99; margin: 1 0 0 0; }
Collapsible { border: none; padding: 0; margin: 0 0 0 2; background: #17181c; }
Collapsible > Contents { padding: 0 0 0 3; }
CollapsibleTitle { padding: 0; color: #c4c4cc; }
.result { color: #9a9fa8; }
.error CollapsibleTitle { color: #e07a6f; }
DataTable { background: #1b1c21; height: 1fr; overflow-x: hidden; }
Tree { background: #1b1c21; overflow-x: hidden; }
PermissionScreen { align: center middle; background: #17181ccc; }
#permission { width: 64; height: auto; border: round #6c5a8e; background: #22242a; padding: 1 2; }
#permission Label { width: 100%; margin: 0 0 1 0; }
#permission Horizontal { height: auto; }
#permission Button { margin: 0 1 0 0; }
"""


def _now() -> str:
    return datetime.now().strftime("%H:%M")


def _arguments(call: ToolCall) -> dict[str, Any]:
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def _is_agent(call: ToolCall) -> bool:
    """Whether a tool call goes to another agent.

    An agent used as a tool, local (`as_tool()`) or remote (`A2AAgentTool`),
    takes a single `message`: a guess until the events say who an agent is.
    """
    return set(_arguments(call)) == {"message"}


def _readable(content: str) -> str:
    """A command result as the terminal would show it, other results as they are."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(data, dict) and "returncode" in data:
        out = (data.get("stdout") or "") + (data.get("stderr") or "")
        return f"{out.rstrip()}\n(exit {data['returncode']})"
    return content


def _shown(content: str) -> str:
    """The start of a result, small enough to lay out: a list one item per
    line, at most `RESULT_LINES` lines of `LINE_CHARS`, `RESULT_CHARS` in all."""
    text = _readable(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        text = "\n".join(str(item) for item in data)
    lines = text.splitlines()
    shown: list[str] = []
    size = 0
    for line in lines[:RESULT_LINES]:
        line = line if len(line) <= LINE_CHARS else line[: LINE_CHARS - 1] + "…"
        if size + len(line) > RESULT_CHARS:
            break
        shown.append(line)
        size += len(line) + 1
    rest = len(lines) - len(shown)
    if rest > 0:
        shown.append(f"… {rest} more lines ({len(text):,} characters in all)")
    return "\n".join(shown)


# The ground the ant walks on, repeated.
CRUMBS = "·  · ·   ·  ·    · ·  "

# The ant, facing right, in block octants (2x4 pixels per character); the two
# frames move its legs. Drawn from:
#   .........#
#   .##..#..#.
#   ###.###.##
#   ...#.#.#..   or   ....#.#.#.
ANT = (
    "\U0001cd2f\U0001cd84\U0001cda4\U0001cd80\U0001cd2d",
    "\U0001cd2f\U0001fbe6\U0001cd69\u2596\U0001cd67",
)


@dataclass
class ToolStats:
    """The calls of one tool in a turn."""

    calls: int = 0
    failed: int = 0


def _bar(text: Text, share: float, width: int, *, full: str, empty: str) -> None:
    filled = round(min(1.0, share) * width)
    text.append(full * filled, style="#a99bc9")
    text.append(empty * (width - filled), style="#3a3d45")


class PermissionScreen(ModalScreen[str]):
    """Asks the user to allow an action; returns `y`, `n` or `a`."""

    BINDINGS = [
        Binding("y", "choose('y')", "allow"),
        Binding("n", "choose('n')", "reject"),
        Binding("a", "choose('a')", "always"),
        Binding("escape", "choose('n')", "reject", show=False),
    ]

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="permission"):
            yield Label(Text(self.question, style="bold"))
            with Horizontal():
                yield Button("Allow (y)", id="y", variant="primary")
                yield Button("Reject (n)", id="n")
                yield Button("Always (a)", id="a")

    def action_choose(self, answer: str) -> None:
        self.dismiss(answer)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id or "n")


class AntApp(App[None]):
    """The conversation with one ACP agent, and a panel that follows its run."""

    TITLE = "ant-ai"
    CSS = CSS
    BINDINGS = [
        Binding("escape", "stop", "stop the turn"),
        Binding("ctrl+l", "clear", "new conversation"),
        Binding("ctrl+o", "toggle_all", "open/close tools"),
        Binding("ctrl+q", "quit", "quit"),
    ]

    def __init__(
        self,
        command: list[str],
        workspace: Path,
        *,
        label: str,
        context_size: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.view = TuiView(self, context_size=context_size, max_steps=max_steps)
        self.session = RemoteSession(
            command,
            workspace,
            label=label,
            renderer=self.view,
            read_answer=self.ask_permission,
        )
        shown = str(self.session.workspace).replace(str(Path.home()), "~", 1)
        self.sub_title = f"{label} · {shown} · ACP"
        self._running_turn: Worker[None] | None = None
        self._queued: list[str] = []
        self._connected = asyncio.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield VerticalScroll(id="conversation")
                with Horizontal(id="statusbar"):
                    yield Static(id="status")
                    yield Static(id="walker")
                yield Input(placeholder="ask ant to do something", id="ask")
            with Vertical(id="side"):
                with Horizontal(id="top"):
                    agents: Tree[str] = Tree("agents", id="agents", classes="panel")
                    agents.show_root = False
                    agents.border_title = "AGENTS"
                    yield agents
                    with Vertical(id="activity", classes="panel") as activity:
                        activity.border_title = "ACTIVITY"
                        yield Static(id="stats")
                        yield Static(id="tools")
                plan = Static(
                    Text("no plan yet", style="#8a8f99"), id="plan", classes="panel"
                )
                plan.border_title = "PLAN"
                yield plan
                table: DataTable[str] = DataTable(
                    id="messages",
                    classes="panel",
                    cursor_type="row",
                    zebra_stripes=True,
                )
                table.border_title = "MESSAGES"
                yield table
                context = Static(id="context", classes="panel")
                context.border_title = "CONTEXT"
                yield context
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#messages", DataTable)
        table.add_column("time", width=5)
        table.add_column("from", width=8)
        table.add_column("to", width=8)
        table.add_column("what", width=40)
        self.query_one("#stats", Static).update(Text("connecting...", style="#8a8f99"))
        self.view.show_status()
        self.set_interval(0.12, self.view.tick)
        self.run_worker(self._connect(), group="connection")

    async def _connect(self) -> None:
        try:
            async with self.session.connect():
                self.view.reset_agents(self.session.agent_name or "agent")
                self.view.reset_context()
                self.view.stats()
                self._connected.set()
                self.query_one("#ask", Input).focus()
                await asyncio.Event().wait()
        except (ConnectionError, OSError) as e:
            lines = "\n".join(self.session.stderr)
            self.view.note(f"the agent stopped: {e}\n{lines}")

    async def ask_permission(self, question: str) -> str:
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.push_screen(
            PermissionScreen(question),
            callback=lambda choice: answer.set_result(choice or "n"),
        )
        before = self.view.current
        self.view.doing("waiting for you")
        try:
            return await answer
        finally:
            self.view.doing(before)

    def _busy(self) -> bool:
        return self._running_turn is not None and self._running_turn.is_running

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
        if self._busy():
            # Sent when the turn ends.
            self._queued.append(text)
            self.view.note(f"queued: {text}")
            return
        await self._connected.wait()
        self._running_turn = self.run_worker(self._turns(text), group="turn")

    async def _turns(self, text: str) -> None:
        while True:
            self.view.user(text)
            await self.session.ask(text)
            if not self._queued:
                return
            text = self._queued.pop(0)

    def action_stop(self) -> None:
        turn = self._running_turn
        if turn is not None and turn.is_running:
            turn.cancel()
            # The queued messages go back to the input, to send or change.
            if self._queued:
                self.query_one("#ask", Input).value = " ".join(self._queued)
                self._queued.clear()

    async def action_clear(self) -> None:
        if self._busy():
            return
        await self.session.clear()
        await self.query_one("#conversation").remove_children()
        self.view.reset_agents(self.session.agent_name or "agent")
        self.view.reset_context()
        self.view.note("new conversation")

    def action_toggle_all(self) -> None:
        tools = list(self.query(Collapsible))
        open_ = any(c.collapsed for c in tools)
        for c in tools:
            c.collapsed = not open_


class TuiView:
    """What `RemoteSession` draws on: the events of the run, shown in the app."""

    def __init__(
        self,
        app: AntApp,
        *,
        context_size: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.app = app
        self.context_size = context_size
        self.max_steps = max_steps
        self._context_chars = 0
        self.verbose = False
        # Each running call: its box, its result and when it started.
        self._calls: dict[str, tuple[ToolCall, Collapsible, Static, float]] = {}
        self._agent_nodes: dict[str, TreeNode[str]] = {}
        self._root: TreeNode[str] | None = None
        self._agent = "agent"
        self._started = 0.0
        self.steps = 0
        self.tool_calls = 0
        self._by_tool: dict[str, ToolStats] = {}
        self._live: Static | None = None
        self._live_text = ""
        self._doing = ""
        self._frame = 0

    # -- what RemoteSession calls

    @contextmanager
    def turn(self) -> Generator[None]:
        self.steps = self.tool_calls = 0
        self._by_tool = {}
        self._started = time.monotonic()
        self._set_state(self._root, "working")
        self.doing("thinking")
        try:
            yield
        finally:
            self._end_live()
            self._set_state(self._root, "waiting")
            self.doing("")
            self.stats()

    @contextmanager
    def paused(self) -> Generator[None]:
        # The time spent waiting for the user is not time of the running tools.
        asked = time.monotonic()
        try:
            yield
        finally:
            waited = time.monotonic() - asked
            for key, (call, box, result, started) in self._calls.items():
                self._calls[key] = (call, box, result, started + waited)

    def note(self, text: str) -> None:
        self._mount(Static(Text(text), classes="note"))

    def partial(self, text: str) -> None:
        """Show the text of the model while it is written, until its step or answer arrives."""
        self._live_text += text
        if self._live is None:
            self._live = Static(classes="live")
            self._mount(self._live)
        self._live.update(Text(self._live_text))
        self.app.query_one("#conversation", VerticalScroll).scroll_end(animate=False)
        self.doing("writing")

    def _end_live(self) -> None:
        if self._live is not None:
            self._live.remove()
        self._live, self._live_text = None, ""

    # -- the walking ant

    @property
    def current(self) -> str:
        return self._doing

    def doing(self, what: str) -> None:
        self._doing = what
        self.show_status()

    def tick(self) -> None:
        # The timer can fire once more while the app closes, after its widgets are gone.
        if self._doing and self.app.query("#status"):
            self._frame += 1
            self.show_status()
            self.stats()

    def show_status(self) -> None:
        line = Text("■ ", style="#a99bc9")
        # The ant stays still at the right and the crumbs come to it, as if the
        # view followed it walking; one step every two ticks.
        step = self._frame // 2
        ground = CRUMBS * 2
        walker = Text(ANT[step % 2] + " ", style="#a99bc9")
        if not self._doing:
            line.append("ready", style="#8a8f99")
            walker.append(" " * len(CRUMBS))
        else:
            elapsed = time.monotonic() - self._started if self._started else 0.0
            line.append(self._doing, style="#d4d4d8")
            line.append(f"  {elapsed:.1f}s", style="#8a8f99")
            at = step % len(CRUMBS)
            walker.append(ground[at : at + len(CRUMBS)], style="#6b707a")
        self.app.query_one("#status", Static).update(line)
        self.app.query_one("#walker", Static).update(walker)

    def event(self, event: BaseEvent) -> None:
        if isinstance(
            event,
            ToolCallingEvent
            | FinalAnswerEvent
            | ClarificationNeededEvent
            | MaxStepsReachedEvent,
        ):
            self._end_live()
        match event:
            case ToolCallingEvent():
                self._step(event)
            case ToolResultEvent():
                self._result(event)
            case FinalAnswerEvent():
                self.steps += 1
                self._mount(Markdown(event.content or "(no answer)", classes="answer"))
                self._seen(event.content)
                self._message(self._agent, "user", "answer", event.content)
            case MaxStepsReachedEvent():
                self.note("stopped: step limit reached")
            case ClarificationNeededEvent():
                self._mount(Static(Text(f"? {event.content}", style="bold #e0b36f")))
                self._message(self._agent, "user", "question", event.content)
        self.stats()

    # -- the app's own calls

    def user(self, text: str) -> None:
        self._mount(Static(Text(f"› {text}"), classes="user"))
        self._seen(text)
        self._message("user", self._agent, "task", text)

    def reset_agents(self, name: str) -> None:
        self._agent = name
        tree = self.app.query_one("#agents", Tree)
        tree.clear()
        self._agent_nodes.clear()
        self._root = tree.root.add(name, expand=True)
        self._root.data = name
        self._set_state(self._root, "waiting")

    def reset_context(self) -> None:
        self._context_chars = 0
        self._seen("")

    def _seen(self, text: str) -> None:
        """Count what went into the conversation, to estimate how full the context is.

        An estimate: about 4 characters per token, and only what passes over
        ACP (not the system prompt nor the tool definitions). The exact count
        needs the agent to send ACP `usage_update`.
        """
        self._context_chars += len(text or "")
        used = self._context_chars // 4
        text_ = Text()
        text_.append(f"≈ {used / 1000:.1f}k", style="bold #ffffff")
        if self.context_size:
            share = min(1.0, used / self.context_size)
            text_.append(f" / {self.context_size // 1000}k tokens  ", style="#8a8f99")
            text_.append(f"{share:.0%}\n", style="bold #ffffff")
            _bar(text_, share, self._width("#context", 30), full="█", empty="░")
        else:
            text_.append(" tokens (model limit unknown)", style="#8a8f99")
        text_.append("\nestimate: what passed over ACP", style="#6b707a")
        self.app.query_one("#context", Static).update(text_)

    def stats(self) -> None:
        elapsed = time.monotonic() - self._started if self._started else 0.0
        text = Text()
        text.append(f"{self.tool_calls}", style="bold #ffffff")
        text.append(" tools  ", style="#8a8f99")
        text.append(f"{elapsed:.1f}", style="bold #ffffff")
        text.append("s", style="#8a8f99")
        self.app.query_one("#stats", Static).update(text)
        self.app.query_one("#tools", Static).update(self._rows())

    def _width(self, selector: str, default: int) -> int:
        """The width inside a widget, once it is laid out."""
        return self.app.query_one(selector).content_size.width or default

    def _rows(self) -> Text:
        """The steps with a bar over the limit, then the calls of each tool, in columns."""
        width = self._width("#tools", 24)
        text = Text("steps ", style="#d4d4d8")
        text.append(f"{self.steps}", style="bold #ffffff")
        if self.max_steps:
            text.append(f"/{self.max_steps}", style="#8a8f99")
            text.append("\n")
            _bar(text, self.steps / self.max_steps, width, full="━", empty="─")
        cells = []
        for name, tool in sorted(self._by_tool.items(), key=lambda kv: -kv[1].calls):
            cell = Text(f"{name} ", style="#d4d4d8")
            cell.append(f"{tool.calls}", style="bold #ffffff")
            if tool.failed:
                cell.append(f" ✗{tool.failed}", style="bold #e07a6f")
            cells.append(cell)
        if not cells:
            return text
        cell_width = max(c.cell_len for c in cells) + 2
        columns = max(1, width // cell_width)
        for i, cell in enumerate(cells):
            text.append("\n" if i % columns == 0 else "")
            if i % columns < columns - 1 and i < len(cells) - 1:
                cell.pad_right(cell_width - cell.cell_len)
            text.append_text(cell)
        return text

    # -- drawing

    def _mount(self, widget: Any) -> None:
        box = self.app.query_one("#conversation", VerticalScroll)
        box.mount(widget)
        box.scroll_end(animate=False)

    def _step(self, event: ToolCallingEvent) -> None:
        self.steps += 1
        self._seen(event.content)
        for call in event.tool_calls:
            self._seen(call.function.arguments or "")
        if event.content.strip():
            self._mount(
                Static(
                    Text.assemble(("● ", "#a99bc9"), event.content.strip()),
                    classes="reason",
                )
            )
        names = ", ".join(c.function.name for c in event.tool_calls)
        self.doing(f"running {names}")
        for call in event.tool_calls:
            self.tool_calls += 1
            name = call.function.name
            self._by_tool.setdefault(name, ToolStats()).calls += 1
            if name == PLAN_TOOL:
                self._plan(call)
            title = f"{name}  {main_argument(call)}  …"
            # Kept by reference: the result can arrive before the box is mounted.
            result = Static("", classes="result")
            box = Collapsible(result, title=title, collapsed=True)
            self._calls[call.id] = (call, box, result, time.monotonic())
            self._mount(box)
            if _is_agent(call):
                node = (self._root or self.app.query_one("#agents", Tree).root).add(
                    name, expand=True
                )
                node.data = name
                self._agent_nodes[call.id] = node
                self._set_state(node, "working")
                task = next(iter(_arguments(call).values()), "")
                self._message(self._agent, name, "task", str(task))

    def _result(self, event: ToolResultEvent) -> None:
        found = self._calls.pop(event.tool_call_id, None)
        if found is None:
            return
        call, box, result, started = found
        took = time.monotonic() - started
        if not self._calls:
            self.doing("thinking")
        name = event.name
        tool = self._by_tool.setdefault(name, ToolStats())
        if event.is_error:
            tool.failed += 1
            summary = "✗ " + event.content.removeprefix("ERROR: ").splitlines()[0]
            box.add_class("error")
        elif name == PLAN_TOOL:
            summary = "plan updated"
        else:
            summary, _ = result_summary(name, event.content)
        box.title = f"{name}  {main_argument(call)}  ·  {summary}  ·  {took:.1f}s"
        self._seen(event.content)
        shown = _shown(event.content)
        result.update(Text(shown))
        node = self._agent_nodes.pop(event.tool_call_id, None)
        if node is not None:
            self._set_state(node, "failed" if event.is_error else "done")
            self._message(name, self._agent, "answer", event.content)

    def _plan(self, call: ToolCall) -> None:
        todos = _arguments(call).get("todos", [])
        text = Text()
        for todo in todos if isinstance(todos, list) else []:
            status = todo.get("status", "pending")
            mark, style = PLAN_MARKS.get(status, PLAN_MARKS["pending"])
            text.append(f"{mark} {todo.get('content', '')}\n", style=style or "#d4d4d8")
        self.app.query_one("#plan", Static).update(text or Text("no plan yet"))

    def _message(self, sender: str, to: str, what: str, content: str) -> None:
        table = self.app.query_one("#messages", DataTable)
        first = (
            (content or "").strip().splitlines()[0] if (content or "").strip() else ""
        )
        table.add_row(_now(), sender[:8], to[:8], f"{what}: {first}"[:40])
        table.move_cursor(row=table.row_count - 1)

    def _set_state(self, node: TreeNode[str] | None, state: str) -> None:
        if node is None:
            return
        marks = {
            "working": ("●", "#a99bc9"),
            "waiting": ("○", "#8a8f99"),
            "done": ("✓", "#7fbf95"),
            "failed": ("✗", "#e07a6f"),
        }
        mark, style = marks[state]
        # The mark first: in a narrow panel only the state word gets cut.
        node.set_label(
            Text.assemble(
                (f"{mark} ", style),
                (node.data or "", "bold"),
                (f" {state}", "#8a8f99"),
            )
        )


def context_size(model: str | None) -> int | None:
    """The context window of the model, when LiteLLM knows it."""
    if not model:
        return None
    try:
        import litellm

        return litellm.get_model_info(model).get("max_input_tokens")
    except Exception:
        return None


def run(
    command: list[str],
    workspace: Path,
    *,
    label: str,
    model: str | None = None,
    max_steps: int | None = None,
) -> None:
    AntApp(
        command,
        workspace,
        label=label,
        context_size=context_size(model),
        max_steps=max_steps,
    ).run()
