"""`ant --tui`: the ACP client of `ant` as a Textual app.

The conversation takes the whole width, with every tool call folded on one line.
The line above the input follows the run. The agents, the tools used and the
messages between agents are in a panel that only Ctrl+T opens and closes: at
the right of the conversation on a wide screen, above it on a narrow one.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from rich.cells import cell_len, set_cell_size
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
from ant_ai.cli.render import (
    LOGO,
    PLAN_MARKS,
    PLAN_TOOL,
    arguments,
    error_line,
    main_argument,
    result_summary,
    shell_output,
    todos,
)
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

RESULT_LINES = 40
RESULT_CHARS = 4000
LINE_CHARS = 300
SIDE_FROM = 120
# The plan items shown in the column at the right; a longer plan shows the
# ones around the current item.
PLAN_ROWS = 8
# The room of the state word: the longest ("connecting", "asking you") and
# three spaces.
STATE_CHARS = 13

CSS = """
Screen { background: #17181c; color: #d4d4d8; }
Header { background: #22242a; color: #d4d4d8; }
Footer { background: #22242a; }
#main { height: 1fr; padding: 0 1; }
#panel { display: none; width: 100%; height: 3; margin: 0; }
#panel.-open { display: block; }
#body { width: 100%; height: 1fr; }
Screen.-side #panel { display: block; dock: right; width: 32; height: 100%; layout: vertical; margin: 0; background: #1b1c21; }
Screen.-side #panel.-shut { display: none; }
#side-plan { display: none; }
Screen.-side #side-plan { display: block; width: 100%; height: auto; max-height: 12; }
#upper { width: 100%; height: 1fr; }
Screen.-side #agents, Screen.-side #tools { width: 100%; height: auto; max-height: 10; }
Screen.-side #messages, Screen.-side #no-messages { width: 100%; height: 1fr; }
.welcome { width: 100%; height: 100%; content-align: center middle; color: #a99bc9; }
#conversation { width: 100%; height: 1fr; scrollbar-size-vertical: 1; }
#status { width: 100%; height: 1; margin: 1 0 0 0; padding: 0 0 0 1; }
#ask { margin: 0; border: round #6c5a8e; background: #1b1c21; }
#ask:focus { background-tint: transparent; }
.panel { border: round #3a3d45; border-title-color: #a99bc9; border-title-style: bold; padding: 0 1; background: #1b1c21; }
#agents, #tools, #messages { height: 100%; }
.plan { margin: 0 0 1 2; }
Screen.-side .plan { display: none; }
Screen.-side.-panel-shut .plan { display: block; }
#agents { width: 1fr; }
#tools { width: 1fr; scrollbar-size-vertical: 1; }
#messages { display: none; scrollbar-size-vertical: 1; scrollbar-gutter: stable; }
#no-messages { height: 100%; color: #8a8f99; }
#messages, #no-messages { width: 2fr; }
DataTable:blur > .datatable--cursor { background: transparent; color: #d4d4d8; text-style: none; }
DataTable > .datatable--even-row, DataTable > .datatable--odd-row { color: #d4d4d8; }
.user { text-style: bold; color: #ffffff; margin: 1 0 1 0; }
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
PermissionScreen { align: left bottom; background: transparent; }
#permission { width: 1fr; height: auto; margin: 0 1 1 1; border: round #6c5a8e; background: #17181c; padding: 0 1; }
#permission Label { width: 100%; }
#permission Horizontal { height: auto; }
#permission Button { margin: 0 1 0 0; }
MessageScreen { align: center middle; background: #17181ccc; }
#message { width: 90; max-height: 80%; height: auto; border: round #3a3d45; border-title-color: #a99bc9; background: #22242a; padding: 1 2; }
#message VerticalScroll { height: auto; max-height: 100%; }
"""


def _is_agent(call: ToolCall) -> bool:
    """Whether a tool call goes to another agent.

    An agent used as a tool, local (`as_tool()`) or remote (`A2AAgentTool`),
    takes a single `message`: a guess until the events say who an agent is.
    """
    return set(arguments(call)) == {"message"}


def _call_title(call: ToolCall) -> str:
    """The tool and what the call is about; the plan by its number of items."""
    if call.function.name == PLAN_TOOL:
        items = len(todos(call))
        return f"plan  {items} item{'' if items == 1 else 's'}"
    return f"{call.function.name}  {main_argument(call)}"


def _readable(content: str) -> str:
    """A command result as the terminal would show it, other results as they are."""
    try:
        data = json.loads(content)
    except ValueError:
        return content
    if (shell := shell_output(data)) is None:
        return content
    stdout, stderr, code = shell
    return f"{(stdout + stderr).rstrip()}\n(exit {code})"


def _shown(content: str) -> str:
    """The start of a result, small enough to lay out: a list one item per
    line, at most `RESULT_LINES` lines of `LINE_CHARS`, `RESULT_CHARS` in all."""
    text = _readable(content)
    try:
        data = json.loads(text)
    except ValueError:
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


def _markdown(text: str) -> Markdown:
    """Text of the model; without linkify, which reads `hello.py` as a web address."""
    return Markdown(
        text,
        classes="answer",
        parser_factory=lambda: MarkdownIt("gfm-like", {"linkify": False}),
    )


@dataclass
class ToolStats:
    """The calls of one tool in a turn."""

    calls: int = 0
    failed: int = 0


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
        # The keys are on the buttons: the ` y/n/a` of the terminal is dropped.
        self.question = question.removesuffix(" y/n/a")

    def compose(self) -> ComposeResult:
        with Vertical(id="permission"):
            yield Label(Text(self.question, style="bold"))
            with Horizontal():
                yield Button("Allow (y)", id="y", variant="primary", compact=True)
                yield Button("Reject (n)", id="n", compact=True)
                yield Button(
                    "Always (a)",
                    id="a",
                    compact=True,
                    tooltip="every action of this kind, for this session",
                )

    def action_choose(self, answer: str) -> None:
        self.dismiss(answer)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id or "n")


class MessageScreen(ModalScreen[None]):
    """One message between the agents, in full."""

    BINDINGS = [Binding("escape,enter,q", "dismiss", "close")]

    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self.title_ = title
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="message") as box:
            box.border_title = self.title_
            with VerticalScroll():
                yield Static(Text(self.text), id="message-text")

    async def action_dismiss(self, result: None = None) -> None:
        self.dismiss(result)


class MessagesTable(DataTable[str]):
    """The messages, cut again to the width each time it changes."""

    def on_resize(self) -> None:
        app = self.app
        if isinstance(app, AntApp):
            app.call_after_refresh(app.view.fill_messages)


class AntApp(App[None]):
    """The conversation with one ACP agent, and what follows its run below it."""

    TITLE = "ant-ai"
    CSS = CSS
    # From 120 columns the panel is a column at the right, where the
    # conversation leaves room; below, it is a strip on top.
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (SIDE_FROM, "-side")]
    BINDINGS = [
        Binding("escape", "stop", "stop"),
        Binding("ctrl+l", "clear", "new conversation"),
        # With priority: it works while a permission question is open too.
        Binding("ctrl+t", "toggle_panel", "panel", priority=True),
        Binding("ctrl+o", "toggle_all", "details"),
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
        task: str | None = None,
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
        # Messages that wait for the agent: to start, or to end its turn.
        self._queued: list[str] = [task] if task else []
        self._connected = asyncio.Event()
        # Set once the agent process is gone: nothing more can be sent.
        self._stopped = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main"):
            with Vertical(id="upper"):
                with Horizontal(id="panel"):
                    agents: Tree[str] = Tree("agents", id="agents", classes="panel")
                    agents.show_root = False
                    agents.guide_depth = 2
                    agents.border_title = "AGENTS"
                    yield agents
                    # Under the agents: the plan of the one at work.
                    plan = Static(id="side-plan", classes="panel")
                    plan.border_title = "PLAN"
                    yield plan
                    with VerticalScroll(id="tools", classes="panel") as tools:
                        tools.border_title = "TOOLS"
                        yield Static(Text("none yet", style="#8a8f99"), id="used")
                    table = MessagesTable(
                        id="messages",
                        classes="panel",
                        cursor_type="row",
                        show_header=False,
                        cell_padding=0,
                    )
                    table.border_title = "MESSAGES"
                    yield table
                    empty = Static("none yet", id="no-messages", classes="panel")
                    empty.border_title = "MESSAGES"
                    yield empty
                with Vertical(id="body"), VerticalScroll(id="conversation"):
                    yield Static(Text(LOGO), classes="welcome")
                # Under the conversation, so the column reaches the input.
                yield Static(id="status")
            yield Input(placeholder="ask ant to do something", id="ask")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#messages", DataTable)
        table.add_column("message", key="message")
        self.view._show_side_plan()
        self.view.reset_context()
        self.view.doing("connecting")
        # Keys typed while the agent starts are kept: the message waits for it.
        self.query_one("#ask", Input).focus()
        self.set_interval(0.12, self.view.tick)
        self.run_worker(self._connect(), group="connection")

    async def _connect(self) -> None:
        try:
            async with self.session.connect():
                self.view.reset_agents(self.session.agent_name or "agent")
                self.view.reset_context()
                self.view.stats()
                self.view.doing("")
                self._connected.set()
                self.query_one("#ask", Input).focus()
                if self._queued:
                    self._start(self._queued.pop(0))
                await asyncio.Event().wait()
        except (ConnectionError, OSError) as e:
            self._agent_stopped(e)

    def _agent_stopped(self, error: BaseException) -> None:
        self._stopped = True
        if not self.is_running:
            # The app is closing and takes the agent down with it: nothing to
            # tell, and the screen is already gone.
            return
        if self.view.current == "connecting":
            self.view.doing("")
        lines = "\n".join(self.session.stderr)
        self.view.note(f"the agent stopped: {error}\n{lines}".rstrip())

    async def ask_permission(self, question: str) -> str:
        """Show the question in a dialog and wait for the user's choice.

        Args:
            question: The action to allow, e.g. "allow run `pytest`?".

        Returns:
            `y`, `n` or `a`; Esc counts as `n`.
        """
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def chosen(choice: str | None) -> None:
            if not answer.done():
                answer.set_result(choice or "n")

        screen = PermissionScreen(question)
        self.push_screen(screen, callback=chosen)
        before = self.view.current
        self.view.doing("asking you")
        try:
            return await answer
        except asyncio.CancelledError:
            # The agent withdrew the request: the question goes away with it.
            if screen.is_current:
                screen.dismiss(None)
            raise
        finally:
            self.view.doing(before)

    def _busy(self) -> bool:
        return self._running_turn is not None and self._running_turn.is_running

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
        if self._stopped:
            self.view.note("the agent is not running: restart ant")
            return
        if self._busy() or not self._connected.is_set():
            # Sent when the turn ends, or once the agent has started.
            self._queued.append(text)
            self.view.note(f"queued: {text}")
            return
        self._start(text)

    def _start(self, text: str) -> None:
        self._running_turn = self.run_worker(self._turns(text), group="turn")

    async def _turns(self, text: str) -> None:
        while True:
            self.view.user(text)
            try:
                await self.session.ask(text)
            except Exception as e:
                # The agent process exited or broke the protocol mid-turn.
                self._agent_stopped(e)
                return
            if not self._queued:
                return
            text = self._queued.pop(0)

    def action_stop(self) -> None:
        turn = self._running_turn
        # A second Esc while the turn stops would cut the stop short.
        if turn is not None and turn.is_running and not turn.is_cancelled:
            turn.cancel()
            # The queued messages go back to the input, to send or change,
            # before what the user is typing.
            if self._queued:
                box = self.query_one("#ask", Input)
                box.value = " ".join([*self._queued, box.value]).strip()
                self._queued.clear()

    async def action_clear(self) -> None:
        if self._busy() or self._stopped or not self._connected.is_set():
            return
        await self.session.clear()
        await self.query_one("#conversation").remove_children()
        self.view.reset(self.session.agent_name or "agent")
        self.view.note("new conversation")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        sender, to, what, text, when = self.view.messages[event.row_key.value or ""]
        self.push_screen(MessageScreen(f"{sender} → {to} · {what} · {when}", text))

    def action_toggle_panel(self) -> None:
        # The main screen, also when a question is open on top of it.
        main = self.screen_stack[0]
        panel = main.query_one("#panel")
        # One switch for both places: what is shown now gets hidden, and it
        # stays so when the window crosses from narrow to wide.
        shown = bool(panel.display)
        panel.set_class(shown, "-shut")
        panel.set_class(not shown, "-open")
        # The plan is in the column at the right, or in the conversation.
        main.set_class(panel.has_class("-shut"), "-panel-shut")
        self.view.show_status()
        if not shown:
            self.view.fit_panel(opening=True)
        focused = main.focused
        if (
            focused is not None
            and panel.display is False
            and panel in focused.ancestors
        ):
            main.query_one("#ask", Input).focus()
        self.call_after_refresh(self.view.follow_end)

    def on_resize(self) -> None:
        # Once the screen has its breakpoint class for the new width.
        self.call_after_refresh(self.view.fit_panel, opening=True)

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
        self._used_tokens = 0
        self._context_text = Text("")
        self.verbose = False
        # Each running call: its box, its result and when it started.
        self._calls: dict[str, tuple[ToolCall, Collapsible, Static, float]] = {}
        self._agent_nodes: dict[str, TreeNode[str]] = {}
        self._root: TreeNode[str] | None = None
        self._agent = "agent"
        self._started = self._took = 0.0
        self._plan_block: Static | None = None
        self._plan_done = ""
        self.steps = 0
        self.tool_calls = 0
        self._by_tool: dict[str, ToolStats] = {}
        # Each row of the messages panel: sender, receiver, task or answer, text.
        # Sender, receiver, kind, text and time of every message: the user's
        # tasks, the answers, and what the agents send each other.
        self.messages: dict[str, tuple[str, str, str, str, str]] = {}
        self._panel_rows = 1
        # Whether the panel is the column at the right of the conversation.
        self._side = False
        self._live: Markdown | None = None
        self._drawn = ""
        self._live_text = ""
        # The reasoning of the model arrives in chunks, shown in one folded box.
        self._thought: Static | None = None
        self._thought_text = ""
        self._doing = ""
        self._frame = 0
        self._follow = self._anchored = False
        # The tools of the whole conversation, for the panel.
        self._used: dict[str, ToolStats] = {}

    # -- what RemoteSession calls

    @contextmanager
    def turn(self) -> Generator[None]:
        self.steps, self.tool_calls = 1, 0
        self._plan_block, self._plan_done = None, ""
        self._show_side_plan()
        self._by_tool = {}
        self._started = time.monotonic()
        self._set_state(self._root, "working")
        self.doing("thinking")
        try:
            yield
        finally:
            self._end_live()
            self._close_open_calls()
            self._set_state(self._root, "waiting")
            self._took = time.monotonic() - self._started
            self.doing("")

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
        if self._live is None and self._live_text.strip():
            self._drawn = self._live_text
            self._live = _markdown(self._drawn)
            self._mount(self._live)
        self.doing("writing")

    def _draw_live(self) -> None:
        if self._live is not None and self._drawn != self._live_text:
            new = self._live_text[len(self._drawn) :]
            self._drawn = self._live_text
            # On the widget's own queue: after its mount, and in order.
            self._live.call_after_refresh(self._live.append, new)

    def _end_live(self) -> bool:
        """Close the text being written, which stays; whether there was one."""
        shown = self._live is not None
        self._draw_live()
        self._live, self._live_text, self._drawn = None, "", ""
        return shown

    # -- the walking ant

    @property
    def current(self) -> str:
        return self._doing

    def doing(self, what: str) -> None:
        self._doing = what
        self.show_status()

    def follow_end(self) -> None:
        """Keep the end in view, only while the conversation fills the screen."""
        # An anchored box puts a short conversation at the bottom and moves it
        # up at every new line.
        box = self.app.query_one("#conversation", VerticalScroll)
        if box.max_scroll_y <= 0:
            if self._anchored:
                self._anchored, self._follow = False, True
                box.anchor(False)
                # Forced: a box with nothing to scroll ignores scroll_home, and an
                # anchored one can be left scrolled below zero.
                box.scroll_home(animate=False, immediate=True, force=True)
        elif self._follow:
            self._follow, self._anchored = False, True
            # Follows what is written, until the user scrolls up.
            box.anchor()

    def tick(self) -> None:
        # The timer can fire once more while the app closes, after its widgets are gone.
        if not self.app.query("#status"):
            return
        self.follow_end()
        if self._doing:
            self._frame += 1
            self._draw_live()
            self.show_status()

    def show_status(self) -> None:
        # The ant at the start of the line moves its legs while the agent
        # works, one step every two ticks, and stands still when it is done.
        legs = ANT[(self._frame // 2) % 2] if self._doing else ANT[0]
        line = Text(legs + " ", style="#a99bc9")
        seconds = self._took
        # What ants do when they look for food: the word while the model thinks.
        state = {"thinking": "foraging", "": "ready"}.get(self._doing, self._doing)
        line.append(
            f"{_cut(state, STATE_CHARS - 1):<{STATE_CHARS}}",
            style="#d4d4d8" if self._doing else "#8a8f99",
        )
        line.append_text(self._context_text)
        if not self.app.screen_stack:
            # The app is closing, e.g. quit while a question was open.
            return
        # The plan first: on a narrow screen the counters are cut, not the plan.
        # In the column at the right while it is there.
        main = self.app.screen_stack[0]
        column = self._side and not main.has_class("-panel-shut")
        if self._plan_done and not column:
            line.append(self._plan_done, style="#a99bc9")
            line.append(" · " if self.steps else "", style="#8a8f99")
        parts = []
        if self.steps:
            if self._doing:
                parts.append(f"step {self.steps}")
                seconds = time.monotonic() - self._started
            else:
                parts.append(f"{self.steps} step{'' if self.steps == 1 else 's'}")
            parts.append(f"{seconds:.1f}s")
            parts.append(f"{self.tool_calls} tool{'' if self.tool_calls == 1 else 's'}")
        line.append(" · ".join(parts), style="#8a8f99")
        if failed := sum(tool.failed for tool in self._by_tool.values()):
            line.append(f" ✗{failed}", style="bold #e07a6f")
        line.no_wrap = True
        line.overflow = "ellipsis"
        self.app.query_one("#status", Static).update(line)

    def event(self, event: BaseEvent) -> None:
        written = False
        if isinstance(
            event,
            ToolCallingEvent
            | FinalAnswerEvent
            | ClarificationNeededEvent
            | MaxStepsReachedEvent,
        ):
            written = self._end_live()
        if not isinstance(event, ReasoningEvent):
            self._thought, self._thought_text = None, ""
        match event:
            case ReasoningEvent():
                self._reasoning(event.content)
            case ToolCallingEvent():
                self._step(event, written)
            case ToolResultEvent():
                self._result(event)
            case FinalAnswerEvent():
                if not written:
                    self._mount(_markdown(event.content or "(no answer)"))
                self._seen(event.content)
                self._message(self._agent, "you", "answer", event.content)
            case MaxStepsReachedEvent():
                self.note("stopped: step limit reached")
            case ClarificationNeededEvent():
                self._mount(Static(Text(f"? {event.content}", style="bold #e0b36f")))
        self.stats()

    # -- the app's own calls

    def user(self, text: str) -> None:
        self._follow = True
        self._mount(Static(Text(f"› {text}"), classes="user"))
        self.follow_end()
        self._seen(text)
        self._message("you", self._agent, "task", text)

    def reset_agents(self, name: str) -> None:
        self._agent = name
        tree = self.app.query_one("#agents", Tree)
        tree.clear()
        self._agent_nodes.clear()
        self._root = tree.root.add(name, expand=True, allow_expand=False)
        self._root.data = name
        self._set_state(self._root, "waiting")

    def fit_panel(self, opening: bool = False) -> None:
        """Above the conversation: as tall as its longest part, up to a quarter
        of the screen.

        While open it only grows, until a new conversation: shrinking would move
        the conversation under the reader. At the right it is as tall as the
        conversation.
        """
        main = self.app.screen_stack[0]
        panel = main.query_one("#panel")
        # From the width: on a resize the breakpoint class comes later.
        if (self.app.size.width >= SIDE_FROM) != self._side:
            self._side = not self._side
            self.fill_messages()
            self._show_context()
            self.show_status()
        if self._side:
            # A quarter of the screen: wider on a wide screen, so the messages
            # are not cut for nothing.
            panel.styles.width = max(32, self.app.size.width // 4)
            panel.styles.height = None
            return
        panel.styles.width = None
        lines = (1 + len(self._agent_nodes), len(self._used), len(self.messages))
        cap = max(1, self.app.size.height // 4 - 2)
        rows = min(max(1, *lines), cap)
        self._panel_rows = rows if opening else max(self._panel_rows, rows)
        main.query_one("#panel").styles.height = self._panel_rows + 2

    def reset(self, name: str) -> None:
        """Empty every panel, for a new conversation."""
        self.reset_agents(name)
        self.reset_context()
        self._calls.clear()
        self.steps = self.tool_calls = 0
        self._by_tool = {}
        self._started = self._took = 0.0
        self._plan_block, self._plan_done = None, ""
        self._show_side_plan()
        self._follow = self._anchored = False
        self.app.query_one("#conversation", VerticalScroll).anchor(False)
        self._used.clear()
        self._show_used()
        self.messages.clear()
        self.app.query_one("#messages", DataTable).clear()
        self.app.query_one("#messages").display = False
        self.app.query_one("#no-messages").display = True
        # A new conversation starts empty: nothing moves if the panel shrinks.
        self.fit_panel(opening=True)
        self.stats()

    def _close_open_calls(self) -> None:
        """Mark the calls that never got a result, once the turn is over."""
        for call_id, (call, box, _, _) in self._calls.items():
            box.title = f"{_call_title(call)}  ·  stopped"
            self._set_state(self._agent_nodes.pop(call_id, None), "stopped")
        self._calls.clear()

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
        self._used_tokens = self._context_chars // 4
        self._show_context()

    def _show_context(self) -> None:
        """The share, a bar and the tokens, after the state in the status line.

        Always as wide, so the counters after it do not move; short on a
        narrow screen.
        """
        tokens = _tokens(self._used_tokens)
        if not self.context_size:
            text = Text.assemble(
                ("context " if self._side else "", "#8a8f99"),
                (f"≈{tokens}", "#d4d4d8"),
            )
            # "context ≈999.9k", the longest count before it reads in M.
            self._set_context(text, 15 if self._side else 7)
            return
        share = min(1.0, self._used_tokens / self.context_size)
        style = "#e07a6f" if share >= 0.9 else "#e0b36f" if share >= 0.7 else "#a99bc9"
        shown = f"{share:.0%}" if share >= 0.01 else "<1%"
        total = _tokens(self.context_size, exact=False)
        if self._side:
            head = Text.assemble(("context ", "#8a8f99"), (f"{shown:>3} ", style))
            tail = Text(f" {tokens}/{total}", style="#8a8f99")
            width = 8
        else:
            head, tail, width = Text(f"{shown:>3} ", style=style), Text(""), 5
        filled = max(1 if self._used_tokens else 0, round(share * width))
        bar = Text.assemble(("█" * filled, style), ("░" * (width - filled), "#5a5e68"))
        # The longest the line gets with this total: 100% and 999.9k used.
        longest = len("context 100% ") + width + len(f" 999.9k/{total}")
        self._set_context(head + bar + tail, longest if self._side else 10)

    def _set_context(self, text: Text, longest: int) -> None:
        text.append(" " * max(2, longest + 2 - len(text.plain)))
        self._context_text = text
        self.show_status()

    def stats(self) -> None:
        self.show_status()

    def _show_used(self) -> None:
        text = Text(no_wrap=True, overflow="ellipsis")
        for name, tool in self._used.items():
            text.append(f"{name:<14} {tool.calls}", style="#d4d4d8")
            if tool.failed:
                text.append(f" ✗{tool.failed}", style="bold #e07a6f")
            text.append("\n")
        text.rstrip()
        if not self._used:
            text.append("none yet", style="#8a8f99")
        self.app.query_one("#used", Static).update(text)
        self.fit_panel()

    # -- drawing

    def _mount(self, widget: Any) -> None:
        box = self.app.query_one("#conversation", VerticalScroll)
        box.query(".welcome").remove()
        box.mount(widget)

    def _step(self, event: ToolCallingEvent, written: bool) -> None:
        self._seen(event.content)
        for call in event.tool_calls:
            self._seen(call.function.arguments or "")
        if event.content.strip() and not written:
            self._mount(
                Static(
                    Text.assemble(("● ", "#a99bc9"), event.content.strip()),
                    classes="reason",
                )
            )
        self.doing("running")
        for call in event.tool_calls:
            name = call.function.name
            # The plan is a block and an agent is in AGENTS: neither is a tool.
            is_tool = name != PLAN_TOOL and not _is_agent(call)
            self.tool_calls += is_tool
            self._by_tool.setdefault(name, ToolStats()).calls += 1
            if is_tool:
                self._used.setdefault(name, ToolStats()).calls += 1
                self._show_used()
            if name == PLAN_TOOL:
                self._plan(call)
            title = f"{_call_title(call)}  …"
            # Kept by reference: the result can arrive before the box is mounted.
            result = Static("", classes="result")
            box = Collapsible(result, title=title, collapsed=True)
            self._calls[call.id] = (call, box, result, time.monotonic())
            if name != PLAN_TOOL:
                self._mount(box)
            if _is_agent(call):
                root = self._root or self.app.query_one("#agents", Tree).root
                node = root.add_leaf(name)
                node.data = name
                self._agent_nodes[call.id] = node
                self.fit_panel()
                self._set_state(node, "working")
                task = next(iter(arguments(call).values()), "")
                self._message(self._agent, name, "task", str(task))

    def _result(self, event: ToolResultEvent) -> None:
        found = self._calls.pop(event.tool_call_id, None)
        if found is None:
            return
        call, box, result, started = found
        took = time.monotonic() - started
        if not self._calls:
            # The last result of a step: the next one starts.
            self.steps = min(self.steps + 1, self.max_steps or self.steps + 1)
            self.doing("thinking")
        name = event.name
        tool = self._by_tool.setdefault(name, ToolStats())
        if event.is_error:
            tool.failed += 1
            if name in self._used:
                self._used[name].failed += 1
                self._show_used()
            summary = "✗ " + error_line(event.content)
            box.add_class("error")
            if name == PLAN_TOOL:
                self._mount(box)
        elif name == PLAN_TOOL:
            summary = "plan updated"
        else:
            summary, _ = result_summary(name, event.content)
        box.title = f"{_call_title(call)}  ·  {summary}  ·  {took:.1f}s"
        self._seen(event.content)
        shown = _shown(event.content)
        result.update(Text(shown))
        node = self._agent_nodes.pop(event.tool_call_id, None)
        if node is not None:
            self._set_state(node, "failed" if event.is_error else "done")
            self._message(name, self._agent, "answer", event.content)

    def _reasoning(self, text: str) -> None:
        self._thought_text += text
        if self._thought is None:
            self._thought = Static(classes="result")
            self._mount(Collapsible(self._thought, title="thinking", collapsed=True))
        self._thought.update(Text(self._thought_text))

    def _plan(self, call: ToolCall) -> None:
        items = todos(call)
        done = sum(status == "completed" for status, _ in items)
        # Also in the status line, which stays in sight when the plan scrolls away.
        self._plan_done = f"plan {done}/{len(items)}"
        text = Text.assemble((self._plan_done, "bold #a99bc9"))
        for status, content in items:
            mark, style = PLAN_MARKS.get(status, PLAN_MARKS["pending"])
            text.append(f"\n{mark} {content}", style=style or "#d4d4d8")
        self._show_side_plan(_plan_window(items), f"{done}/{len(items)}")
        if self._plan_block is None:
            self._plan_block = Static(text, classes="plan")
            self._mount(self._plan_block)
        else:
            self._plan_block.update(text)

    def _show_side_plan(self, text: Text | None = None, count: str = "") -> None:
        if text is None:
            text = Text("no plan yet", style="#8a8f99")
        box = self.app.query_one("#side-plan", Static)
        box.border_title = f"PLAN {count}".rstrip()
        box.update(text)

    def _message(self, sender: str, to: str, what: str, content: str) -> None:
        table = self.app.query_one("#messages", DataTable)
        key = str(len(self.messages))
        self.messages[key] = (sender, to, what, content or "", time.strftime("%H:%M"))
        self._add_message_row(key)
        table.border_title = f"MESSAGES {len(self.messages)}"
        # The same place and width as the empty one it replaces.
        table.display = True
        self.app.query_one("#no-messages").display = False
        table.move_cursor(row=table.row_count - 1)
        self.fit_panel()

    def _add_message_row(self, key: str) -> None:
        """Who to whom, what and when, then the first line of the message under it."""
        sender, to, what, content, when = self.messages[key]
        first = content.strip().splitlines()[0] if content.strip() else ""
        table = self.app.query_one("#messages", DataTable)
        # Hidden, the table has no width yet: a guess, cut again once shown.
        room = max(8, table.scrollable_content_region.width or 36)
        if not self._side:
            # In the short panel at the top: all on one line, both names kept.
            who = f"{_cut(sender, 8)}→{_cut(to, 8)}"
            text = _cut(first, room - len(when) - cell_len(who) - 3)
            table.add_row(
                Text.assemble((when, "#8a8f99"), " ", (who, "bold"), "  " + text),
                key=key,
            )
            return
        # The names in bold as in AGENTS, kind and time in grey, the text plain.
        what = f"{what} {when}"
        who = _cut(f"{sender} → {to}", room - len(what) - 1)
        row = Text.assemble(
            (who, "bold"),
            " " * max(1, room - cell_len(who) - len(what)),
            (what, "#8a8f99"),
            "\n" + _cut(first, room),
        )
        table.add_row(row, key=key, height=2)

    def fill_messages(self) -> None:
        """The rows again, cut to the width the table has now."""
        table = self.app.screen_stack[0].query_one("#messages", DataTable)
        # The column too: it keeps the width of its widest row otherwise.
        table.clear(columns=True)
        table.add_column("message", key="message")
        for key in self.messages:
            self._add_message_row(key)
        if table.row_count:
            table.move_cursor(row=table.row_count - 1)

    def _set_state(self, node: TreeNode[str] | None, state: str) -> None:
        if node is None:
            return
        marks = {
            "working": ("●", "#a99bc9"),
            "waiting": ("○", "#8a8f99"),
            "done": ("✓", "#7fbf95"),
            "failed": ("✗", "#e07a6f"),
            "stopped": ("■", "#e0b36f"),
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


def _cut(text: str, size: int) -> str:
    """At most `size` terminal cells, wide characters counted twice."""
    if size <= 0:
        return ""
    if cell_len(text) <= size:
        return text
    return set_cell_size(text, size - 1) + "…"


def _plan_window(items: list[tuple[str, str]]) -> Text:
    """The plan for the column: all of it, or the items around the current one."""
    current = next(
        (i for i, (status, _) in enumerate(items) if status != "completed"),
        len(items) - 1,
    )
    start = max(0, min(current - 1, len(items) - PLAN_ROWS))
    shown = items[start : start + PLAN_ROWS]
    text = Text()
    if start:
        text.append(f"✓ {start} done before\n", style="#8a8f99")
    for status, content in shown:
        mark, style = PLAN_MARKS.get(status, PLAN_MARKS["pending"])
        text.append(f"{mark} {content}\n", style=style or "#d4d4d8")
    if start + len(shown) < len(items):
        text.append(f"… {len(items) - start - len(shown)} more\n", style="#8a8f99")
    text.rstrip()
    return text


def _tokens(count: int, exact: bool = True) -> str:
    """A token count in a few characters: 45.0k, 999.9k, 1.2M (200k, 1M as totals)."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M" if exact else f"{count / 1_000_000:g}M"
    return f"{count / 1000:.1f}k" if exact else f"{count // 1000}k"


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
    task: str | None = None,
) -> None:
    """Open the full-screen view on the ACP agent that `command` starts.

    Args:
        command: The agent's command line; it runs in `workspace`.
        workspace: The folder of the conversation.
        label: The name shown in the header, e.g. the model.
        model: The model, to estimate how full its context window is.
        max_steps: The agent's step limit, shown in the activity panel.
        task: A first message, sent once the agent has started.
    """
    AntApp(
        command,
        workspace,
        label=label,
        context_size=context_size(model),
        max_steps=max_steps,
        task=task,
    ).run()
