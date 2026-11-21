from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shlex
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from rich.console import Console
from rich.text import Text

from ant_ai.agent.agent import Agent
from ant_ai.cli import config, history, prompt
from ant_ai.cli.render import Renderer, model_label
from ant_ai.cli.tools import Confirm, coding_tools
from ant_ai.core.message import Message, sanitize_messages
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.protocol import ChatLLM

SYSTEM_PROMPT = """\
You are ant, a minimal coding agent working in the directory {workspace}.

- Read the relevant files before changing them; keep changes small, and prefer edit_file to rewriting a file.
- Before every tool call, write one short sentence: what you are about to do and why.
- Run the tests or the program after a change when the project has them.
- For a task with several steps, write a plan first and keep it up to date.
- Finish with a short answer: what you did, or what you found."""

HELP = """\
/clear    new conversation (the current one stays saved)
/resume   continue the last conversation in this folder
/model    show the model in use (set it in ~/.ant/env or with --model)
/details  show or hide the model's reasoning and tool output
/exit     quit (or Ctrl-D)
Ctrl-C stops the current turn."""


def build_agent(llm: ChatLLM, workspace: Path, *, confirm: Confirm) -> Agent:
    """The coding agent of `ant`."""
    return Agent(
        name="ant",
        system_prompt=SYSTEM_PROMPT.format(workspace=workspace),
        llm=llm,
        tools=coding_tools(workspace, confirm=confirm),
        planning=True,
    )


class Session:
    """One conversation in the terminal: the agent, its history and the renderer."""

    def __init__(
        self,
        llm: ChatLLM,
        workspace: Path,
        *,
        renderer: Renderer | None = None,
        auto_approve: bool = False,
        max_steps: int = 25,
        read_answer: Callable[[str], Awaitable[str]] | None = None,
        history_dir: Path | None = None,
    ):
        self.renderer = renderer or Renderer()
        self._history_dir = history_dir
        self._read_answer = read_answer or prompt.read_answer
        self.workspace = workspace.resolve()
        self.max_steps = max_steps
        self.auto_approve = auto_approve
        self._always: set[str] = set()
        self.agent = build_agent(llm, self.workspace, confirm=self._confirm)
        self.state = State()
        self.file = history.new_file(self.workspace, self._history_dir)
        self.ctx = InvocationContext(session_id=self.file.stem)

    @property
    def label(self) -> str:
        return model_label(self.agent.llm.model)

    async def _confirm(self, action: str) -> bool:
        # "a" allows every action of the same kind, writes or commands.
        kind = "run" if action.startswith("run ") else "write"
        if self.auto_approve or kind in self._always:
            return True
        with self.renderer.paused():
            answer = await self._read_answer(f"allow {action}? y/n/a")
        answer = answer.strip().lower()
        if answer == "a":
            self._always.add(kind)
        return answer in ("y", "yes", "a")

    async def ask(self, text: str) -> None:
        """Send one user message and print the run until the answer."""
        self.state.add_message(Message(role="user", content=text))
        with self.renderer.turn():
            try:
                async for event in self.agent.stream(
                    self.state, max_steps=self.max_steps, ctx=self.ctx
                ):
                    self.renderer.event(event)
            except asyncio.CancelledError:
                self.renderer.note("interrupted")
            finally:
                # A turn cut short can leave a tool call without its result,
                # which the next request to the model would be refused for.
                self.state.messages = sanitize_messages(self.state.messages)
                history.save(self.file, self.state.messages)

    def clear(self) -> None:
        """Start a new conversation; the current one stays saved."""
        self.state = State()
        self.file = history.new_file(self.workspace, self._history_dir)
        self.ctx = InvocationContext(session_id=self.file.stem)

    def resume(self) -> bool:
        """Continue the latest conversation saved for this workspace."""
        last = history.latest(self.workspace, self._history_dir)
        if last is None:
            return False
        self.state = State(messages=history.load(last))
        self.file = last
        self.ctx = InvocationContext(session_id=last.stem)
        return True


async def _repl(session: Session, hint: str, renderer: Renderer) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await prompt.read_message(hint)
        if line is None:
            return
        text = line.strip()
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return
        if text == "/model":
            renderer.note(f"model {session.label}")
            renderer.console.print()
            continue
        if text == "/help":
            renderer.note(HELP)
            renderer.console.print()
            continue
        if text == "/clear":
            session.clear()
            renderer.note("new conversation")
            renderer.console.print()
            continue
        if text == "/resume":
            if session.resume():
                renderer.note(f"resumed: {len(session.state.messages)} messages")
            else:
                renderer.note("no saved conversation in this folder")
            renderer.console.print()
            continue
        if text == "/details":
            renderer.verbose = not renderer.verbose
            renderer.note(f"details {'on' if renderer.verbose else 'off'}")
            renderer.console.print()
            continue
        renderer.user(text)
        turn = asyncio.ensure_future(session.ask(text))
        with contextlib.suppress(NotImplementedError):  # no signal handlers on Windows
            loop.add_signal_handler(signal.SIGINT, turn.cancel)
        try:
            await turn
        finally:
            with contextlib.suppress(NotImplementedError):
                loop.remove_signal_handler(signal.SIGINT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ant",
        description="A minimal coding agent in the terminal, built on ant-ai.",
        epilog="Settings (ANT_MODEL, ANT_API_BASE, ANT_API_KEY) can go in ~/.ant/env, one KEY=value "
        "per line; the environment wins over the file. Without ANT_API_KEY the provider's own "
        "variable is used (OPENAI_API_KEY, ...).",
    )
    parser.add_argument("prompt", nargs="*", help="run this task once and exit")
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get("ANT_MODEL"),
        help="any model name LiteLLM accepts (default: $ANT_MODEL)",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("ANT_API_BASE"),
        help="endpoint URL, e.g. a vLLM server",
    )
    parser.add_argument(
        "-C",
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="directory the agent works in (default: current)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=25, help="step limit of one turn (default: 25)"
    )
    parser.add_argument(
        "-c",
        "--continue",
        dest="resume",
        action="store_true",
        help="continue the last conversation in this folder",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="do not ask before writing files or running commands",
    )
    acp = parser.add_mutually_exclusive_group()
    acp.add_argument(
        "--serve",
        action="store_true",
        help="serve the agent over ACP on stdin/stdout, for an editor or `ant --tui`",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="full-screen view: conversation on the left, agents, plan and messages "
        "on the right; the agent runs in its own process and talks ACP",
    )
    acp.add_argument(
        "--agent",
        metavar="COMMAND",
        help="open the full-screen view on the ACP agent this command starts, e.g. "
        '"python my_agent.py"',
    )
    return parser


def _serve_command(args: argparse.Namespace) -> list[str]:
    """The command that starts this same agent as an ACP server."""
    command = [sys.executable, "-m", "ant_ai.cli", "--serve", "-m", args.model]
    command += ["-C", str(args.workspace.resolve()), "--max-steps", str(args.max_steps)]
    if args.api_base:
        command += ["--api-base", args.api_base]
    if args.yes:
        command.append("--yes")
    return command


def main(argv: list[str] | None = None) -> int:
    warnings = config.load_env()
    args = _parser().parse_args(argv)
    console = Console(highlight=False, stderr=args.serve)
    for warning in warnings:
        console.print(Text(f"ant: {warning}", style="yellow"))
    if not args.model and not args.agent:
        console.print(
            Text(
                "ant: no model. Pass --model, or set ANT_MODEL in ~/.ant/env.",
                style="red",
            )
        )
        return 2
    if not args.workspace.is_dir():
        console.print(Text(f"ant: not a directory: {args.workspace}", style="red"))
        return 2

    if args.tui or args.agent:
        from ant_ai.cli.tui import run

        command = shlex.split(args.agent) if args.agent else _serve_command(args)
        label = model_label(args.model) if args.model else command[0]
        max_steps = None if args.agent else args.max_steps
        run(
            command,
            args.workspace,
            label=label,
            model=args.model,
            max_steps=max_steps,
        )
        return 0

    import litellm

    from ant_ai.llm.integrations import LiteLLMChat

    # LiteLLM prints a provider list when it has no price for the model; not for the user.
    litellm.suppress_debug_info = True
    llm = LiteLLMChat(
        model=args.model,
        api_base=args.api_base,
        api_key=os.environ.get("ANT_API_KEY"),
    )
    if args.serve:
        from ant_ai.cli.serve import serve

        serve(
            llm,
            args.workspace.resolve(),
            max_steps=args.max_steps,
            auto_approve=args.yes,
        )
        return 0

    session = Session(
        llm,
        args.workspace,
        renderer=Renderer(console),
        auto_approve=args.yes,
        max_steps=args.max_steps,
    )
    if args.resume and not session.resume():
        console.print(Text("ant: no saved conversation in this folder", style="dim"))
    try:
        if args.prompt:
            asyncio.run(session.ask(" ".join(args.prompt)))
        else:
            workspace = str(session.workspace).replace(str(Path.home()), "~", 1)
            session.renderer.banner(args.model, workspace)
            asyncio.run(
                _repl(
                    session,
                    f"{model_label(args.model)} · {workspace} · /help",
                    session.renderer,
                )
            )
    except KeyboardInterrupt:
        pass
    return 0
