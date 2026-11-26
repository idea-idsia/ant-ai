"""`ant --serve`: the coding agent of `ant`, served over ACP on stdin/stdout.

An editor, or `ant --tui`, starts this process and talks to it. Writing a file
or running a command asks the client for permission first.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from ant_ai.acp import ACPServer, request_permission
from ant_ai.cli.tools import Confirm
from ant_ai.core.events import AnyEvent
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.protocol import ChatLLM
from ant_ai.workflow.workflow import END, START, Workflow


def acp_confirm(*, auto_approve: bool = False) -> Confirm:
    """Asks the client before an action.

    An "always" answer covers every action of the same kind, writes or
    commands, for the rest of that ACP session only: the library keeps it per
    session (see `request_permission`).

    Args:
        auto_approve: Allow every action without asking (`--yes`).

    Returns:
        The `confirm` callback of `coding_tools`.
    """

    async def confirm(action: str) -> bool:
        if auto_approve:
            return True
        kind = "execute" if action.startswith("run ") else "edit"
        choice = await request_permission(action, kind=kind, key=kind)
        return choice in ("allow_once", "allow_always")

    return confirm


def workflow(max_steps: int) -> Workflow:
    """One node that runs the agent for a turn.

    Args:
        max_steps: The step limit of one turn.

    Returns:
        The workflow `ACPServer` runs for each prompt.
    """

    async def run(
        agent, state: State, ctx: InvocationContext | None
    ) -> AsyncIterator[AnyEvent | State]:
        async for event in agent.stream(state, max_steps=max_steps, ctx=ctx):
            yield event
        yield state

    wf = Workflow()
    wf.add_node("run", run)
    wf.add_edge(START, "run")
    wf.add_edge("run", END)
    return wf


def serve(llm: ChatLLM, workspace: Path, *, max_steps: int, auto_approve: bool) -> None:
    """Serve the coding agent of `ant` over ACP on stdin/stdout, until the client leaves.

    Args:
        llm: The model of the agent.
        workspace: The folder its tools work in, for every session: the
            folder a client sends with a new session is not used.
        max_steps: The step limit of one turn.
        auto_approve: Allow every action without asking the client.
    """
    from ant_ai.cli.app import build_agent

    agent = build_agent(llm, workspace, confirm=acp_confirm(auto_approve=auto_approve))
    ACPServer(agent=agent, workflow=workflow(max_steps)).serve_stdio()
