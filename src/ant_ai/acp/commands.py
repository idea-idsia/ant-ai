from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from acp.interfaces import Client
from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
    ClientCapabilities,
    UnstructuredCommandInput,
)

from ant_ai.agent.agent import Agent
from ant_ai.core.message import Message


@dataclass
class ACPCommandContext:
    """Everything a code-dispatched command handler may need about the live session.

    ``history`` is the *same* list the adapter keeps for the session, so a handler
    that wants to change the transcript (e.g. ``/compact``) mutates it in place::

        ctx.history[:] = [Message(role="user", content=summary)]

    ``replace_agent`` swaps the agent used for the rest of the session (e.g.
    ``/skill`` installs a skill by handing back a freshly built agent).
    """

    args: str
    session_id: str
    client: Client | None
    capabilities: ClientCapabilities | None
    cwd: str | None
    history: list[Message]
    agent: Agent
    replace_agent: Callable[[Agent], None]


# Return a string to have the adapter send it to the client as one agent message,
# or None if the handler already pushed its own session updates.
CommandHandler = Callable[[ACPCommandContext], Awaitable[str | None]]


@dataclass(frozen=True)
class ACPCommand:
    """A slash command advertised to the ACP client, plus how it is handled.

    - ``kind="code"``: ``handler`` runs against an :class:`ACPCommandContext`; no
      model turn happens. Use for concrete operations (``/compact``, ``/skill``).
    - ``kind="prompt"``: the command expands via ``template`` (``"{args}"`` is
      replaced with the text after the command) and the normal workflow runs.
      With ``template=None`` this is just "advertise it and pass the raw text to
      the agent" -- the pre-existing behaviour.
    """

    name: str
    description: str
    input_hint: str | None = None
    kind: Literal["code", "prompt"] = "prompt"
    handler: CommandHandler | None = None
    template: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "code" and self.handler is None:
            raise ValueError(
                f"ACPCommand {self.name!r}: kind='code' requires a handler"
            )

    def expand(self, args: str) -> str:
        # str.replace, not str.format: `args` may legitimately contain braces.
        return (self.template or "{args}").replace("{args}", args)

    def to_available_command(self) -> AvailableCommand:
        return AvailableCommand(
            name=self.name,
            description=self.description,
            input=AvailableCommandInput(
                root=UnstructuredCommandInput(hint=self.input_hint)
            )
            if self.input_hint
            else None,
        )


_SLASH_RE = re.compile(r"/([A-Za-z0-9_-]+)(?:\s+(.*))?\Z", re.DOTALL)


def parse_slash_command(text: str) -> tuple[str | None, str]:
    """Split a leading slash command off the prompt text.

    ``"/skill ~/s/foo"`` -> ``("skill", "~/s/foo")``; ``"/compact"`` ->
    ``("compact", "")``; anything else (plain text, a bare path like
    ``/etc/hosts``) -> ``(None, "")``.
    """
    match = _SLASH_RE.match(text.lstrip())
    if match is None:
        return None, ""
    return match.group(1), (match.group(2) or "").strip()
