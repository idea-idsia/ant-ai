"""ACP agent example — connect from VSCode, Zed, Gemini CLI, or any ACP-compatible editor.

Features demonstrated
---------------------
- **Filesystem tools** – read and write files in the IDE via ``acp_fs_read_file`` /
  ``acp_fs_write_file`` (requires the client to advertise ``clientCapabilities.fs``).
- **Terminal tools** – run shell commands in the IDE terminal via ``acp_terminal_run``
  (requires ``clientCapabilities.terminal``).
- **Plan updates** – push a plan to the IDE UI via ``acp_send_plan``.
- **Hybrid slash commands** – advertised to the IDE on the first prompt of each
  session and handled two ways:
    * ``/compact`` and ``/skill`` are ``kind="code"`` – a Python handler runs
      against the live session (no model turn). ``/compact`` summarises the
      transcript in place; ``/skill install|remove|list [name]`` manages the
      session's agentskills.io skills, loaded from the catalog root
      ``<cwd>/.skills`` (a ``SKILL.md`` body lands in the system prompt, so that
      path check is the trust boundary).
    * ``/plan`` is ``kind="prompt"`` – it expands into a templated instruction and
      the normal agent turn runs.
- **Per-session MCP servers** – HTTP or SSE MCP servers passed by the client in
  ``session/new`` are automatically loaded and made available as tools.

Usage
-----
**stdio (VSCode / Gemini CLI / Zed)**
    Point your editor's ACP config to this script:

    .. code-block:: json

        {
            "acp": {
                "agents": [
                    {
                        "name": "ant-ai",
                        "command": "python",
                        "args": ["examples/acp_agent.py"]
                    }
                ]
            }
        }

    The editor will spawn the script as a subprocess and communicate over
    stdin/stdout.  No port needed.

**WebSocket (browser / custom client)**
    Run with ``--ws`` to serve over WebSocket at ``ws://127.0.0.1:9001/acp/ws``:

    .. code-block:: bash

        python examples/acp_agent.py --ws

Environment
-----------
``OPENAI_API_KEY`` (or any provider key that LiteLLM supports) must be set.
The model defaults to ``gpt-4o-mini`` but can be overridden via ``ACP_MODEL``.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from ant_ai.acp import (
    ACP_FILESYSTEM_TOOLS,
    ACP_PLAN_TOOLS,
    ACP_SESSION_TOOLS,
    ACP_TERMINAL_TOOLS,
    ACPCommand,
    ACPCommandContext,
    ACPServer,
)
from ant_ai.agent.agent import Agent
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.skills import SkillLoader
from ant_ai.tools.tool import tool
from ant_ai.workflow.workflow import END, START, Workflow


@tool
def get_current_time() -> str:
    """Return the current UTC date and time."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression and return the result.

    Supports basic arithmetic (+, -, *, /), powers (**), and math functions
    such as sqrt, sin, cos, log, floor, ceil, abs, round.

    Args:
        expression: A mathematical expression string, e.g. "sqrt(144) + 2**8".
    """
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["abs"] = abs
    allowed["round"] = round
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


async def _cmd_compact(ctx: ACPCommandContext) -> str:
    """`/compact` — summarise the session transcript in place (no model turn)."""
    convo = [m for m in ctx.history if m.role in ("user", "assistant")]
    if len(convo) < 4:
        return "Not enough conversation to compact yet."
    transcript = "\n".join(f"{m.role}: {m.content}" for m in convo)
    resp = await ctx.agent.llm.ainvoke(
        [
            Message(
                role="user",
                content=(
                    "Summarise the conversation below into a compact briefing the "
                    "assistant can continue from. Keep decisions, facts and open "
                    f"threads; drop chit-chat.\n\n{transcript}"
                ),
            )
        ]
    )
    summary = str(resp.message.content or "")
    n = len(ctx.history)
    ctx.history[:] = [
        Message(role="user", content=f"[Conversation so far, compacted]\n{summary}")
    ]
    return f"Compacted {n} messages into a {len(summary)}-char summary."


def _skills_root(ctx: ACPCommandContext) -> Path:
    """Catalog directory this session may install skills from."""
    return (Path(ctx.cwd or ".") / ".skills").resolve()


def _rebuilt_with_skills(agent: Agent, skill_dirs: list[Path]) -> Agent:
    # A fresh Agent is required: model_copy does not re-run the validator that
    # loads `_skills`. Passing individual skill folders relies on SkillLoader
    # accepting a dir that is itself a skill.
    return Agent(
        name=agent.name,
        description=agent.description,
        llm=agent.llm,
        system_prompt=agent.system_prompt,
        tools=list(agent.tools),
        skills=skill_dirs or None,
    )


async def _cmd_skill(ctx: ACPCommandContext) -> str:
    """`/skill install|remove|list [name]` — manage this session's skills.

    Skills are installed from a per-session catalog root (``<cwd>/.skills``). The
    body of a ``SKILL.md`` is injected into the agent's system prompt, so the
    catalog-root check below is the trust boundary — never widen it to arbitrary
    paths.
    """
    verb, _, name = ctx.args.strip().partition(" ")
    verb, name = verb.strip(), name.strip()
    installed: list[Path] = [s.skill_dir for s in ctx.agent._skills]

    if verb in ("", "list"):
        names = [s.name for s in ctx.agent._skills]
        return "Installed skills: " + (", ".join(names) if names else "(none)")

    if verb == "install":
        if not name:
            return "Usage: /skill install <name>"
        root = _skills_root(ctx)
        folder = (root / name).resolve()
        if not folder.is_relative_to(root) or folder.is_symlink():
            return f"'{name}' is outside the allowed skills root ({root})."
        if not (folder / "SKILL.md").is_file():
            return f"No skill '{name}' in {root}."
        if not SkillLoader(folder).load():
            return f"Skill '{name}' has an invalid SKILL.md."
        if folder in installed:
            return f"Skill '{name}' is already installed."
        ctx.replace_agent(_rebuilt_with_skills(ctx.agent, [*installed, folder]))
        return f"Installed skill '{name}' for this session."

    if verb == "remove":
        keep = [s.skill_dir for s in ctx.agent._skills if s.name != name]
        if len(keep) == len(installed):
            return f"Skill '{name}' is not installed."
        ctx.replace_agent(_rebuilt_with_skills(ctx.agent, keep))
        return f"Removed skill '{name}'."

    return "Usage: /skill install <name> | /skill remove <name> | /skill list"


async def _run_agent(agent, state: State, ctx: InvocationContext | None):
    async for event in agent.stream(state, ctx=ctx):
        yield event
    yield state


def _build_workflow() -> Workflow:
    wf = Workflow()
    wf.add_node("run", _run_agent)
    wf.add_edge(START, "run")
    wf.add_edge("run", END)
    return wf


def main() -> None:
    model = os.environ.get("ACP_MODEL", "gpt-4o-mini")

    agent = Agent(
        name="ant-ai",
        description="A helpful AI assistant powered by ant-ai.",
        llm=LiteLLMChat(model),
        system_prompt=(
            "You are a helpful AI assistant with access to the IDE's filesystem and terminal.\n"
            "Use acp_fs_read_file / acp_fs_write_file to read and write files — relative paths are resolved against the session working directory automatically.\n"
            "If you need to know the current working directory, call acp_get_cwd() once.\n"
            "Use acp_terminal_run to execute shell commands in the IDE terminal.\n"
            "Use acp_send_plan to share your plan with the user before starting complex tasks.\n"
            "When the user provides @-mentioned files they appear as [File: ...] in the message."
        ),
        tools=[
            get_current_time,
            calculate,
            *ACP_SESSION_TOOLS,
            *ACP_FILESYSTEM_TOOLS,
            *ACP_TERMINAL_TOOLS,
            *ACP_PLAN_TOOLS,
        ],
    )
    workflow: Workflow[State] = _build_workflow()

    server = ACPServer(
        agent=agent,
        workflow=workflow,
        commands=[
            ACPCommand(
                name="compact",
                description="Summarise the conversation so far to free up context",
                kind="code",
                handler=_cmd_compact,
            ),
            ACPCommand(
                name="skill",
                description="Manage this session's skills (from <cwd>/.skills)",
                input_hint="install <name> | remove <name> | list",
                kind="code",
                handler=_cmd_skill,
            ),
            ACPCommand(
                name="plan",
                description="Draft a plan before acting",
                input_hint="what you want done",
                kind="prompt",
                template=(
                    "Draft a short step-by-step plan for the task below, then stop "
                    "and wait for my approval:\n\n{args}"
                ),
            ),
        ],
    )

    if "--ws" in sys.argv:
        server.serve()
    else:
        server.serve_stdio()


if __name__ == "__main__":
    main()
