"""ACP agent example — connect from VSCode, Zed, Gemini CLI, or any ACP-compatible editor.

Features demonstrated
---------------------
- **Filesystem tools** – read and write files in the IDE via ``acp_fs_read_file`` /
  ``acp_fs_write_file`` (requires the client to advertise ``clientCapabilities.fs``).
- **Terminal tools** – run shell commands in the IDE terminal via ``acp_terminal_run``
  (requires ``clientCapabilities.terminal``).
- **Plan updates** – push a plan to the IDE UI via ``acp_send_plan``.
- **Slash commands** – ``/read-file`` and ``/run`` are advertised to the IDE on the
  first prompt of each session.
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

from acp.schema import AvailableCommand, AvailableCommandInput

from ant_ai.acp import (
    ACP_FILESYSTEM_TOOLS,
    ACP_PLAN_TOOLS,
    ACP_SESSION_TOOLS,
    ACP_TERMINAL_TOOLS,
    ACPServer,
)
from ant_ai.agent.agent import Agent
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
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
        slash_commands=[
            AvailableCommand(
                name="read-file",
                description="Read a file from the IDE filesystem",
                input=AvailableCommandInput(hint="path/to/file.txt"),
            ),
            AvailableCommand(
                name="run",
                description="Run a shell command in the IDE terminal",
                input=AvailableCommandInput(hint="ls -la"),
            ),
            AvailableCommand(
                name="plan",
                description="Ask the agent to create a plan before acting",
            ),
        ],
    )

    if "--ws" in sys.argv:
        server.serve()
    else:
        server.serve_stdio()


if __name__ == "__main__":
    main()
