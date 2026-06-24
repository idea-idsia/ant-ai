"""Simple ACP agent example — connect from VSCode or any ACP-compatible editor.

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
    Run with ``--ws`` to serve over WebSocket at ``ws://127.0.0.1:9001/acp/ws``
    instead:

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

from ant_ai.acp import ACPServer
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


@tool
def get_weather(city: str) -> str:
    """Return a mock current weather report for a city (for testing purposes).

    Args:
        city: Name of the city, e.g. "London".
    """
    mock_data: dict[str, str] = {
        "london": "Overcast, 14°C, wind 20 km/h from the SW.",
        "new york": "Partly cloudy, 22°C, wind 15 km/h from the NE.",
        "tokyo": "Sunny, 28°C, humidity 60%, wind 10 km/h from the S.",
        "sydney": "Clear skies, 19°C, wind 25 km/h from the SE.",
    }
    return mock_data.get(city.lower(), f"No weather data available for '{city}'.")


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
        system_prompt="You are a helpful AI assistant. Use the available tools when relevant.",
        tools=[get_current_time, calculate, get_weather],
    )
    workflow: Workflow[State] = _build_workflow()

    server = ACPServer(agent=agent, workflow=workflow)

    if "--ws" in sys.argv:
        server.serve()
    else:
        server.serve_stdio()


if __name__ == "__main__":
    main()
