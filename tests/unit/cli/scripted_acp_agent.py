"""The `ant` coding agent served over ACP with a scripted model, for the tests.

Run as `python scripted_acp_agent.py <workspace> [script]`:

- `write` (default): one step that writes `a.txt`, then an answer.
- `twice`: one step that writes `a.txt`, one that writes `b.txt`, then an answer.
- `stuck`: the first turn never ends (it writes `started` in the workspace
  once it waits), the next ones answer `done`.
- `slow`: the first turn writes `started` and waits for a `go` file in the
  workspace before it answers `first`; the next one answers `second`.
"""

import asyncio
import json
import sys
from pathlib import Path

from ant_ai.acp import ACPServer
from ant_ai.cli.app import build_agent
from ant_ai.cli.serve import acp_confirm, workflow
from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.response import ChatLLMResponse
from ant_ai.llm.protocol import ChatLLM

WORKSPACE = Path(sys.argv[1])
SCRIPT = sys.argv[2] if len(sys.argv) > 2 else "write"


class ScriptedLLM(ChatLLM):
    model = "scripted"
    api_key = None
    api_base = None

    def __init__(self, *responses: ChatLLMResponse | str | None):
        self.responses = list(responses)

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        response = self.responses.pop(0)
        if response is None:
            (WORKSPACE / "started").touch()
            await asyncio.Event().wait()
        if isinstance(response, str):
            (WORKSPACE / "started").touch()
            while not (WORKSPACE / "go").exists():
                await asyncio.sleep(0.02)
            return answer(response)
        return response


def write(path: str, call_id: str) -> ChatLLMResponse:
    return ChatLLMResponse(
        message=Message(role="assistant", content=f"I write {path}."),
        tool_calls=[
            ToolCall(
                id=call_id,
                function=ToolFunction(
                    name="write_file",
                    arguments=json.dumps({"path": path, "content": "x"}),
                ),
            )
        ],
    )


def answer(text: str) -> ChatLLMResponse:
    return ChatLLMResponse(message=Message(role="assistant", content=text))


SCRIPTS = {
    "write": [write("a.txt", "c1"), answer("Wrote a.txt.")],
    "twice": [write("a.txt", "c1"), write("b.txt", "c2"), answer("Wrote both.")],
    "stuck": [None, answer("done")],
    "slow": ["first", answer("second")],
}

llm = ScriptedLLM(*SCRIPTS[SCRIPT])
agent = build_agent(llm, WORKSPACE, confirm=acp_confirm())
ACPServer(agent=agent, workflow=workflow(max_steps=5)).serve_stdio()
