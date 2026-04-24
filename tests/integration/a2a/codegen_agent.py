from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import uvicorn
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

import ant_ai.llm.integrations.lite_llm as _llm_mod
from ant_ai.a2a.colony import Colony
from ant_ai.agent.agent import Agent
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.workflow.workflow import END, START, NodeYield, Workflow


class _FakeUsage:
    def model_dump(self) -> dict:
        return {"in_tokens": 1, "out_tokens": 1}


class _FakeMessage:
    role = "assistant"
    tool_calls: list = []

    def get(self, key: str, default=None):
        return "pong from codegen" if key == "content" else default


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


async def _stub_acompletion(**_) -> _FakeResponse:
    return _FakeResponse()


_llm_mod.acompletion = _stub_acompletion


PORT = int(os.environ.get("CODEGEN_AGENT_PORT", "9001"))


def _make_card() -> AgentCard:
    card = AgentCard(
        name="codegen",
        description="A 10x Software Developer",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
    )
    card.supported_interfaces.append(
        AgentInterface(protocol_binding="JSONRPC", url=f"http://127.0.0.1:{PORT}/")
    )
    card.skills.append(AgentSkill(id="codegen", name="codegen", description="codegen"))
    return card


async def _run_agent_once(
    agent: Agent,
    state: State,
    ctx: InvocationContext | None,
) -> AsyncGenerator[NodeYield]:
    async for event in agent.stream(state, ctx=ctx):
        yield event
    yield state


def _make_workflow() -> Workflow:
    wf = Workflow()
    wf.add_node("run", _run_agent_once)
    wf.add_edge(START, "run")
    wf.add_edge("run", END)
    return wf


agent = Agent(
    name="codegen",
    llm=LiteLLMChat("test-model"),
    system_prompt="You are codegen, a 10x Software Developer.",
    description="A 10x Software Developer",
)

colony = Colony()
colony.agent("codegen", agent=agent, workflow=_make_workflow(), card=_make_card())
app: Starlette = colony.asgi(agent_name="codegen", use_fastapi=True)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
