"""The smallest adaptive-topology run: three agents, one string, one report.

uv run python -m examples.topology_quickstart --dry-run   # no LLM needed
uv run python -m examples.topology_quickstart             # needs an LLM
"""

from __future__ import annotations

import argparse
import asyncio

from a2a.types import AgentCapabilities, AgentCard, AgentInterface

from ant_ai.a2a import Colony
from ant_ai.agent.agent import Agent
from ant_ai.core.events import HealingEvent, TopologyEvent
from ant_ai.llm.integrations import LiteLLMChat
from ant_ai.workflow.workflow import Workflow

ROLES = {
    "architect": "Designs the shape of a solution. Does not write code.",
    "developer": "Writes the code. Asks the architect when the design is unclear.",
    "reviewer": "Reviews code for correctness and says what must change.",
}


def card(name: str, description: str, port: int) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=f"http://{name}:{port}/")
        ],
        skills=[],
    )


def build(model: str) -> Colony:
    colony = Colony()
    for i, (name, description) in enumerate(ROLES.items()):
        colony.agent(
            name,
            agent=Agent(
                name=name,
                system_prompt=f"You are the {name}. {description}",
                description=description,
                llm=LiteLLMChat(model),
            ),
            # Registered because a colony serves requests through one. An
            # ensemble invokes the agent directly, so this one is never run.
            workflow=Workflow(),
            card=card(name, description, 9001 + i),
        )
    # The round-0 topology, and the standing one for any round no stage rewires.
    colony.collab("architect", "developer", mutual=True)
    colony.collab("developer", "reviewer", mutual=True)

    # Semantic rewiring each round (DyTopo), plus structural repair (DIG).
    # Equivalent to: DyTopo() | DigToHeal()
    return colony.evolve("dytopo|dig")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--task", default="Build a CSV parser that handles quoted fields."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and check the ensemble, then print it. No LLM is called.",
    )
    args = parser.parse_args()

    # Configuration is checked here: a combination that cannot work raises
    # TopologyConfigurationError rather than running and producing an artefact.
    ensemble = build(args.model).ensemble(max_rounds=args.rounds)

    if args.dry_run:
        print(
            "pipeline    :",
            " -> ".join(type(s).__name__ for s in ensemble.pipeline.stages),
        )
        print("scheduler   :", type(ensemble.pipeline.scheduler).__name__)
        print("materialiser:", type(ensemble.pipeline.materialiser).__name__)
        print("seed links  :", [(x.src, x.dst) for x in ensemble.seed])
        print("provenance  :", ensemble.provenance["strategy"])
        print("\nConfiguration is valid. Drop --dry-run to run it.")
        return

    async for event in ensemble.stream(args.task):
        if isinstance(event, TopologyEvent):
            edges = ", ".join(f"{link.src}->{link.dst}" for link in event.links)
            print(f"[round {event.round}] topology: {edges or 'none'}")
        elif isinstance(event, HealingEvent):
            print(f"[round {event.round}] {event.pattern}: {event.content}")

    # The two lines that say whether the topology actually did anything.
    print("\n" + ensemble.report().render())


if __name__ == "__main__":
    asyncio.run(main())
