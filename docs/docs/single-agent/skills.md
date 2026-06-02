---
title: Skills
---

# Agent Skills

Skills are reusable, self-contained capability bundles that give an agent step-by-step procedural knowledge without changing its code. They follow the [agentskills.io](https://agentskills.io/specification) specification and are injected into the agent's system prompt on startup.

## Installing skills

Browse the [skills.sh](https://www.skills.sh/) directory and install any skill with a single command:

```sh
npx skills add <owner>/<skill-name>
```

For example:

```sh
npx skills add vercel-labs/find-skills
```

This downloads the skill into a local `.agents/skills/` folder and records it in `skills-lock.json` (similar to a package lock file).

## Loading skills into an agent

Pass the path (or a list of paths) to your skills directories directly to [`Agent`][ant_ai.agent.agent.Agent] via the `skills` parameter:

```python
from ant_ai import Agent
from ant_ai.llm.integrations import LiteLLMChat

agent = Agent(
    name="Assistant",
    system_prompt="You are a helpful assistant.",
    llm=LiteLLMChat("gpt-5-nano"),
    skills=[".agents/skills"],
)
```

The agent loads skills from all listed directories on startup, silently skipping any folder that is missing a `SKILL.md`, has invalid frontmatter, or fails validation — so partial or corrupted installs never crash the agent.

## How it works

On each call to `agent.stream()` the agent receives a `## Skills System` block appended to its system prompt. The block lists every available skill by name and description, and points to the skill's `SKILL.md` path on disk:

```
## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

**Available Skills:**

- **csv-to-markdown**: Convert a CSV file to a Markdown table. ...
  -> Read `/path/to/.agents/skills/csv-to-markdown/SKILL.md` for full instructions (pass `limit=1000`)
```

When the task matches a skill, the agent reads the full `SKILL.md` via `cat` (or any shell command) to get step-by-step instructions and any script references — no custom activation tool is required.

## Writing your own skill

Skills follow the [agentskills.io specification](https://agentskills.io/specification).


A skill is a folder whose name matches the `name` field in its `SKILL.md` frontmatter:

```
.agents/skills/
└── my-skill/
    ├── SKILL.md
    └── scripts/
        └── helper.py   # optional supporting scripts
```

`SKILL.md` structure:

```markdown
---
name: my-skill
description: One-line description of what the skill does and when to use it.
compatibility: Python 3.10+      # optional
license: MIT                     # optional
allowed-tools: Bash Read Write   # optional — space-separated list
metadata:                        # optional arbitrary key-value pairs
  author: you
  version: "1.0"
---

# My Skill

Detailed instructions for the agent. Explain when to use this skill, what steps
to follow, and how to invoke any supporting scripts.

## Steps

1. ...
2. ...
```

Key rules:

- `name` must match the folder name exactly (lowercase letters, numbers, hyphens).
- `description` is what the agent sees in the skills list — keep it one line and action-oriented.
- Scripts in `scripts/` are **not** auto-registered as tools. Reference them by path in the instructions so the agent can invoke them via `Bash`.

## Sharing skills

Publish your skill to [skills.sh](https://www.skills.sh/) by pushing it to a public GitHub repository. Other users can then install it with:

```sh
npx skills add <your-github-user>/<repo-name>
```

## Example: running `find-skills`

`find-skills` is the most popular skill on skills.sh. Given a user question like _"is there a skill for X?"_, the agent activates it, reads the full instructions, and walks the user through discovering and installing the right skill.

### 1. Install the skill

```sh
npx skills add https://github.com/vercel-labs/skills --skill find-skills
```

This creates:

```
.agents/skills/
└── find-skills/
    └── SKILL.md
skills-lock.json
```

### 2. Write the script

A single `run_command` tool is enough — the agent uses it both to read `SKILL.md` (via `cat`) and to run `npx skills find` searches.

```python
# main.py
import asyncio
import subprocess

from ant_ai import Agent, Message, State, InvocationContext, tool
from ant_ai.llm.integrations import LiteLLMChat
from ant_ai.core import FinalAnswerEvent, ToolCallingEvent


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout or result.stderr


agent = Agent(
    name="Assistant",
    system_prompt="You are a helpful assistant.",
    llm=LiteLLMChat("gpt-5-nano"),
    tools=[run_command],
    skills=[".agents/skills"],
)


async def main():
    ctx = InvocationContext(session_id="skills-demo")
    state = State()
    state.add_message(
        Message(role="user", content="Is there a skill for React development?")
    )

    async for event in agent.stream(state, ctx=ctx):
        if isinstance(event, ToolCallingEvent):
            for tc in event.tool_calls:
                print(f"[tool] {tc.function.name} {tc.function.arguments}")
        elif isinstance(event, FinalAnswerEvent):
            print(f"\n{event.content}")


asyncio.run(main())
```

### 3. Run it

```sh
OPENAI_API_KEY=your-key python main.py
```

### 4. Expected output

```
[tool] run_command {"command":"cat .agents/skills/find-skills/SKILL.md"}
[tool] run_command {"command": "npx skills find react"}
[tool] run_command {"command": "npx skills find react development"}

Yes! There are several highly-rated React skills on skills.sh:

- **vercel-react-best-practices** (412K installs) — React and Next.js best
  practices from Vercel Engineering.
  Install: npx skills add vercel-labs/agent-skills@vercel-react-best-practices

Would you like me to install one of these, or tailor the search further?
```

The first `[tool]` line confirms the agent read the `find-skills` SKILL.md via `cat` — that is the skill activating. The subsequent lines show the agent running `npx skills find` searches in parallel before forming its answer.
