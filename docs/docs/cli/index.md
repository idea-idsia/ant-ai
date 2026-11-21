---
title: Terminal client
---

# Terminal client

!!! note "Prototype"
    `ant` is an early prototype. Its commands, options and layout may change between releases.

`ant` is a small coding agent that runs in the terminal, built only on the public ant-ai API. It serves two purposes: a ready-to-use agent that reads, edits and runs code in a folder, and a client for any agent served over [ACP](../acp/index.md), with a full-screen view that follows the run.

```bash
uv add "ant-ai[cli]"
```

The `cli` extra adds [Rich](https://github.com/Textualize/rich), [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) and [Textual](https://github.com/Textualize/textual).

## Settings

`ant` takes the model from `--model` or from the `ANT_MODEL` variable, with any name [LiteLLM](https://docs.litellm.ai/docs/providers) accepts. The variables can also live in `~/.ant/env`, one `KEY=value` per line; a variable already set in the environment wins over the file.

```bash
# ~/.ant/env (chmod 600: it can hold a key)
ANT_MODEL=anthropic/claude-sonnet-5-5
ANT_API_KEY=...
# ANT_API_BASE=http://localhost:8000/v1   for a vLLM server
```

Without `ANT_API_KEY`, LiteLLM reads the provider's own variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).

## Ways to run it

| Command                         | What it does                                                                        |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| `ant`                           | Conversation in the terminal, the agent in the same process.                        |
| `ant "fix the failing test"`    | Runs one task and exits.                                                            |
| `ant --tui`                     | Full-screen view; the agent runs in its own process (`ant --serve`) and talks ACP.  |
| `ant --agent "python a.py"` | The same view over any ACP agent, e.g. one served with [`ACPServer`][ant_ai.acp.server.ACPServer]. |

`-C <dir>` sets the folder the agent works in (the current one by default), `--max-steps` the step limit of one turn, and `-c` continues the last conversation in that folder. `ant --serve` alone serves the agent over ACP on stdin/stdout, so an editor can use it like any other ACP agent.

## The agent

The agent is a plain [`Agent`][ant_ai.agent.agent.Agent] with `planning=True` and six tools, all confined to the working folder:

| Tool          | What it does                                               | Asks first |
| ------------- | ---------------------------------------------------------- | :--------: |
| `list_dir`    | Lists a directory.                                         |            |
| `read_file`   | Reads a file, optionally a range of lines.                 |            |
| `search`      | Searches the files for a pattern.                          |            |
| `write_file`  | Writes a whole file.                                       |    yes     |
| `edit_file`   | Replaces one exact piece of text in a file.                |    yes     |
| `run_command` | Runs a shell command in the folder.                        |    yes     |

Before a write or a command, `ant` asks: **y** allows it once, **n** refuses it (the model receives the refusal as a tool error and continues from there), **a** allows every action of the same kind, writes or commands, for the rest of the session. `--yes` skips the questions.

Conversations are saved per folder under `~/.ant/sessions` (or `$ANT_HOME/sessions`); `/resume` or `-c` picks the last one up again.

## The full-screen view

`ant --tui` shows the conversation on the left, with each tool call folded on one line (its main argument, a short result and how long it took) and the model's reasoning, when it sends one, folded too. A panel on the right follows the run:

| Panel        | What it shows                                                                     |
| ------------ | --------------------------------------------------------------------------------- |
| **Agents**   | The agent and any agent it calls as a tool, with their state.                     |
| **Plan**     | The checklist the agent writes with its planning tool.                            |
| **Messages** | The messages between agents: the user's task, the tasks to sub-agents, their answers. `Enter` on a row shows it in full. |
| **Activity** | Steps used out of the limit, and calls per tool with failures.                    |
| **Context**  | An estimate of how full the model's context window is.                            |

| Key      | Action                       |
| -------- | ---------------------------- |
| `Esc`    | Stop the current turn; queued messages go back to the input. |
| `Ctrl+O` | Open or close all tool calls. |
| `Ctrl+L` | Start a new conversation; every panel starts empty. |
| `Ctrl+P` | Command palette.             |
| `Ctrl+Q` | Quit.                        |

A message sent while the agent works waits and is sent when the turn ends.

The view only speaks ACP, so it works the same with an agent written by someone else. Some parts depend on what that agent sends: the plan appears only if it uses a planning tool, and the step limit is shown only for `ant`'s own agent.

## Limits of the prototype

- The context size is estimated from the characters seen, not read from the model's token usage.
- `/resume` and `-c` are not available in the full-screen view yet.
- A command stopped with `Esc` may keep running in the background.
- Tool results are not limited in size yet ([#198](https://github.com/idea-idsia/ant-ai/issues/198)): reading a very long file, listing a large folder or a command with a long output sends all of it to the model, which can fill its context or make a turn expensive. Run `ant` in a project folder, not in your home.
- The ant in the status line is drawn with Unicode 16 block characters; an older terminal or font may show empty boxes instead.
