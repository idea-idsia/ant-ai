from __future__ import annotations

from collections.abc import Awaitable
from typing import Literal, overload

from a2a.types import AgentCard
from a2a.utils.errors import A2AError
from pydantic import Field, PrivateAttr, model_validator

from ant_ai.a2a.client import A2AClient, AgentClientError
from ant_ai.a2a.config import A2AConfig
from ant_ai.a2a.session import current_session_id
from ant_ai.core.events import ClarificationNeededEvent, FinalAnswerEvent
from ant_ai.core.result import ClarificationNeededOutput
from ant_ai.core.types import InvocationContext
from ant_ai.tools.tool import Tool


class A2AAgentTool(Tool):
    """
    A tool that provides an interface to interact with the Agent via the A2A protocol.
    """

    config: A2AConfig = Field(..., description="Configuration for the A2A client.")
    agent_input_description: str = Field(
        default="Message to send to the agent. Contains all the necessary information to answer the question in a clear way.",
        description="Description of the input to the agent. This description will be used by the agent to generate the prompt for remote agent request.",
    )

    _a2a: A2AClient | None = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _agent_card: AgentCard | None = PrivateAttr(default=None)
    # Remote tasks waiting on an answer, by caller session. The tool instance is
    # shared by every session, so one slot would hand a task to the wrong one.
    _pending_tasks: dict[str | None, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _set_defaults(self) -> A2AAgentTool:
        return self

    def _ensure_a2a(self) -> None:
        if self._a2a is None:
            self._a2a = A2AClient(config=self.config)
            self._a2a._agent_card = self._agent_card

    def _init_metadata(self, agent_card: AgentCard) -> None:
        """Set Tool metadata exactly once from an AgentCard."""
        if not self.name:
            self.name = agent_card.name
        if not self.description:
            self.description = self._create_agent_description(agent_card)

        self.parameters = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (self.agent_input_description),
                }
            },
            "required": ["message"],
        }

    async def _ensure_initialized(self) -> None:
        """Fetch AgentCard (if needed) and set metadata/_func exactly once."""
        if self._initialized:
            return

        self._ensure_a2a()
        if self._a2a is None:
            raise RuntimeError("A2A client not initialized")

        agent_card: AgentCard = self._agent_card or await self._a2a.get_agent_card()
        self._agent_card: AgentCard = agent_card
        self._init_metadata(agent_card)

        self._attach_func()
        self._initialized = True

    def _attach_func(self) -> None:
        """Attach the call function to the _func (single callable Tool)."""

        async def _call_remote(
            message: str, ctx: InvocationContext | None = None
        ) -> str | ClarificationNeededOutput:
            """Send `message` to the remote agent and return its answer.

            When the remote agent asks a question instead (its task ends
            `input-required`), the question is returned as a
            `ClarificationNeededOutput`, so the caller's run stops and the
            user is asked. The next call in the same session continues that
            task (its `task_id`), so the remote agent resumes with its own
            history and reads the message as the answer.
            """
            await self._ensure_initialized()
            self._ensure_a2a()
            if self._a2a is None:
                raise RuntimeError("A2A client not initialized")

            # The A2A executor sets the ambient session; a caller run anywhere
            # else (in-process, ACP) carries it only in its `ctx`.
            session_id: str | None = current_session_id.get(None) or (
                ctx.session_id if ctx else None
            )
            # Without a session there is nothing to keep a waiting task for:
            # it would be shared by every caller of this tool.
            pending: str | None = (
                self._pending_tasks.get(session_id) if session_id else None
            )

            last_text: str = ""
            try:
                async for ev in self._a2a.send_message(
                    message,
                    context_id=session_id,
                    # Only an agent marked `trusted` receives the caller's context.
                    request_metadata=ctx.outbound_metadata()
                    if ctx and self.config.trusted
                    else None,
                    task_id=pending,
                ):
                    if ev.content:
                        last_text: str = ev.content
                    if isinstance(ev, ClarificationNeededEvent):
                        if session_id and ev.task_id:
                            self._pending_tasks[session_id] = ev.task_id
                        elif session_id:
                            self._pending_tasks.pop(session_id, None)
                        return ClarificationNeededOutput(
                            question=ev.content or last_text
                        )
                    if isinstance(ev, FinalAnswerEvent):
                        break
            except AgentClientError as exc:
                # A failed request keeps the waiting task, so a retry still
                # continues it, unless the remote agent rejected the task
                # (gone, finished, other context): retrying it cannot work.
                if pending and isinstance(exc.__cause__, A2AError):
                    self._pending_tasks.pop(session_id, None)
                raise
            if session_id:
                self._pending_tasks.pop(session_id, None)
            return last_text

        self._func = _call_remote
        self._wants_ctx = True

    @overload
    @classmethod
    def from_config(cls, config: A2AConfig, agent_card: AgentCard) -> A2AAgentTool:
        """Creates an A2A agent tool from a configuration and an agent card.

        Returns:
            An A2A agent tool.
        """
        ...

    @overload
    @classmethod
    def from_config(
        cls, config: A2AConfig, agent_card: None = None
    ) -> Awaitable[A2AAgentTool]:
        """Creates an A2A agent tool from a configuration.

        Args:
            config: _description_
            agent_card: _description_. Defaults to None.

        Returns:
            An awaitable of an A2A agent tool.
        """
        ...

    @classmethod
    def from_config(
        cls, config: A2AConfig, agent_card: AgentCard | None = None
    ) -> A2AAgentTool | Awaitable[A2AAgentTool]:
        """Creates an A2A agent tool from a configuration and an optional agent card. If no agent card is provided, the tool will be initialized asynchronously and the tool will be returned as an awaitable tool.

        Args:
            config: _description_
            agent_card: _description_. Defaults to None.

        Returns:
            An A2A agent tool or an awaitable of an A2A agent tool.
        """
        tool: A2AAgentTool = cls(name=None, description=None, config=config)

        if agent_card is not None:
            tool._agent_card = agent_card
            tool._ensure_a2a()
            tool._init_metadata(agent_card)
            tool._attach_func()
            tool._initialized = True
            return tool

        async def _build() -> A2AAgentTool:
            await tool._ensure_initialized()
            return tool

        return _build()

    def _create_agent_description(self, agent_card: AgentCard) -> str:
        parts: list[str] = [
            agent_card.description,
        ]

        if agent_card.skills:
            parts += ["", "### Available Skills", ""]
            for skill in agent_card.skills:
                parts.append(f"**{skill.name}**")
                parts.append(skill.description)
                if skill.tags:
                    parts.append(f"Tags: {', '.join(skill.tags)}")
                examples = list(skill.examples) if skill.examples else []
                if examples:
                    parts.append("Examples:")
                    for ex in examples:
                        parts.append(f"  - {ex}")
                parts.append("")

        return "\n".join(parts)

    @property
    def is_namespace(self) -> Literal[False]:
        return False

    def _sid(self) -> str:
        return current_session_id.get()
