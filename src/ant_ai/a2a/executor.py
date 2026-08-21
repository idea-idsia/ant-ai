from __future__ import annotations

from contextvars import Token

from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Message as A2AMessage, Task

from ant_ai.a2a.compression import (
    find_compression_checkpoint,
    is_checkpoint_message,
    persist_compression_checkpoint,
)
from ant_ai.a2a.session import current_session_id
from ant_ai.a2a.translator import A2AToHVEvent, HVEventToA2A
from ant_ai.agent.agent import Agent
from ant_ai.core.events import CompletedEvent, Event
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.observer import obs
from ant_ai.workflow.workflow import Workflow


class A2AExecutor(AgentExecutor):
    """
    A2AExecutor is responsible for processing an A2A request.

    It handles the execution the workflow and propages the updates generated in it.
    """

    def __init__(
        self, agent: Agent, workflow: Workflow, *, stream_artifacts: bool = False
    ):
        """Initialize the A2AExecutor. A2AExecutor is a subclass of AgentExecutor. The AgentExecutor is the a2a-sdk class
        that is responsible for processing the request made to the agent.

        Args:
            agent: Agent that will be used to execute the workflow.
            workflow: Workflow that will be executed.
            stream_artifacts: Whether to translate ContentDeltaEvent into A2A
                artifact-update chunks. See `HVEventToA2A`.
        """
        self.workflow: Workflow = workflow
        self.agent: Agent = agent
        self._translator: HVEventToA2A = HVEventToA2A(stream_artifacts=stream_artifacts)
        self._a2a_to_hv: A2AToHVEvent = A2AToHVEvent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Entry point of the A2A. This is what processes each request to the agent via a2a.

        Args:
            context: The object containing all info about the request.
            event_queue: The event queue to which events will be enqueued.
        """

        if not context.message:
            raise Exception("No message provided")

        task: Task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        with (
            obs.attach_propagation_context(context.metadata),
            obs.bind(
                agent_name=self.agent.name, task_id=task.id, context_id=task.context_id
            ),
        ):
            await obs.event("a2a.execute", task_id=task.id, context_id=task.context_id)
            try:
                await self._execute(context, updater, task)
            except Exception as exc:
                await obs.exception("a2a.error", exc)
                raise InternalError() from exc

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("Task cancel not supported yet.")

    async def _execute(
        self,
        context: RequestContext,
        updater: TaskUpdater,
        task: Task,
    ) -> None:
        ctx = InvocationContext(
            session_id=task.context_id,
            user_id=context.metadata.get("user_id", None),
            llm_settings=context.metadata.get("llm_settings", None),
            workflow_settings=context.metadata.get("workflow_settings", None),
        )

        history = self._build_history(context.related_tasks, context.get_user_input())

        await obs.event("a2a.history", history_messages=len(history))

        state: State = self.workflow.create_state(messages=history)
        token: Token[str] = current_session_id.set(ctx.session_id)

        await obs.event("a2a.workflow.start")
        try:
            async for event in self.workflow.stream(
                agent=self.agent, ctx=ctx, state=state
            ):
                await obs.event(
                    "a2a.workflow.event",
                    workflow_event=getattr(event, "kind", type(event).__name__),
                    node=getattr(event.origin, "node", "-"),
                    step=getattr(event.origin, "run_step", "-"),
                )
                if isinstance(event, CompletedEvent):
                    await persist_compression_checkpoint(state, updater)
                await self.process_event(event, updater)
        finally:
            current_session_id.reset(token)

        await obs.event("a2a.workflow.end")

    async def process_event(self, event: Event, updater: TaskUpdater) -> None:
        await self._translator.apply(event=event, updater=updater)

    def _build_history(
        self, related_tasks: list[Task], user_input: str
    ) -> list[Message]:
        checkpoint, cp_idx = find_compression_checkpoint(related_tasks)
        if checkpoint is not None:
            post_a2a = [
                m
                for r_task in related_tasks[cp_idx:]
                for m in r_task.history or []
                if not is_checkpoint_message(m)
            ]
            history = list(checkpoint) + self._convert_history(post_a2a)
        else:
            history = self._convert_history(
                [m for r_task in related_tasks for m in r_task.history or []]
            )
        history.append(Message(role="user", content=user_input))
        return history

    def _convert_history(self, a2a_history: list[A2AMessage]) -> list[Message]:
        return [
            m
            for msg in a2a_history
            if (m := self._a2a_to_hv.to_history_message(msg)) is not None
        ]
