from __future__ import annotations

from contextvars import Token

from a2a.helpers import get_message_text, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Message as A2AMessage, Role, Task

from ant_ai.a2a.session import current_session_id
from ant_ai.a2a.translator import HVEventToA2A
from ant_ai.agent.agent import Agent
from ant_ai.core.events import Event
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.observer import obs
from ant_ai.workflow.workflow import Workflow


class A2AExecutor(AgentExecutor):
    """
    A2AExecutor is responsible for processing an A2A request.

    It handles the execution the workflow and propages the updates generated in it.
    """

    def __init__(self, agent: Agent, workflow: Workflow):
        """Initialize the A2AExecutor. A2AExecutor is a subclass of AgentExecutor. The AgentExecutor is the a2a-sdk class
        that is responsible for processing the request made to the agent.

        Args:
            agent: Agent that will be used to execute the workflow.
            workflow: Workflow that will be executed.
        """
        self.workflow: Workflow = workflow
        self.agent: Agent = agent
        self._translator: HVEventToA2A = HVEventToA2A()
        self.running_tasks: set[str] = set()

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
            # self.running_tasks.add(task.id)
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
        # """Cancels a task."""
        # task_id: str | None = context.task_id
        # context_id: str | None = context.context_id

        # if not task_id or not context_id:
        #     return

        # if task_id in self.running_tasks:
        #     self.running_tasks.remove(task_id)

        # updater = TaskUpdater(
        #     event_queue=event_queue,
        #     task_id=task_id,
        #     context_id=context_id,
        # )
        # await updater.cancel()
        raise Exception("Task cancel not supported yet.")

    async def _execute(
        self,
        context: RequestContext,
        updater: TaskUpdater,
        task: Task,
    ) -> None:
        ctx = InvocationContext(
            session_id=task.context_id,
            llm_settings=context.metadata.get("llm_settings", None),
            workflow_settings=context.metadata.get("workflow_settings", None),
        )

        a2a_history: list[A2AMessage] = [
            m for r_task in context.related_tasks for m in r_task.history or []
        ]
        history: list[Message] = self._convert_history(a2a_history)
        history.append(Message(role="user", content=context.get_user_input()))

        await obs.event("a2a.history", history_messages=len(history))

        state = State(messages=history)
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
                await self.process_event(event, updater)
        finally:
            current_session_id.reset(token)

        await obs.event("a2a.workflow.end")

    async def process_event(self, event: Event, updater: TaskUpdater) -> None:
        await self._translator.apply(event=event, updater=updater)

    def _convert_history(self, a2a_history: list[A2AMessage]) -> list[Message]:
        return [
            Message(
                role="assistant" if msg.role == Role.ROLE_AGENT else "user",
                content=get_message_text(msg),
                metadata=dict(msg.metadata) if msg.metadata else {},
            )
            for msg in a2a_history
        ]
