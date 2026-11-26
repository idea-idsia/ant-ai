from __future__ import annotations

from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Task

from ant_ai.a2a.server.history import HistoryBuilder, without_message
from ant_ai.a2a.server.observability import observe_cancel, observe_execution
from ant_ai.a2a.server.runner import forward_events
from ant_ai.a2a.server.translator import HVEventToA2A
from ant_ai.agent.agent import Agent
from ant_ai.core.events import BaseEvent
from ant_ai.core.message import BaseMessage
from ant_ai.core.types import InvocationContext, State
from ant_ai.workflow.workflow import Workflow


class A2AExecutor(AgentExecutor):
    """
    A2AExecutor is responsible for processing an A2A request.

    It handles the execution the workflow and propages the updates generated in it.
    """

    def __init__(
        self,
        agent: Agent,
        workflow: Workflow,
        *,
        stream_artifacts: bool = True,
        context_class: type[InvocationContext] = InvocationContext,
    ):
        """Initialize the A2AExecutor. A2AExecutor is a subclass of AgentExecutor. The AgentExecutor is the a2a-sdk class
        that is responsible for processing the request made to the agent.

        Args:
            agent: Agent that will be used to execute the workflow.
            workflow: Workflow that will be executed.
            stream_artifacts: Whether to translate ContentDeltaEvent into A2A
                artifact-update chunks. See `HVEventToA2A`.
            context_class: The `InvocationContext` (sub)class built for each
                request from the message metadata via `from_metadata`.
        """
        self.workflow: Workflow = workflow
        self.agent: Agent = agent
        self.context_class: type[InvocationContext] = context_class
        self._translator: HVEventToA2A = HVEventToA2A(stream_artifacts=stream_artifacts)
        self._history: HistoryBuilder = HistoryBuilder()

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
        async with observe_execution(self.agent.name, task, context.metadata):
            await self._execute(context, updater, task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task by publishing the canceled status.

        Stopping the coroutine is the SDK's job: `DefaultRequestHandler.on_cancel_task`
        calls this and then cancels the producer task itself. Raising here (as this
        method used to) aborted the handler before it got that far, so `tasks/cancel`
        could not stop anything.

        What the handler needs from us is the canceled status: `consume_all` refuses
        any other final state, and a canceled status update is a final event, which
        is what closes the queue the cancelled producer never gets to close.
        """
        task: Task | None = context.current_task
        if task is None:
            raise Exception("No task to cancel.")

        await observe_cancel(task)
        await TaskUpdater(event_queue, task.id, task.context_id).cancel()

    async def _execute(
        self,
        context: RequestContext,
        updater: TaskUpdater,
        task: Task,
    ) -> None:
        ctx: InvocationContext = self.context_class.from_metadata(
            session_id=task.context_id, metadata=context.metadata
        )

        related_tasks: list[Task] = list(context.related_tasks)
        if context.current_task is not None and context.message is not None:
            # The message continues a task that waited on input: its own
            # messages (the request and the question) come right before it.
            related_tasks.append(
                without_message(context.current_task, context.message.message_id)
            )

        history: list[BaseMessage] = await self._history.build(
            related_tasks, context.get_user_input()
        )
        state: State = self.workflow.create_state(messages=history)
        await forward_events(
            self.workflow.stream(agent=self.agent, ctx=ctx, state=state),
            session_id=ctx.session_id,
            state=state,
            updater=updater,
            apply=self.process_event,
        )

    async def process_event(self, event: BaseEvent, updater: TaskUpdater) -> None:
        await self._translator.apply(event=event, updater=updater)
