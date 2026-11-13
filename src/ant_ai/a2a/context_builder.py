from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import override

from a2a.server.agent_execution import (
    RequestContext,
    RequestContextBuilder,
)
from a2a.server.context import ServerCallContext
from a2a.server.id_generator import IDGenerator
from a2a.server.tasks import TaskStore
from a2a.types import SendMessageRequest, Task

from ant_ai.observer import obs


class HistoryRequestContextBuilder(RequestContextBuilder):
    def __init__(
        self,
        should_populate_referred_tasks: bool = True,
        task_store: TaskStore | None = None,
        task_id_generator: IDGenerator | None = None,
        context_id_generator: IDGenerator | None = None,
    ) -> None:
        """Initializes the HistoryRequestContextBuilder.

        Args:
            should_populate_referred_tasks: If True (should always be True), the builder will fetch tasks
                referenced in `params.message.reference_task_ids` and populate the
                `related_tasks` field in the RequestContext. Kept to match interface of SimpleRequestContextBuilder. Defaults to True.
            task_store: The TaskStore instance to use for fetching referred tasks.
                Required if `should_populate_referred_tasks` is True.
            task_id_generator: ID generator for new task IDs. Defaults to None.
            context_id_generator: ID generator for new context IDs. Defaults to None.
        """
        self._task_store: TaskStore | None = task_store
        self._should_populate_referred_tasks: bool = should_populate_referred_tasks
        self._task_id_generator: IDGenerator | None = task_id_generator
        self._context_id_generator: IDGenerator | None = context_id_generator

    @override
    async def build(
        self,
        context: ServerCallContext,
        params: SendMessageRequest | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
    ) -> RequestContext:
        """Builds the request context for an agent execution.
        This method assembles the RequestContext object.

        Args:
            params: The parameters of the incoming message send request.
            task_id: The ID of the task being executed.
            context_id: The ID of the current execution context.
            task: The primary task object associated with the request.
            context: The server call context, containing metadata about the call.

        Returns:
            An instance of RequestContext populated with the provided information
            and potentially a list of related tasks.
        """

        related_tasks: list[Task] | None = None

        reference_ids: list[str] = (
            list(params.message.reference_task_ids) if params else []
        )
        # A message that continues a task (one waiting on input) does not
        # reference the earlier turns itself: the task's own messages do.
        # Some request handlers pass only the id of that task, not the task.
        continued: Task | None = task
        if continued is None and task_id and self._task_store:
            continued = await self._task_store.get(task_id, context=context)
        if continued is not None:
            reference_ids += [
                tid for msg in continued.history for tid in msg.reference_task_ids
            ]

        if self._task_store and self._should_populate_referred_tasks and reference_ids:
            related_tasks: list[Task] = await self.collect_all_referenced_tasks(
                context,
                self._task_store,
                reference_ids,
                # The task being continued is not related to itself.
                visited={continued.id} if continued is not None else None,
            )

        return RequestContext(
            call_context=context,
            request=params,
            task_id=task_id,
            context_id=context_id,
            task=task,
            related_tasks=related_tasks,
            task_id_generator=self._task_id_generator,
            context_id_generator=self._context_id_generator,
        )

    async def collect_all_referenced_tasks(
        self,
        context: ServerCallContext,
        task_store: TaskStore,
        initial_ids: Sequence[str],
        visited: set[str] | None = None,
    ) -> list[Task]:
        """
        Recursively collects all tasks referenced by the given list of task IDs.
        This method performs a breadth-first search through the task graph, starting from the initial task IDs,
        and collects all tasks that are directly or indirectly referenced.

        Args:
            context: The server call context containing user information and other relevant
            task_store: The task store to retrieve tasks from given their IDs.
            initial_ids: The list of initial task IDs to start the search from.
            visited: A set of task IDs that have already been visited during the search to avoid cycles.
                This is used internally during the recursion and should not be provided by the caller.

        Returns:
            A list of Task objects that are referenced by the initial task IDs, including the tasks corresponding to the initial IDs themselves.
        """

        visited: set[str] = set() if visited is None else visited

        tasks: list[Task] = []
        queue: list[str] = [tid for tid in initial_ids if tid not in visited]

        while queue:
            batch: list[str] = list(dict.fromkeys(queue))
            queue: list[str] = []

            results: list[Task | None] = await asyncio.gather(
                *(task_store.get(tid, context=context) for tid in batch),
            )

            for tid, item in zip(batch, results, strict=True):
                if not isinstance(item, Task):
                    await obs.event(
                        "a2a.referenced_task.unresolved",
                        task_id=tid,
                    )
                    continue

                if item.id in visited:
                    continue

                visited.add(item.id)
                tasks.append(item)

                for msg in item.history or ():
                    queue.extend(msg.reference_task_ids or ())

            queue: list[str] = [tid for tid in queue if tid not in visited]

        return tasks[::-1]  # reverse to maintain original order
