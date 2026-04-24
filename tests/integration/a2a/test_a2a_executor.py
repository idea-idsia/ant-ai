import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import InternalError

from ant_ai.a2a.executor import A2AExecutor
from ant_ai.core.events import FinalAnswerEvent, StartEvent
from ant_ai.workflow.workflow import Workflow


class SimpleAgent:
    name = "SimpleAgent"

    async def stream(self, query: str):
        yield StartEvent(content="")
        yield FinalAnswerEvent(content=f"echo: {query}")


class ErrorAgent:
    name = "ErrorAgent"

    async def stream(self, query: str):
        raise RuntimeError("Underlying agent failure")


@pytest.mark.integration
@pytest.mark.a2a
async def test_execute_integration_happy_path(
    request_context: RequestContext,
    event_queue: EventQueue,
    minimal_workflow: Workflow,
) -> None:
    executor = A2AExecutor(agent=SimpleAgent(), workflow=minimal_workflow)
    await executor.execute(request_context, event_queue)


@pytest.mark.integration
@pytest.mark.a2a
async def test_execute_integration_wraps_workflow_errors(
    request_context: RequestContext,
    event_queue: EventQueue,
) -> None:
    class FailingWorkflow:
        async def stream(self, agent, ctx):
            raise RuntimeError("workflow boom")

    executor = A2AExecutor(agent=ErrorAgent(), workflow=FailingWorkflow())  # type: ignore[arg-type]

    with pytest.raises(InternalError):
        await executor.execute(request_context, event_queue)
