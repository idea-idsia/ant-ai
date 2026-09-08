from __future__ import annotations

from collections.abc import Awaitable, Callable

from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from ant_ai.core.events import (
    ClarificationNeededEvent,
    CompletedEvent,
    ContentDeltaEvent,
    Event,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ReasoningEvent,
    StartEvent,
    ToolCallingEvent,
    ToolResultEvent,
    UpdateEvent,
)

type Handler = Callable[[Event, Client, str], Awaitable[None]]
type UnboundHandler = Callable[["HVEventToACP", Event, Client, str], Awaitable[None]]


def handler(*event_types: type[Event]):
    """Marks a method as the handler for the given event types."""

    def decorator(fn: UnboundHandler) -> UnboundHandler:
        fn._event_types = event_types  # ty:ignore[unresolved-attribute]
        return fn

    return decorator


def _text_block(text: str) -> TextContentBlock:
    return TextContentBlock(type="text", text=text)


class HVEventToACP:
    """Translates internal HV events into ACP session-update calls.

    Mirrors :class:`ant_ai.a2a.translator.HVEventToA2A` but targets the ACP
    protocol, using semantic ACP types (AgentMessageChunk, AgentThoughtChunk,
    ToolCallStart, ToolCallProgress) instead of a flat text stream.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], Handler] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        for attr_name in dir(self):
            method = getattr(self, attr_name)
            types = getattr(method, "_event_types", None)
            if types:
                for t in types:
                    self._handlers[t] = method

    async def apply(self, event: Event, client: Client, session_id: str) -> None:
        """Dispatch *event* to its registered handler.

        Raises:
            ValueError: If no handler is registered for the event type.
        """
        event_handler = self._handlers.get(type(event))
        if event_handler is None:
            raise ValueError(
                f"No handler registered for event type: {type(event).__name__}"
            )
        await event_handler(event, client, session_id)

    @handler(StartEvent, UpdateEvent, MaxStepsReachedEvent, CompletedEvent)
    async def _noop(self, event: Event, client: Client, session_id: str) -> None:
        pass

    @handler(FinalAnswerEvent, ClarificationNeededEvent)
    async def _agent_message(
        self, event: Event, client: Client, session_id: str
    ) -> None:
        # If the text was already streamed token-by-token as ContentDeltaEvents (stream_id set on the terminal event), re-sending the whole thing would uplicate it in the client; the deltas already delivered the message.
        if getattr(event, "stream_id", None):
            return
        await client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(
                session_update="agent_message_chunk",
                content=_text_block(event.content),
            ),
        )

    @handler(ReasoningEvent)
    async def _agent_thought(
        self, event: Event, client: Client, session_id: str
    ) -> None:
        if getattr(event, "stream_id", None):
            return
        await client.session_update(
            session_id=session_id,
            update=AgentThoughtChunk(
                session_update="agent_thought_chunk",
                content=_text_block(event.content),
            ),
        )

    @handler(ContentDeltaEvent)
    async def _content_delta(
        self, event: Event, client: Client, session_id: str
    ) -> None:
        assert isinstance(event, ContentDeltaEvent)
        if not event.delta:
            return
        if event.target_kind == "reasoning":
            update = AgentThoughtChunk(
                session_update="agent_thought_chunk",
                content=_text_block(event.delta),
            )
        elif event.target_kind == "content":
            update = AgentMessageChunk(
                session_update="agent_message_chunk",
                content=_text_block(event.delta),
            )
        else:
            # tool-calling fragments are surfaced via ToolCallingEvent instead
            return
        await client.session_update(session_id=session_id, update=update)

    @handler(ToolCallingEvent)
    async def _tool_call_start(
        self, event: Event, client: Client, session_id: str
    ) -> None:
        assert isinstance(event, ToolCallingEvent)
        # acp-ui only attaches tool_call updates to an existing assistant message; if none exists yet the update is silently dropped. Send an empty chunk to create the message container before sending ToolCallStart events.
        await client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(
                session_update="agent_message_chunk",
                content=_text_block(""),
            ),
        )
        for tool_call in event.tool_calls:
            await client.session_update(
                session_id=session_id,
                update=ToolCallStart(
                    session_update="tool_call",
                    tool_call_id=tool_call.id,
                    title=tool_call.function.name,
                    status="in_progress",
                ),
            )

    @handler(ToolResultEvent)
    async def _tool_call_done(
        self, event: Event, client: Client, session_id: str
    ) -> None:
        assert isinstance(event, ToolResultEvent)
        await client.session_update(
            session_id=session_id,
            update=ToolCallProgress(
                session_update="tool_call_update",
                tool_call_id=event.tool_call_id,
                status="completed",
                raw_output=event.content or None,
            ),
        )
