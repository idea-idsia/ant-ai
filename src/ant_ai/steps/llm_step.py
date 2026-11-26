from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ant_ai.core.message import Message
from ant_ai.llm.protocol import ChatLLM
from ant_ai.steps.llm_response import StreamedResponse, response_items
from ant_ai.steps.tracing import generation_span

if TYPE_CHECKING:
    from ant_ai.core.message import BaseMessage, ToolCall
    from ant_ai.core.response import ChatLLMResponse
    from ant_ai.core.types import InvocationContext, State


class LLMStep(BaseModel):
    """Invokes the language model and wraps the response in a `StepResult[LLMOutput]`.

    Emits a `ToolCallingEvent` and routes to `"tool"` when the model requests
    tool calls, or emits a `FinalAnswerEvent` and ends the loop otherwise.

    Mirrors `ChatLLM`'s own `ainvoke`/`stream` split: `run()` is the whole-response
    path via `.ainvoke()`, unchanged regardless of streaming; `stream()` is a
    separate method that calls the LLM's `.stream()` API and also emits live
    `ContentDeltaEvent`s as the model generates. Callers (the agent loop)
    choose which method to invoke based on whether live streaming is safe and
    wanted for that call — this class has no streaming flag of its own.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "llm"

    llm: SkipValidation[ChatLLM]
    system_message: Message
    serialized_tools: list[dict] = Field(default_factory=list)

    response_format: type[BaseModel] | None = Field(default=None, exclude=True)

    def _build_llm_input(self, state: State) -> list[BaseMessage]:
        return [self.system_message, *state.messages]

    def _generation_span(self, llm_input: list[BaseMessage], state: State):
        return generation_span(
            self.llm,
            llm_input,
            metadata={
                "message_count": len(state.messages),
                "tool_count": len(self.serialized_tools),
                "has_response_format": self.response_format is not None,
            },
        )

    async def run(
        self,
        state: State,
        ctx: InvocationContext | None,
    ):
        llm_input: list[BaseMessage] = self._build_llm_input(state)

        async with self._generation_span(llm_input, state) as span:
            response: ChatLLMResponse = await self.llm.ainvoke(
                llm_input,
                ctx=ctx,
                tools=self.serialized_tools or None,
                response_format=self.response_format,
            )

            raw: str = response.message.content or ""
            tool_calls: list[ToolCall] = response.tool_calls or []

            update_payload: dict[str, object] = {
                "output": raw,
                "metadata": {
                    "tool_call_count": len(tool_calls),
                },
            }

            response_model = getattr(response, "model", None)
            if response_model is not None:
                update_payload["model"] = response_model

            usage_details = getattr(response, "usage", None)
            if usage_details is not None:
                update_payload["usage"] = usage_details

            span.update(**update_payload)

        reasoning = getattr(response, "reasoning", None)
        for item in response_items(raw, tool_calls, reasoning, stream_id=None):
            yield item

    async def stream(
        self,
        state: State,
        ctx: InvocationContext | None,
    ):
        """Live token-level counterpart to `run()`.

        Calls the LLM's `.stream()` API, yielding `ContentDeltaEvent`s as
        fragments arrive, then the same terminal event/`StepResult` sequence
        as `run()` would produce for the same response, correlated via
        `stream_id`.
        """
        llm_input: list[BaseMessage] = self._build_llm_input(state)
        response = StreamedResponse(stream_id=str(uuid4()))

        async with self._generation_span(llm_input, state) as span:
            async for chunk in self.llm.stream(
                messages=llm_input,
                ctx=ctx,
                tools=self.serialized_tools or None,
                response_format=self.response_format,
            ):
                for event in response.add(chunk):
                    yield event
            tool_calls: list[ToolCall] = response.tool_calls()

            # ChatLLMStreamChunk carries no usage/model fields, unlike
            # ChatLLMResponse, so those span attributes are unavailable here.
            span.update(
                output=response.raw, metadata={"tool_call_count": len(tool_calls)}
            )

        for item in response_items(
            response.raw, tool_calls, response.reasoning, stream_id=response.stream_id
        ):
            yield item
