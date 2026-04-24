from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ant_ai.core.events import FinalAnswerEvent, ReasoningEvent, ToolCallingEvent
from ant_ai.core.message import Message, ToolCall, ToolCallMessage
from ant_ai.core.response import ChatLLMResponse
from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    Transition,
    TransitionAction,
)
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.protocol import ChatLLM
from ant_ai.observer import obs


class LLMStep(BaseModel):
    """Invokes the language model and wraps the response in a `StepResult[LLMOutput]`.

    Emits a `ToolCallingEvent` and routes to `"tool"` when the model requests
    tool calls, or emits a `FinalAnswerEvent` and ends the loop otherwise.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "llm"

    llm: SkipValidation[ChatLLM]
    system_message: Message
    serialized_tools: list[dict] = Field(default_factory=list)

    response_format: type[BaseModel] | None = Field(default=None, exclude=True)

    async def run(
        self,
        state: State,
        ctx: InvocationContext | None,
    ):
        llm_input: list[Message] = [self.system_message, *state.messages]

        async with obs.span(
            getattr(self.llm, "model", "llm"),
            as_type="generation",
            model=getattr(self.llm, "model", None),
            input=llm_input,
            metadata={
                "message_count": len(state.messages),
                "tool_count": len(self.serialized_tools),
                "has_response_format": self.response_format is not None,
            },
        ) as span:
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

        output = LLMOutput(raw=raw, tool_calls=tuple(tool_calls))

        reasoning = getattr(response, "reasoning", None)
        if reasoning:
            yield ReasoningEvent(content=reasoning)

        if tool_calls:
            transition = Transition(action=TransitionAction.CONTINUE, next_step="tool")
            event = ToolCallingEvent(
                content=raw,
                message=ToolCallMessage(tool_calls=tool_calls),
            )
        else:
            transition = Transition(action=TransitionAction.END)
            event = FinalAnswerEvent(
                content=raw,
                message=response.message,
            )

        yield event
        yield StepResult(output=output, transition=transition)
