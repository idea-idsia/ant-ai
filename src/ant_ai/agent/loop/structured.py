from __future__ import annotations

from typing import TYPE_CHECKING

from ant_ai.core.message import Message
from ant_ai.core.types import State
from ant_ai.steps import LLMStep

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ant_ai.llm.protocol import ChatLLM


def structuring_request(
    llm: ChatLLM, text: str, schema: type[BaseModel]
) -> tuple[LLMStep, State]:
    """The step and state of one LLM call that converts `text` into JSON matching `schema`.

    Uses response_format so constrained decoding guarantees valid output —
    no retry loop, no hook needed.
    """
    step = LLMStep(
        llm=llm,
        system_message=Message(
            role="system",
            content=(
                "Convert the following text into a JSON object. "
                "Respond with valid JSON only, no explanation."
            ),
        ),
        serialized_tools=[],
        response_format=schema,
    )
    state = State()
    state.add_message(Message(role="user", content=text))
    return step, state
