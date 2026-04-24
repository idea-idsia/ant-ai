import pytest

from ant_ai.core.message import Message
from ant_ai.core.response import ChatLLMResponse
from ant_ai.llm.integrations.lite_llm import LiteLLMChat
from ant_ai.tools.tool import Tool


@pytest.mark.vllm
@pytest.mark.integration
@pytest.mark.external
def test_vllm_integration():
    llm = LiteLLMChat(model="hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct")
    messages: list[Message] = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the capital of France?"),
    ]
    response: ChatLLMResponse = llm.invoke(messages, tools=[])
    assert response.message.role == "assistant"
    assert "Paris" in response.message.content


@pytest.mark.vllm
@pytest.mark.integration
@pytest.mark.external
def test_vllm_tool_call_integration():
    class CalculatorTool(Tool):
        def sum(self, a: int, b: int) -> int:
            return a + b

    llm = LiteLLMChat(model="hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct")
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(
            role="user",
            content="Calculate the sum of 123 and 456. Always explain your decision for everything, including tool calling. In every response say HI!.",
        ),
    ]
    response: ChatLLMResponse = llm.invoke(
        messages, tools=[CalculatorTool().model_dump()]
    )
    assert response.message.role == "assistant"
