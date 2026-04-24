from ant_ai.core.result import ClarificationNeededOutput
from ant_ai.tools.tool import Tool


class HumanInputNeededTool(Tool):
    async def ask(self, question: str) -> ClarificationNeededOutput:
        return ClarificationNeededOutput(question=question)
