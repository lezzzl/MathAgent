from langchain_core.tools import BaseTool, tool


def create_final_answer_tool() -> BaseTool:
    """Создаёт терминальный tool со структурированным полем итогового ответа."""

    @tool("final_answer")
    def final_answer(answer: str) -> str:
        """Submit the final mathematical answer, preferably in \\boxed{...} form."""
        return answer

    return final_answer
