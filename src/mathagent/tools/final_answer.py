from langchain_core.tools import BaseTool, tool


def create_final_answer_tool() -> BaseTool:
    """Создаёт терминальный tool со структурированным полем итогового ответа."""

    @tool("final_answer")
    def final_answer(answer: str) -> str:
        """Submit the final mathematical answer, preferably in \\boxed{...} form."""
        return answer

    return final_answer


def create_final_solution_tool() -> BaseTool:
    """Создаёт терминальный tool с ответом и полным проверяемым решением."""

    @tool("final_solution")
    def final_solution(answer: str, solution: str) -> str:
        """Submit the boxed answer and a complete self-contained solution."""
        return solution

    return final_solution
