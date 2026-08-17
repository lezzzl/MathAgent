from langchain_core.tools import BaseTool, tool


def create_final_answer_tool() -> BaseTool:
    """Создаёт терминальный tool со структурированным полем итогового ответа."""

    @tool("final_answer")
    def final_answer(answer: str) -> str:
        """Submit the final mathematical answer, preferably in \\boxed{...} form."""
        return answer

    return final_answer


def create_contextual_final_answer_tool() -> BaseTool:
    """Создаёт terminal tool с ответом и кратким outline для writer."""

    @tool("final_answer")
    def final_answer(answer: str, solution_context: str) -> str:
        """Submit the answer and a concise, verifiable solution outline."""
        del solution_context
        return answer

    return final_answer


def create_derived_solution_tool() -> BaseTool:
    """Создаёт structured output tool независимого solution writer."""

    @tool("submit_solution")
    def submit_solution(derived_answer: str | None, solution: str) -> str:
        """Submit the independently derived answer and mathematical solution."""
        del derived_answer
        return solution

    return submit_solution


def create_final_solution_tool() -> BaseTool:
    """Создаёт терминальный tool с ответом и полным проверяемым решением."""

    @tool("final_solution")
    def final_solution(answer: str, solution: str) -> str:
        """Submit the boxed answer and a complete self-contained solution."""
        return solution

    return final_solution
