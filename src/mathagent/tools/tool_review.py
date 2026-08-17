from typing import Literal

from langchain_core.tools import BaseTool, tool


def create_tool_review_tool() -> BaseTool:
    """Создаёт structured output tool для решения precheck-ноды."""

    @tool("tool_review")
    def tool_review(
        decision: Literal["approve", "reject"],
        feedback: str,
    ) -> str:
        """Return the pre-execution review decision and concise feedback."""
        return feedback

    return tool_review
