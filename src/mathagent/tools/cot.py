import time
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from mathagent.pipelines.agent_eval.nodes import (
    get_message_reasoning,
    get_message_usage,
    load_prompt_role,
)


def create_cot_tool(model: Any, prompt_path: Path) -> BaseTool:
    """Создаёт LLM-tool для решения одной локальной математической цели."""
    _, role = load_prompt_role(prompt_path, "cot")
    system_prompt = role.get("system")
    task_template = role.get("task")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("CoT prompt role must contain a non-empty system prompt")
    if not isinstance(task_template, str) or not task_template.strip():
        raise ValueError("CoT prompt role must contain a non-empty task prompt")

    @tool("cot", response_format="content_and_artifact")
    def cot_tool(
        goal: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> tuple[str, dict[str, Any]]:
        """Solve one focused mathematical subgoal from the original problem.

        Provide a precise local goal that requires conceptual reasoning, a
        derivation, a case analysis, or a proof step. The tool automatically sees
        the original problem but not the full ReAct history. It returns a concise
        justification and the result established for that goal. It has no access
        to Python or other tools.
        """
        problem = state.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("CoT tool requires the original problem in graph state")

        task_prompt = task_template.replace("{problem}", problem).replace(
            "{goal}", goal
        )
        started = time.perf_counter()
        message = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_prompt),
            ]
        )
        latency = round(time.perf_counter() - started, 3)
        result = message.content.strip() if isinstance(message.content, str) else ""
        success = bool(result)
        artifact = {
            "goal": goal.strip(),
            "result": result,
            "success": success,
            "latency_seconds": latency,
            "usage": get_message_usage(message),
            "reasoning": get_message_reasoning(message),
        }
        if not success:
            return (
                "The CoT subagent returned an empty result. Choose another "
                "valid action.",
                artifact,
            )
        return result, artifact

    return cot_tool
