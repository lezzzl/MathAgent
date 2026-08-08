"""ReAct-луп с инструментами (prompt-based).

Расширяет скелет из loop.py: START → agent → (tools → agent)* → END.
Модель на каждом шаге либо зовёт инструмент (огороженный блок ```<tool> ... ```),
либо выдаёт финальный ответ. Инструменты перечислены в систем-промпте как JSON,
поэтому серверная поддержка function-calling не нужна.

Протокол вызова инструмента модель узнаёт из TOOL_PROTOCOL (см. build_system_prompt).
"""

import re
from pathlib import Path
from typing import Annotated, Any, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from mathagent.agent.tools import TOOLS, Tool, render_tools_block

# Огороженный блок ```<имя_инструмента>\n<тело>``` — тело уходит инструменту
_FENCE = re.compile(r"```([A-Za-z_][A-Za-z0-9_]*)[ \t]*\n(.*?)```", re.S)

TOOL_PROTOCOL = """\
У тебя есть инструменты (перечислены ниже как JSON). Чтобы вызвать инструмент,
ответь ОДНИМ огороженным блоком, где после ``` стоит имя инструмента, а внутри —
его аргумент:

```run_python
# твой код: проверь утверждение на примерах
print(...)
```

После вызова ты получишь строку `Observation (<tool>): ...` с выводом инструмента —
используй её и продолжай. Не выдумывай Observation сам.

Когда решение готово и проверено — ответь БЕЗ блока инструмента и заверши
финальный ответ в \\boxed{...}.

Доступные инструменты:
"""


class ReactState(TypedDict):
    """История сообщений + счётчик шагов (предохранитель от бесконечного лупа)."""

    messages: Annotated[list, add_messages]
    steps: int


def build_system_prompt(persona: str, tools: dict[str, Tool]) -> str:
    """Склеивает роль (persona) + протокол инструментов + их JSON-описание."""
    return f"{persona.strip()}\n\n{TOOL_PROTOCOL}{render_tools_block(tools)}"


def parse_tool_call(text: str, tools: dict[str, Tool]) -> tuple[str, str] | None:
    """Вернуть (имя_инструмента, тело) для ПОСЛЕДНЕГО валидного блока или None."""
    found: tuple[str, str] | None = None
    for match in _FENCE.finditer(text or ""):
        name = match.group(1)
        if name in tools:
            found = (name, match.group(2))
    return found


def build_react_loop(
    model: Any,
    tools: dict[str, Tool],
    max_steps: int = 6,
) -> Any:
    """Собрать ReAct-граф вокруг модели и набора инструментов.

    max_steps ограничивает число обращений к инструментам, чтобы луп гарантированно
    завершался (после лимита переходим в END с тем, что модель успела наработать).
    """

    def agent(state: ReactState) -> dict[str, Any]:
        """Шаг рассуждения: вызвать модель на текущей истории."""
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def route(state: ReactState) -> str:
        """Решить: исполнять инструмент или завершать."""
        if state.get("steps", 0) >= max_steps:
            return "end"
        last = state["messages"][-1]
        return "tools" if parse_tool_call(last.content, tools) else "end"

    def tool_node(state: ReactState) -> dict[str, Any]:
        """Исполнить запрошенный инструмент и вернуть Observation в историю."""
        last = state["messages"][-1]
        call = parse_tool_call(last.content, tools)
        assert call is not None  # route гарантирует наличие вызова
        name, body = call
        observation = tools[name].run(body)
        return {
            "messages": [("human", f"Observation ({name}):\n{observation}")],
            "steps": state.get("steps", 0) + 1,
        }

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")  # вот здесь луп замыкается
    return graph.compile()


def make_test_claim_tool(
    model: Any,
    tester_persona: str,
    max_steps: int = 4,
) -> Tool:
    """Собрать инструмент test_claim, за которым стоит вложенный tester-агент.

    Так солвер↔тестировщик живут в одном графе: солвер зовёт test_claim(утверждение),
    внутри крутится отдельный ReAct-луп тестировщика (только с run_python), и его
    вердикт возвращается солверу как Observation. Рекурсии нет — у тестировщика
    инструмента test_claim нет.
    """
    # тестировщику даём проверочные инструменты, но НЕ test_claim (иначе рекурсия)
    tester_tools = {
        name: tool for name, tool in TOOLS.items() if name != "test_claim"
    }
    tester_system = build_system_prompt(tester_persona, tester_tools)
    tester_app = build_react_loop(model, tester_tools, max_steps=max_steps)

    def run(claim: str) -> str:
        """Прогнать утверждение через тестировщика и вернуть его финальный вердикт."""
        result = tester_app.invoke(
            {
                "messages": [
                    ("system", tester_system),
                    ("human", f"Проверь утверждение:\n{claim.strip()}"),
                ],
                "steps": 0,
            }
        )
        verdict = (result["messages"][-1].content or "").strip()
        return verdict[:2000] if verdict else "(тестировщик не дал вердикт)"

    return Tool(
        name="test_claim",
        description=(
            "Отдать утверждение тестировщику: он проверит его на конкретных примерах "
            "через код и вернёт вердикт PASS (контрпример не найден) или "
            "FAIL с контрпримером. Используй для нетривиальных утверждений, прежде "
            "чем опираться на них в решении."
        ),
        argument="Одно математическое утверждение на естественном языке.",
        run=run,
    )


# --- Адаптер под контракт benchmark_runner (pipeline: react) --------------

def _sum_usage(messages: list[Any]) -> dict[str, Any]:
    """Суммирует token usage по всем ответам модели верхнего уровня.

    Прим.: токены вложенного тестировщика (внутри test_claim) сюда НЕ попадают —
    они расходуются на том же сервере, но живут в под-графе. Для честного учёта
    токенов запускай react без test_claim (use_tester=False)."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        for field in totals:
            totals[field] += int(usage.get(field, 0) or 0)
    return totals


def _transcript(messages: list[Any]) -> str:
    """Собрать читаемую трассу ReAct (без системного промпта) для поля reasoning."""
    lines = []
    for message in messages:
        role = getattr(message, "type", "?")
        if role == "system":
            continue
        lines.append(f"[{role}]\n{getattr(message, 'content', '')}")
    return "\n\n".join(lines)


class _ReactSolver:
    """Обёртка ReAct-лупа под интерфейс graph.invoke({'problem': ...}) раннера."""

    def __init__(self, app: Any, system_prompt: str, prompt_version: str) -> None:
        self._app = app
        self._system = system_prompt
        self._prompt_version = prompt_version

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Прогнать задачу через агента и вернуть поля, которые ждёт solve_item."""
        result = self._app.invoke(
            {
                "messages": [
                    ("system", self._system),
                    ("human", str(inputs["problem"])),
                ],
                "steps": 0,
            }
        )
        messages = result["messages"]
        return {
            "solution": messages[-1].content,
            "reasoning": _transcript(messages[:-1]),
            "usage": _sum_usage(messages),
            "prompt_version": self._prompt_version,
            "trace": {"tool_steps": result.get("steps", 0)},
        }


def create_react_graph(
    model_config: Any,
    prompt_path: Path,
    role_name: str = "solver",
    *,
    max_steps: int = 6,
    use_tester: bool = False,
) -> _ReactSolver:
    """Собрать ReAct-агента с инструментами под контракт benchmark_runner.

    model_config — ModelConfig (как у create_solver_graph). prompt_path — YAML в
    формате react-tools.yml (version + roles[role].system). По умолчанию у солвера
    только run_python (полный учёт токенов); use_tester=True добавляет test_claim.
    """
    from mathagent.agent.vllm_chat import ChatVLLM

    config = yaml.safe_load(Path(prompt_path).read_text(encoding="utf-8"))
    try:
        prompt_version = str(config["version"])
        persona = config["roles"][role_name]["system"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Промпт {prompt_path} не в формате react-tools (нужны version и "
            f"roles.{role_name}.system)"
        ) from exc

    model = ChatVLLM(
        model=model_config.name,
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        temperature=model_config.temperature,
        top_p=model_config.top_p,
        presence_penalty=model_config.presence_penalty,
        seed=model_config.seed,
        timeout=model_config.timeout,
        max_retries=model_config.max_retries,
        extra_body={
            "max_tokens": model_config.max_tokens,
            "top_k": model_config.top_k,
            "min_p": model_config.min_p,
            "repetition_penalty": model_config.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": model_config.thinking},
        },
    )

    tools = dict(TOOLS)
    if use_tester:
        tester_persona = config["roles"]["tester"]["system"]
        tester = make_test_claim_tool(model, tester_persona, max_steps=max_steps)
        tools[tester.name] = tester

    system_prompt = build_system_prompt(persona, tools)
    app = build_react_loop(model, tools, max_steps=max_steps)
    return _ReactSolver(app, system_prompt, prompt_version)
