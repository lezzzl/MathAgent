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

# Модель по привычке пишет ```python вместо ```run_python — принимаем как алиас,
# иначе код «утекает» в финальный ответ и не исполняется.
_FENCE_ALIASES = {"python": "run_python", "py": "run_python"}

TOOL_PROTOCOL = """\
У тебя есть инструменты (перечислены ниже как JSON). Чтобы вызвать инструмент,
ответь ОДНИМ огороженным блоком, где после ``` стоит имя инструмента, а внутри —
его аргумент:

```python
# твой код: проверь утверждение на примерах
print(...)
```

Для исполнения кода подойдёт и ```python, и ```run_python (это один инструмент).
После вызова ты получишь строку `Observation (<tool>): ...` с выводом инструмента —
используй её и продолжай. Не выдумывай Observation сам.

Когда решение готово и проверено — ответь БЕЗ блока инструмента и заверши
финальный ответ в \\boxed{...}.

Доступные инструменты:
"""


class ReactState(TypedDict):
    """История сообщений + счётчик шагов (предохранитель от бесконечного лупа).

    ctx_tokens — реальные prompt-токены, которые уйдут в СЛЕДУЮЩИЙ вызов модели
    (0 = ещё неизвестно, оценим по символам). Ведём его по usage_metadata ответов
    сервера, а не по длине .content: reasoning-модель прячет весь <think> в
    additional_kwargs и возвращает его в контекст — по символам он не виден."""

    messages: Annotated[list, add_messages]
    steps: int
    ctx_tokens: int


def build_system_prompt(persona: str, tools: dict[str, Tool]) -> str:
    """Склеивает роль (persona) + протокол инструментов + их JSON-описание."""
    return f"{persona.strip()}\n\n{TOOL_PROTOCOL}{render_tools_block(tools)}"


def parse_tool_call(text: str, tools: dict[str, Tool]) -> tuple[str, str] | None:
    """Вернуть (имя_инструмента, тело) для ПОСЛЕДНЕГО валидного блока или None."""
    found: tuple[str, str] | None = None
    for match in _FENCE.finditer(text or ""):
        name = _FENCE_ALIASES.get(match.group(1), match.group(1))
        if name in tools:
            found = (name, match.group(2))
    return found


# Запас контекста на разметку чата/спец-токены и погрешность оценки длины.
_CTX_MARGIN = 8192
# Минимум выхода: меньше просить нет смысла (гард не даёт входу разрастись).
_MIN_OUTPUT = 2048


def _message_text(message: Any) -> str:
    """Весь текст сообщения, включая скрытый reasoning из additional_kwargs.

    Важно для оценки размера: reasoning-модель кладёт <think> не в .content, а в
    additional_kwargs (reasoning/reasoning_content), и он уходит обратно в контекст."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, (tuple, list)) and len(message) == 2:
        content = message[1]
    parts = [str(content or "")]
    extra = getattr(message, "additional_kwargs", None) or {}
    for value in extra.values():
        if isinstance(value, str):
            parts.append(value)
    return "".join(parts)


def _estimate_tokens(messages: list[Any]) -> int:
    """ВЕРХНЯЯ оценка длины истории в токенах по символам (~3 симв/токен).

    Запасной путь, когда реальных usage-токенов ещё нет (самый первый вызов).
    Учитывает и скрытый reasoning — иначе недосчитывает половину и max_tokens
    выходит слишком большим (→ 400 переполнения контекста)."""
    return sum(len(_message_text(m)) // 3 + 8 for m in messages)


def _prompt_tokens_after(response: Any, fallback: int) -> int:
    """Реальные prompt-токены для СЛЕДУЮЩЕГО вызова = вход + выход этого ответа.

    Берём из usage_metadata сервера (там учтён и reasoning). fallback — если
    сервер usage не отдал."""
    usage = getattr(response, "usage_metadata", None) or {}
    total = int(usage.get("input_tokens", 0) or 0) + int(
        usage.get("output_tokens", 0) or 0
    )
    return total or fallback


def build_react_loop(
    model: Any,
    tools: dict[str, Tool],
    max_steps: int = 6,
    *,
    context_budget: dict[str, Any] | None = None,
) -> Any:
    """Собрать ReAct-граф вокруг модели и набора инструментов.

    max_steps ограничивает число обращений к инструментам, чтобы луп гарантированно
    завершался (после лимита переходим в END с тем, что модель успела наработать).

    context_budget (опц.): {'max_model_len', 'max_tokens_cap', 'base_extra_body'}.
    Включает ДИНАМИЧЕСКИЙ max_tokens на каждый вызов (вход+выход ≤ контекста, чтобы
    растущий ReAct-диалог не ловил 400) и защиту от зацикливания генераций,
    обрезанных по длине. None → фиксированный max_tokens модели (для вложенного
    тестировщика, где история короткая)."""

    def _invoke(state: ReactState, messages: list[Any]) -> dict[str, Any]:
        """Вызвать модель с max_tokens под остаток контекста; вернуть узловой апдейт.

        Размер входа берём из ctx_tokens (реальные prompt-токены прошлого ответа),
        а для самого первого вызова — из оценки по символам. После ответа обновляем
        ctx_tokens по usage сервера — так учитывается и скрытый reasoning."""
        if context_budget is None:
            return {"messages": [model.invoke(messages)]}
        input_tokens = state.get("ctx_tokens", 0) or _estimate_tokens(messages)
        budget = context_budget["max_model_len"] - input_tokens - _CTX_MARGIN
        dyn = max(_MIN_OUTPUT, min(context_budget["max_tokens_cap"], budget))
        extra = {**context_budget["base_extra_body"], "max_tokens": dyn}
        response = model.bind(extra_body=extra).invoke(messages)
        return {
            "messages": [response],
            "ctx_tokens": _prompt_tokens_after(response, input_tokens + dyn),
        }

    def agent(state: ReactState) -> dict[str, Any]:
        """Шаг рассуждения: вызвать модель на текущей истории."""
        return _invoke(state, state["messages"])

    def route(state: ReactState) -> str:
        """Решить: исполнять инструмент, дожать финал или завершать."""
        last = state["messages"][-1]
        content = last.content or ""
        finish = (getattr(last, "response_metadata", None) or {}).get("finish_reason")
        wants_tool = parse_tool_call(content, tools) is not None
        has_boxed = "\\boxed" in content
        # Генерацию обрезало по длине, а готового \boxed нет → дожимаем финал, но НЕ
        # возвращаем огромный обрезанный текст в контекст новым шагом. Именно это
        # рвало прогон: вход раздувался до >130k токенов → сервер отвечал 400.
        if finish == "length" and not has_boxed:
            return "finalize"
        if state.get("steps", 0) >= max_steps:
            # Шаги кончились. Если модель всё ещё зовёт инструмент (значит финала в
            # \boxed нет) — принудительно дожимаем ответ, а не обрываем пустышкой.
            return "finalize" if wants_tool else "end"
        return "tools" if wants_tool else "end"

    def finalize(state: ReactState) -> dict[str, Any]:
        """Последний шаг: заставить модель выдать \\boxed без вызова инструмента."""
        force = (
            "Лимит вызовов инструментов исчерпан. Больше инструменты недоступны. "
            "Сейчас же дай ОКОНЧАТЕЛЬНЫЙ ответ, оформи его в \\boxed{...}. "
            "Если полностью не уверен — дай лучшую текущую оценку, но \\boxed обязателен."
        )
        # force-реплика добавляет ~40 токенов ко входу — учитываем в бюджете.
        state = {**state, "ctx_tokens": (state.get("ctx_tokens", 0) or 0) + 48}
        return _invoke(state, state["messages"] + [("human", force)])

    def tool_node(state: ReactState) -> dict[str, Any]:
        """Исполнить запрошенный инструмент и вернуть Observation в историю."""
        last = state["messages"][-1]
        call = parse_tool_call(last.content, tools)
        assert call is not None  # route гарантирует наличие вызова
        name, body = call
        observation = tools[name].run(body)
        message = f"Observation ({name}):\n{observation}"
        return {
            "messages": [("human", message)],
            "steps": state.get("steps", 0) + 1,
            # Observation войдёт во вход следующего вызова — доучитываем его размер.
            "ctx_tokens": (state.get("ctx_tokens", 0) or 0) + len(message) // 3 + 8,
        }

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "finalize": "finalize", "end": END}
    )
    graph.add_edge("tools", "agent")  # вот здесь луп замыкается
    graph.add_edge("finalize", END)  # дожатый финал — сразу в конец, без новых тулов
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

    def __init__(
        self, app: Any, system_prompt: str, task_template: str, prompt_version: str
    ) -> None:
        self._app = app
        self._system = system_prompt
        self._task_template = task_template
        self._prompt_version = prompt_version

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Прогнать задачу через агента и вернуть поля, которые ждёт solve_item."""
        # Задача идёт через task-шаблон (как у обычного solver) — там, напр., /think
        task = self._task_template.format(problem=str(inputs["problem"]))
        result = self._app.invoke(
            {
                "messages": [
                    ("system", self._system),
                    ("human", task),
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
    max_model_len: int = 131072,
) -> _ReactSolver:
    """Собрать ReAct-агента с инструментами под контракт benchmark_runner.

    model_config — ModelConfig (как у create_solver_graph). prompt_path — YAML в
    формате react-tools.yml (version + roles[role].system). По умолчанию у солвера
    только run_python (полный учёт токенов); use_tester=True добавляет test_claim.

    max_model_len — окно контекста сервинга: из него на каждом шаге вычитается
    оценка входа и получается динамический max_tokens (чтобы растущий ReAct-диалог
    не переполнял контекст и не ловил 400)."""
    from mathagent.agent.vllm_chat import ChatVLLM

    config = yaml.safe_load(Path(prompt_path).read_text(encoding="utf-8"))
    try:
        prompt_version = str(config["version"])
        persona = config["roles"][role_name]["system"]
        task_template = config["roles"][role_name].get("task", "{problem}")
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

    # База для per-call bind: все sampling-поля, КРОМЕ max_tokens (его на каждом
    # шаге подставляет _invoke под остаток контекста).
    base_extra_body = {
        "top_k": model_config.top_k,
        "min_p": model_config.min_p,
        "repetition_penalty": model_config.repetition_penalty,
        "chat_template_kwargs": {"enable_thinking": model_config.thinking},
    }
    context_budget = {
        "max_model_len": max_model_len,
        "max_tokens_cap": model_config.max_tokens,
        "base_extra_body": base_extra_body,
    }

    tools = dict(TOOLS)
    if use_tester:
        tester_persona = config["roles"]["tester"]["system"]
        tester = make_test_claim_tool(model, tester_persona, max_steps=max_steps)
        tools[tester.name] = tester

    system_prompt = build_system_prompt(persona, tools)
    app = build_react_loop(
        model, tools, max_steps=max_steps, context_budget=context_budget
    )
    return _ReactSolver(app, system_prompt, task_template, prompt_version)
