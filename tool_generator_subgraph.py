# tool_generator_subgraph.py
import json
import uuid
from typing import Annotated, List, Optional, Any, Dict, Tuple

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI


from tools import python_exec

TOOLS = [python_exec]
tool_node = ToolNode(TOOLS)


def count_chain_tokens(messages: List[AnyMessage]) -> int:
    """Считает токены одной цепочки сообщений без двойного учёта.

    Наивная сумма total_tokens по всем сообщениям завышает расход в разы:
    каждое следующее AI-сообщение в tool-цикле включает в input_tokens весь
    предыдущий диалог. Корректный расход = input последнего обращения к модели
    (он покрывает всю историю) + все сгенерированные output-токены.
    """
    last_input = 0
    total_output = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        last_input = usage.get("input_tokens", 0) or last_input
        total_output += usage.get("output_tokens", 0)
    return last_input + total_output


def make_llm(temperature, *, model_name, base_url, api_key,
              max_tokens=None, json_format=False, with_tools=True,
              enable_thinking=None):
    """Собирает клиента модели.

    with_tools=False не просто запрещает исполнять тулы, а вообще не отдаёт их
    модели: иначе она продолжает видеть описания инструментов и генерировать
    вызовы, которые некому выполнить.

    enable_thinking=False выключает блок размышлений у thinking-моделей
    (Qwen3.x) через chat_template_kwargs. Замер на оценщике: 420 токенов и 12 с
    против 5277 токенов и 153 с. None — не трогать, для обычных моделей это
    единственный корректный вариант.
    """
    kwargs = {}
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model_name,
                      temperature=temperature, max_tokens=max_tokens, **kwargs)
    if json_format:
        llm = llm.bind(response_format={"type": "json_object"})
    return llm.bind_tools(TOOLS) if with_tools else llm


class GenState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    hops: int
    max_hops: int
    total_tool_calls: int
    max_total_tool_calls: int

import re
import threading

# Общий лок печати: сэмплы self-consistency идут в потоках и без него рвут
# строки друг друга прямо посреди лога tool-вызова.
print_lock = threading.Lock()


def log(message: str) -> None:
    with print_lock:
        print(message)


_TOOL_CALL_LOOKS_LIKE_RE = re.compile(r'(?:<step>)?\s*(?:calculator|python_exec)\s*\(', re.IGNORECASE)

# Блок <tool_call>...</tool_call>, который сервер НЕ распознал и отдал текстом.
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL | re.IGNORECASE)
_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*["\']([\w.]+)["\']')
# Жадный поиск значения "code": захватывает всё до последней кавычки перед
# закрывающими скобками, поэтому переживает кавычки внутри самого кода.
_TOOL_CODE_RE = re.compile(r'["\']code["\']\s*:\s*(["\'])(.*)\1\s*\}', re.DOTALL)


def _unescape_code(raw: str) -> str:
    """Разворачивает escape-последовательности из literal-строки в тексте."""
    out = raw
    for src, dst in (("\\n", "\n"), ("\\t", "\t"), ("\\r", "\r"),
                     ("\\'", "'"), ('\\"', '"'), ("\\\\", "\\")):
        out = out.replace(src, dst)
    return out


def salvage_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Достаёт вызов инструмента из текста, который сервер не смог распарсить.

    Модель регулярно оформляет аргументы не как JSON — например, берёт код в
    одинарные кавычки, да ещё с одинарными кавычками внутри (symbols('R B')).
    Парсер hermes такой блок отвергает и возвращает его сырым текстом в
    content, из-за чего шаг терял вызов инструмента и оценка становилась
    ненадёжной. Здесь мы разбираем блок сами, терпимо к формату.

    Возвращает {"name", "args"} либо None, если вытащить код не удалось
    (например, генерация оборвалась на середине блока).
    """
    if not text:
        return None
    block_match = _TOOL_CALL_BLOCK_RE.search(text)
    if not block_match:
        return None
    block = block_match.group(1)

    # Сначала честный путь: вдруг это валидный JSON и сервер просто не в духе.
    try:
        parsed = json.loads(block, strict=False)
        args = parsed.get("arguments") or parsed.get("args") or {}
        if isinstance(args, str):
            args = json.loads(args, strict=False)
        if isinstance(args, dict) and args.get("code"):
            return {"name": parsed.get("name", "python_exec"), "args": {"code": args["code"]}}
    except Exception:
        pass

    code_match = _TOOL_CODE_RE.search(block)
    if not code_match:
        return None
    code = _unescape_code(code_match.group(2)).strip()
    if not code:
        return None
    name_match = _TOOL_NAME_RE.search(block)
    return {"name": name_match.group(1) if name_match else "python_exec", "args": {"code": code}}

def call_model(state: GenState, config: RunnableConfig):
    llm = config["configurable"]["llm"]
    hops = state.get("hops", 0)
    max_hops = state.get("max_hops", 3)
    if hops >= max_hops - 1 and hasattr(llm, "bind"):
        llm = llm.bind(tool_choice="none")
    ai_msg = llm.invoke(state["messages"])
    # Сервер не распознал вызов инструмента и отдал его текстом — пробуем
    # разобрать сами, иначе шаг молча теряет и вычисление, и оценку.
    if not ai_msg.tool_calls and "<tool_call>" in (ai_msg.content or "").lower():
        salvaged = salvage_tool_call(ai_msg.content)
        if salvaged:
            ai_msg = ai_msg.model_copy(update={
                "content": _TOOL_CALL_BLOCK_RE.sub("", ai_msg.content).strip(),
                "tool_calls": [{
                    "name": salvaged["name"],
                    "args": salvaged["args"],
                    "id": f"salvaged_{hops}_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }],
            })
            log(f"  [TOOL SALVAGE] Сервер не распознал <tool_call>, вызов восстановлен "
                f"клиентом: {salvaged['name']}")
        else:
            log("  [TOOL PARSING BROKEN] В ответе сырой <tool_call>, восстановить не "
                "удалось (вероятно, генерация оборвана лимитом токенов). "
                "Увеличьте num_predict для этой роли.")
    elif not ai_msg.tool_calls and _TOOL_CALL_LOOKS_LIKE_RE.search(ai_msg.content or ""):
        log("  [TOOL PARSING BROKEN] Модель написала вызов тула текстом, "
            "но сервер не распознал его как tool_call. Проверьте, что vLLM запущен с "
            "--enable-auto-tool-choice --tool-call-parser hermes.")
    for tc in ai_msg.tool_calls or []:
        code = str(tc["args"].get("code", tc["args"]))
        preview = " ".join(code.split())[:160]
        log(f"      [tool_call] {tc['name']}: {preview}{'...' if len(code) > 160 else ''}")
    return {"messages": [ai_msg], "hops": state.get("hops", 0) + 1}


def route_tools(state: GenState):
    last = state["messages"][-1]
    max_total = state.get("max_total_tool_calls", 8)

    already_over = state.get("total_tool_calls", 0) >= max_total
    if already_over:
        # Предыдущий forced-final ход должен был обойтись без tool_calls;
        # если модель всё равно их выдала — не исполняем, просто завершаем.
        if getattr(last, "tool_calls", None):
            print("  ⚠️  [TOOL BURST] Лимит tool calls исчерпан, модель всё "
                  "равно запросила ещё — завершаем без исполнения.")
        return "end"


    if state.get("total_tool_calls", 0) >= max_total:
        print(f"  ⚠️  [TOOL BURST] Модель запросила {state['total_tool_calls']} tool calls "
              f"суммарно — это подозрительно похоже на застревание/перебор вслепую. "
              f"Останавливаем цикл тулов.")
        return "end"

    if state.get("hops", 0) >= state.get("max_hops", 3):
        return "end"
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


_subgraph = StateGraph(GenState)
_subgraph.add_node("agent", call_model)
_subgraph.add_node("tools", tool_node)
_subgraph.set_entry_point("agent")
_subgraph.add_conditional_edges("agent", route_tools, {"tools": "tools", "end": END})
_subgraph.add_edge("tools", "agent")
generator_with_tools = _subgraph.compile()