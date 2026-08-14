import json
import operator
import os
import random
import re
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, NamedTuple, Optional, Tuple

import requests
import yaml
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from answer_utils import extract_answer, iter_boxed
from trajectory import RECORDER


# ---------------------------------------------------------------------------
# 0. Роли и загрузка промптов
# ---------------------------------------------------------------------------
@dataclass
class Role:
    name: str
    system_prompt: str
    user_template: str = "{context}"
    temperature: float = 0.6
    json_format: bool = False
    num_predict: Optional[int] = None
    enable_thinking: Optional[bool] = None

    def build_messages(self, **kwargs) -> List[dict]:
        """Собирает messages из system_prompt + user_template.

        Лишние kwargs, которых нет в шаблоне, игнорируются — так один вызов
        подходит и ролям, которым нужен только {context}, и тем, кому нужны
        ещё {step} или {raw}.
        """
        try:
            user_content = self.user_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Роль '{self.name}': user_template ссылается на плейсхолдер {e}, "
                f"которого нет среди переданных аргументов ({sorted(kwargs)}). "
                f"Проверьте user_template в agent-step-qwen4b-v1.yml для этой роли."
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]


_FALLBACK_SYSTEM = (
    "You are a careful mathematician. Follow the output format requested by the "
    "user message exactly. Prompts failed to load from YAML — results will be poor."
)

_DEFAULT_ROLE_DEFS: Dict[str, Dict[str, Any]] = {
    "generator": {
        "system": _FALLBACK_SYSTEM,
        "user_template": "{context}",
        "temperature": 0.6,
        "json_format": False,
        "num_predict": 24000,
        "enable_thinking": True,
    },
    "segmenter": {
        "system": _FALLBACK_SYSTEM,
        "user_template": "{context}\n\nRaw draft:\n{raw}",
        "temperature": 0.2,
        "json_format": False,
        "num_predict": 4000,
        "enable_thinking": False,
    },
    "evaluator": {
        "system": _FALLBACK_SYSTEM,
        "user_template": "{context}\nGive a score of the new step:\n{step}",
        "temperature": 0.6,
        "json_format": False,
        "num_predict": 8000,
        "enable_thinking": False,
    },
    "verifier": {
        "system": _FALLBACK_SYSTEM,
        "user_template": "{context}",
        "temperature": 0.6,
        "json_format": False,
        "num_predict": 16000,
        "enable_thinking": True,
    },
}

ROLES: Dict[str, Role] = {
    name: Role(
        name=name,
        system_prompt=cfg["system"],
        user_template=cfg["user_template"],
        temperature=cfg["temperature"],
        json_format=cfg["json_format"],
        num_predict=cfg["num_predict"],
        enable_thinking=cfg.get("enable_thinking"),
    )
    for name, cfg in _DEFAULT_ROLE_DEFS.items()
}


def _num_predict_overrides() -> Dict[str, int]:
    """Разбирает ROLE_NUM_PREDICT="generator=55000,verifier=30000".
    """
    out: Dict[str, int] = {}
    for chunk in ROLE_NUM_PREDICT_ENV.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        role, _, value = chunk.partition("=")
        try:
            out[role.strip()] = int(value)
        except ValueError:
            print(f"[Prompts] ROLE_NUM_PREDICT: не разобрал '{chunk}', пропускаю.")
    return out


def load_prompts_from_yaml(yaml_path: "Path | str") -> None:
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        roles_cfg = config.get("roles", {})
        if not roles_cfg:
            print(f"[Prompts] Warning: {yaml_path} has no 'roles' section. Using hardcoded defaults.")
            return

        for role_name, role_cfg in roles_cfg.items():
            if role_name not in _DEFAULT_ROLE_DEFS:
                print(f"[Prompts] Warning: unknown role '{role_name}' in {yaml_path}, ignoring.")
                continue
            defaults = _DEFAULT_ROLE_DEFS[role_name]
            num_predict = role_cfg.get("num_predict", defaults["num_predict"])
            override = _num_predict_overrides().get(role_name)
            if override:
                print(f"[Prompts] {role_name}: num_predict {num_predict} -> {override} "
                      f"(ROLE_NUM_PREDICT)")
                num_predict = override
            ROLES[role_name] = Role(
                name=role_name,
                system_prompt=role_cfg.get("system", defaults["system"]),
                user_template=role_cfg.get("user_template", defaults["user_template"]),
                temperature=float(role_cfg.get("temperature", defaults["temperature"])),
                json_format=bool(role_cfg.get("json_format", defaults["json_format"])),
                num_predict=num_predict,
                enable_thinking=role_cfg.get("enable_thinking", defaults.get("enable_thinking")),
            )

        print(f"[Prompts] Successfully loaded prompts from {yaml_path}")
    except Exception as e:
        print(f"[Prompts] ОШИБКА: промпты не загрузились из {yaml_path} ({e}).\n"
              f"           Работаем на аварийной заглушке — качество будет мусорным.")


# ---------------------------------------------------------------------------
# 1. HTTP-клиент модели (без инструментов)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL", "Qwen/Qwen3-4B")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "40000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))
MAX_STEP_CHARS = int(os.getenv("MAX_STEP_CHARS", "1500"))
SEGMENTER_INPUT_CHARS = int(os.getenv("SEGMENTER_INPUT_CHARS", "8000"))
CHAT_RETRIES = int(os.getenv("CHAT_RETRIES", "1"))
CHAT_RETRY_BACKOFF = float(os.getenv("CHAT_RETRY_BACKOFF", "5"))
EVAL_REASK = (
    "Your previous reply did not contain a parseable verdict. Reply now with "
    "ONLY the two sections, nothing before or after:\n"
    "###SCORE###\n0.0 or 1.0\n###RATIONALE###\none sentence."
)
EVAL_REASK_NUM_PREDICT = int(os.getenv("EVAL_REASK_NUM_PREDICT", "1000"))
ROLE_NUM_PREDICT_ENV = os.getenv("ROLE_NUM_PREDICT", "")
SEED: Optional[int] = None
_seed_state = threading.local()


def begin_task_seed(task_id: Any) -> None:
    """Начинает новую задачу: сбрасывает счётчик вызовов для зерна."""
    if SEED is None:
        _seed_state.base = None
        return
    _seed_state.base = (SEED * 1_000_003 + zlib.crc32(str(task_id).encode())) % (2 ** 31 - 1)
    _seed_state.counter = 0


def _next_seed() -> Optional[int]:
    base = getattr(_seed_state, "base", None)
    if base is None:
        return None
    _seed_state.counter = getattr(_seed_state, "counter", 0) + 1
    return (base + _seed_state.counter * 7919) % (2 ** 31 - 1)


# ---------------------------------------------------------------------------
# Режим оценщика (эксперимент: нужен ли он вообще)
# ---------------------------------------------------------------------------
# "llm"    — обычный оценщик, роль evaluator из yaml;
# "random" — модель не зовём, вердикт бросаем монетой.
#
# Зачем. Оценщик стоит заметной доли бюджета и отвергает больше половины
# кандидатов, но что он даёт качеству — никогда не измерялось. Случайный
# оценщик отвечает на это прямо: если счёт бенчмарка не изменится, значит
# вердикт не нёс информации и весь этот расход был впустую.
#
# Шкала БИНАРНАЯ — 0.0 или 1.0. Промежуточных значений у qwen4b нет: промпт
# роли требует «ONLY as 0.0 or 1.0», и по всем прогонам (золотому и четырём
# августовским) в eval_history не встречается ни одного другого значения.
# Градуированная шкала 0/0.25/0.5/0.75/1.0 живёт только в линейке 9B.
#
# Доля нулей подобрана под наблюдаемую у живого оценщика (52% в прогоне на
# vLLM, 59-63% в прогонах на SGLang). Без этого сравнивались бы не «оценка
# против случайности», а «строгий судья против мягкого», и разница в счёте
# ничего не сказала бы про качество вердикта.
EVALUATOR_MODE = "llm"
RANDOM_EVAL_REJECT_RATE = float(os.getenv("RANDOM_EVAL_REJECT_RATE", "0.6"))

# Глубина, начиная с которой оценщик вообще включается. 0 — как всегда.
#
# Зачем. Замер по двум базовым прогонам: 83% всех оценок и 87% всех отвержений
# приходятся на глубины 0-1, и отвергает он там заметно строже (69% и 66%)
# против 49-50% на глубине 2-3. Вопрос: эта ранняя строгость — работа или шум.
# С min_depth=2 шаги на глубинах 0-1 принимаются без вызова модели; если счёт
# не упадёт, значит ранние отвержения ничего не давали.
EVALUATOR_MIN_DEPTH = 0


def _random_eval_score() -> float:
    """Вердикт случайного оценщика. Воспроизводим: зерно из той же цепочки,
    что и у обращений к модели (seed + task_id + номер вызова)."""
    return 0.0 if random.Random(_next_seed()).random() < RANDOM_EVAL_REJECT_RATE else 1.0


# ---------------------------------------------------------------------------
# Инструмент python_exec поверх OpenAI-API
# ---------------------------------------------------------------------------
# Модуль намеренно не тянет langchain (в отличие от 9B-пайплайна), поэтому цикл
# вызова инструментов реализован напрямую на /chat/completions: описываем
# функцию в поле tools, читаем tool_calls из ответа, исполняем и возвращаем
# результат сообщением роли "tool".
TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Execute Python for exact mathematics and print the result. Preloaded: "
            "sympy (solve, Eq, Rational, symbols, factorint, isprime, divisors, "
            "binomial, simplify, expand, factor, Matrix, primerange), numpy as np, "
            "scipy, itertools, math, Fraction. Sympy names shadow math, so sqrt(8) "
            "stays exact. Write multi-line code and print() what you need. Prefer "
            "exact types (Rational, sqrt) over floats — float rounding silently "
            "breaks point deduplication in geometry enumerations. Execution is "
            "capped at 30 seconds. Each call starts from a clean namespace: "
            "nothing from a previous call survives, so declare symbols again in "
            "every script. If a script prints results and then crashes, the reply "
            "is marked PARTIAL RESULT — the printed values are still valid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."}
            },
            "required": ["code"],
        },
    },
}]

# Потолки на один вызов роли: сколько раз возвращаемся к модели после
# инструмента и сколько исполнений разрешено суммарно. Без них модель на
# трудной задаче уходит в бесконечный перебор вслепую.
MAX_TOOL_HOPS = int(os.getenv("MAX_TOOL_HOPS", "5"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))

# Отдельные, более жёсткие лимиты для оценщика: ему нужно проверить один шаг,
# а не решить задачу. Замер на aime24 показал, что при 3/4 стадия evaluate
# раздувается с 21.2% до 31.5% бюджета. Настраиваются независимо от генератора,
# чтобы ужимать проверку, не ослабляя решение.
EVAL_TOOL_HOPS = int(os.getenv("EVAL_TOOL_HOPS", "2"))
EVAL_TOOL_CALLS = int(os.getenv("EVAL_TOOL_CALLS", "2"))

# Потолок расхода на ОДИН вызов роли, считая все витки с инструментом.
# Замер на aime24/25/26: медиана вызова генератора 21.8k токенов, p90 82k, а
# максимум — 220k при бюджете задачи 600k. Из-за этого 8 задач из 90 (все
# «нет ответа») перебрали бюджет на 4k–229k: раунд стартовал, когда бюджет был
# почти исчерпан, и один цикл тулов уводил далеко за лимит. Кратность к
# num_predict, а не константа, чтобы правило не зависело от лимита роли.
TOOL_LOOP_TOKEN_FACTOR = float(os.getenv("TOOL_LOOP_TOKEN_FACTOR", "2.5"))


def _run_tool(name: str, args: Dict[str, Any]) -> str:
    """Исполняет инструмент и возвращает текст результата для модели."""
    if name != "python_exec":
        return f"ERROR: unknown tool {name!r}."
    from tools import python_exec  # ленивый импорт: песочница поднимается не всегда
    try:
        return python_exec.func(args.get("code", ""))
    except Exception as exc:  # noqa: BLE001 — ошибка инструмента не валит шаг
        return f"ERROR: tool raised {type(exc).__name__}: {exc}"


class ChatResult(NamedTuple):
    content: str
    reasoning: str
    tokens: int
    finish_reason: Optional[str]
    error: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @property
    def text(self) -> str:
        """content, а при его пустоте — reasoning_content.

        Reasoning-парсер сервера кладёт размышления в reasoning_content, а сам
        ответ — в content. Обычно нам нужен content; reasoning — только фоллбек
        на случай, когда сервер не отдал content отдельно.
        """
        return (self.content or "").strip() or (self.reasoning or "").strip()


def _chat(messages, temperature=0.2, seed=None, num_predict=None, json_format=False,
          enable_thinking=None, *, stage="chat", depth=None, branch=None,
          record_extra=None, tools=None, tool_choice=None) -> ChatResult:
    """Один вызов модели. Всё, что здесь происходит, попадает в траекторию:
    промпт, сырой ответ, размышления, finish_reason, токены и время."""
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": num_predict if num_predict is not None else DEFAULT_MAX_TOKENS,
    }
    seed = seed if seed is not None else _next_seed()
    if seed is not None:
        payload["seed"] = seed
    if json_format:
        payload["response_format"] = {"type": "json_object"}
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    system_text = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_text = next((m["content"] for m in messages if m.get("role") == "user"), "")

    def _emit(result: ChatResult, elapsed: float, usage: Optional[dict] = None) -> ChatResult:
        usage = usage or {}
        RECORDER.record(
            stage=stage, depth=depth, branch=branch,
            system=system_text, user=user_text,
            content=result.content, reasoning=result.reasoning,
            finish_reason=result.finish_reason,
            tokens={
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
                "total": result.tokens,
            },
            elapsed=elapsed, error=result.error,
            temperature=temperature, num_predict=num_predict,
            enable_thinking=enable_thinking, model=MODEL_NAME,
            **(record_extra or {}),
        )
        return result

    last_error: Optional[str] = None
    started = time.perf_counter()
    for attempt in range(CHAT_RETRIES + 1):
        response = None
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices")
            if not choices:
                print(f"[WARN] No choices in response: {data}")
                return _emit(ChatResult("", "", 0, None), time.perf_counter() - started)

            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content", "") or ""
            reasoning = message.get("reasoning_content", "") or message.get("reasoning", "") or ""
            # Ход, состоящий только из tool_calls, законно приходит с пустым
            # content — это не сбой, и предупреждать о нём не нужно.
            if not content and not reasoning and not message.get("tool_calls"):
                print(f"[WARN] Empty content and reasoning in response: {data}")

            usage = data.get("usage", {}) or {}
            tokens_used = usage.get("total_tokens", 0)
            finish_reason = choice.get("finish_reason")
            # Обрыв по лимиту больше не молчит: раньше он был заметен, только
            # если content пуст целиком, и частичные обрывы никак не всплывали.
            if finish_reason == "length":
                print(f"      ⚠️  [TRUNCATED] {stage}: finish_reason=length при лимите "
                      f"{num_predict} — ответ оборван{' (и пуст)' if not content.strip() else ''}.")
            calls = message.get("tool_calls") or None
            return _emit(ChatResult(content, reasoning, tokens_used, finish_reason,
                                    None, calls),
                         time.perf_counter() - started, usage)

        except requests.exceptions.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < CHAT_RETRIES:
                print(f"  [API RETRY] {last_error} — попытка "
                      f"{attempt + 2}/{CHAT_RETRIES + 1} через {CHAT_RETRY_BACKOFF:.0f}с")
                time.sleep(CHAT_RETRY_BACKOFF)
                continue
            print(f"API Error (после {attempt + 1} попыт.): {e}")
            if response is not None:
                print(f"Response content: {response.text}")

    return _emit(ChatResult("", "", 0, None, last_error), time.perf_counter() - started)


_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>|<\|tool_call\|>|<function=", re.IGNORECASE)

# Нативный формат Qwen3.x: <function=name><parameter=code>...</parameter></function>.
_QWEN_FN_RE = re.compile(r"<function=([\w.]+)\s*>(.*?)(?:</function>|$)", re.DOTALL | re.IGNORECASE)
_QWEN_PARAM_RE = re.compile(r"<parameter=([\w.]+)\s*>(.*?)(?:</parameter>|$)", re.DOTALL | re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL | re.IGNORECASE)


def _salvage_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Достаёт вызовы инструмента из текста, который сервер не распарсил.

    Поддерживает два формата: нативный Qwen3.x (<function=…><parameter=…>) и
    JSON внутри <tool_call>. Клиентский разбор снимает зависимость от того,
    какой --tool-call-parser поднят на сервере: hermes для Qwen3.x не работает.
    """
    if not text:
        return []
    calls: List[Dict[str, Any]] = []

    for i, m in enumerate(_QWEN_FN_RE.finditer(text)):
        name, body = m.group(1), m.group(2)
        args = {k: v.strip() for k, v in _QWEN_PARAM_RE.findall(body)}
        if args.get("code"):
            calls.append({
                "id": f"salvaged_{i}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
    if calls:
        return calls

    for i, m in enumerate(_JSON_BLOCK_RE.finditer(text)):
        try:
            parsed = json.loads(m.group(1), strict=False)
        except Exception:
            continue
        args = parsed.get("arguments") or parsed.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args, strict=False)
            except Exception:
                args = {"code": args}
        if isinstance(args, dict) and args.get("code"):
            calls.append({
                "id": f"salvaged_json_{i}", "type": "function",
                "function": {"name": parsed.get("name", "python_exec"),
                             "arguments": json.dumps(args)},
            })
    return calls


def _strip_tool_call_text(text: str) -> str:
    """Убирает разобранный вручную блок вызова из текста ответа."""
    cleaned = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text or "",
                     flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<function=.*?(?:</function>|$)", "", cleaned,
                     flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _chat_with_tools(messages: List[dict], *, temperature, num_predict, enable_thinking,
                     stage, depth=None, branch=None,
                     max_hops=None, max_calls=None) -> Tuple[ChatResult, int, int]:
    """Диалог с моделью, в котором она может вызывать python_exec.

    Возвращает (ответ, токены, число исполнений, число спасённых вручную вызовов).
    На последнем витке инструменты отключаются (tool_choice="none"), иначе модель
    может закончить ход вызовом и не выдать сам шаг.
    """
    max_hops = MAX_TOOL_HOPS if max_hops is None else max_hops
    max_calls = MAX_TOOL_CALLS if max_calls is None else max_calls
    convo = list(messages)
    total_tokens = 0
    n_calls = 0
    n_salvaged = 0
    result = None

    token_cap = int((num_predict or DEFAULT_MAX_TOKENS) * TOOL_LOOP_TOKEN_FACTOR)
    for hop in range(max_hops):
        over_budget = total_tokens >= token_cap
        if over_budget and hop:
            print(f"      [TOOL BUDGET] Вызов роли израсходовал {total_tokens} токенов "
                  f"(потолок {token_cap}) — завершаю без новых обращений к инструменту.")
        last_hop = hop == max_hops - 1 or n_calls >= max_calls or over_budget
        result = _chat(
            convo, temperature=temperature, num_predict=num_predict,
            enable_thinking=enable_thinking, stage=stage, depth=depth, branch=branch,
            tools=TOOL_SCHEMA, tool_choice="none" if last_hop else None,
            record_extra={"hop": hop, "tool_calls_so_far": n_calls},
        )
        total_tokens += result.tokens
        if result.error:
            break

        calls = result.tool_calls
        assistant_content = result.content or ""
        if not calls and not last_hop:
            calls = _salvage_tool_calls(assistant_content)
            if calls:
                n_salvaged += len(calls)
                assistant_content = _strip_tool_call_text(assistant_content)
                print(f"      [TOOL SALVAGE] Сервер не распознал вызов "
                      f"(формат Qwen3.x вместо JSON) — разобрано клиентом: {len(calls)}. "
                      f"Для нативного разбора нужен подходящий --tool-call-parser.")
        if not calls:
            break

        convo.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": calls,
        })
        for call in calls:
            if n_calls >= max_calls:
                convo.append({
                    "role": "tool", "tool_call_id": call.get("id", ""),
                    "content": "ERROR: tool call budget for this step is exhausted. "
                               "Answer with what you already have.",
                })
                continue
            fn = call.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {"code": str(raw_args)}
            out = _run_tool(name, args)
            n_calls += 1
            code_preview = " ".join(str(args.get("code", "")).split())[:120]
            print(f"      [tool_call #{n_calls}] {name}: {code_preview}")
            print(f"      [tool_result] {' '.join(out.split())[:160]}")
            RECORDER.record(
                stage="tool", depth=depth, branch=branch,
                user=str(args.get("code", "")), content=out, tool_name=name, hop=hop,
            )
            convo.append({
                "role": "tool", "tool_call_id": call.get("id", ""), "content": out,
            })

    if result is None:
        result = ChatResult("", "", 0, None, "no response")
    if _TEXT_TOOL_CALL_RE.search(result.text or ""):
        result = result._replace(content=_strip_tool_call_text(result.text), reasoning="")
    return result, total_tokens, n_calls, n_salvaged


# ---------------------------------------------------------------------------
# 2. Извлечение шага и контекст
# ---------------------------------------------------------------------------
_STEP_TAG_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL | re.IGNORECASE)
_STEP_OPEN_TAG_RE = re.compile(r"<step>", re.IGNORECASE)
_STEP_ANY_TAG_RE = re.compile(r"</?step\s*>", re.IGNORECASE)
_BOXED_START_RE = re.compile(r"\\boxed\s*\{")


def _extract_step_content(raw: str) -> str:
    """Извлекает содержимое <step>...</step> (быстрый путь, без вызова модели).

    Если модель вынесла \\boxed{ за пределы тегов — приклеивает хвост обратно к
    шагу. Это тот же экстрактор, что и в оригинале: он остаётся первой линией
    обороны, а сегментатор подключается только когда тегов нет или в них
    втиснуто слишком много.
    """
    if not raw:
        return ""

    closed = list(_STEP_TAG_RE.finditer(raw))
    if closed:
        step = closed[-1].group(1).strip()
    else:
        opens = list(_STEP_OPEN_TAG_RE.finditer(raw))
        step = raw[opens[-1].end():].strip() if opens else raw.strip()

    matches = list(_BOXED_START_RE.finditer(raw))
    if matches:
        last_match = matches[-1]
        if last_match.group(0) not in step:
            tail = raw[last_match.start():]
            clean_tail = _STEP_ANY_TAG_RE.split(tail)[0].strip()
            if step:
                step += f"\n\nFinal answer: {clean_tail}"
            else:
                step = f"Final answer: {clean_tail}"

    return _STEP_ANY_TAG_RE.sub("", step).strip()


# Дословные куски системного промпта генератора: модель их копирует в шаг вместе
# с образцом «\boxed{...}». Такой шаг выглядит валидным (теги на месте, короткий,
# один \boxed), проскакивал быстрым путём, и extract_answer доставал из него '...'.
_PROMPT_ECHO_RE = re.compile(
    r"ONLY if the accepted steps|nothing remains but to report|"
    r"giving a final answer is FORBIDDEN|Dumping the full solution|"
    r"a later stage will clean up|WHAT COUNTS AS ONE STEP|"
    r"WHEN THE FINAL ANSWER IS ALLOWED|Prefer exact forms",
    re.IGNORECASE,
)
# Следы незавершённых размышлений: модель рассуждает вслух прямо внутри <step>.
_THINKING_NOISE_RE = re.compile(
    r"\bWait,|\bHmm\b|\bOkay,|Let me (?:check|verify|think|try)|"
    r"I should check|I need to justify|\bActually,",
    re.IGNORECASE,
)


def _has_clean_single_step(raw: str) -> bool:
    """Можно ли доверять быстрому пути без вызова сегментатора.

    Тегов и длины мало: на прогоне aime24 через быстрый путь прошли 16 шагов с
    размышлениями вслух и 9 с дословными кусками промпта — их обязан чистить
    сегментатор, а он не вызывался. Поэтому дополнительно требуем, чтобы в шаге
    не было эха промпта, следов размышлений и заглушки \\boxed{...}.
    """
    if not _STEP_TAG_RE.search(raw or ""):
        return False
    step = _extract_step_content(raw)
    if not step or len(step) > MAX_STEP_CHARS:
        return False
    if len(list(iter_boxed(step))) > 1:
        return False
    if _PROMPT_ECHO_RE.search(step) or _THINKING_NOISE_RE.search(step):
        return False
    return True


_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_TRIM_NOTE = "[... ранние размышления опущены ...]\n"


def _trim_for_segmenter(raw: str) -> str:
    """Оставляет от черновика тот хвост, в котором действительно лежит шаг.

    Порядок стратегий — от самой надёжной к запасной:
      1. после последнего </think> — там ровно «чистовик» модели;
      2. от последнего <step> — если теги есть, но не прошли быстрый путь;
      3. просто хвост нужной длины.
    Начало черновика (перебор гипотез, самокритика) сегментатору не нужно и
    стоило пятой части бюджета прогона.
    """
    raw = raw or ""
    if len(raw) <= SEGMENTER_INPUT_CHARS:
        return raw

    closes = list(_THINK_CLOSE_RE.finditer(raw))
    if closes:
        tail = raw[closes[-1].end():].strip()
        if tail:
            return (_TRIM_NOTE + tail[-SEGMENTER_INPUT_CHARS:]) if len(tail) > SEGMENTER_INPUT_CHARS else _TRIM_NOTE + tail

    opens = list(_STEP_OPEN_TAG_RE.finditer(raw))
    if opens:
        tail = raw[opens[-1].start():]
        if len(tail) <= SEGMENTER_INPUT_CHARS:
            return _TRIM_NOTE + tail

    return _TRIM_NOTE + raw[-SEGMENTER_INPUT_CHARS:]


def _build_context(problem: str, steps: List[str]) -> str:
    """Собирает контекст, отбрасывая самые ранние шаги при переполнении."""
    if not steps:
        return f"Task: {problem}\n"

    head = f"Task: {problem}\n\nCurrent steps of solution:\n"
    rendered = [f"Step {i}: {step}\n" for i, step in enumerate(steps, 1)]

    budget = MAX_CONTEXT_CHARS - len(head)
    dropped = 0
    while len(rendered) > 1 and sum(len(s) for s in rendered) > budget:
        rendered.pop(0)
        dropped += 1

    if dropped:
        print(f"  [CONTEXT TRIM] Отброшено {dropped} ранних шагов, чтобы уложиться "
              f"в {MAX_CONTEXT_CHARS} символов контекста.")
        head += f"[... {dropped} earlier step(s) omitted for brevity ...]\n"

    tail = "".join(rendered)
    if len(tail) > budget:
        tail = tail[: max(0, budget)] + "\n[... step truncated ...]\n"

    return head + tail


_STEP_PREFIX_RE = re.compile(r"^(step\s*\d+\s*:\s*)+", re.IGNORECASE)


def _normalize_step_text(text: str) -> str:
    stripped = _STEP_PREFIX_RE.sub("", (text or "").strip())
    return re.sub(r"\s+", " ", stripped.lower()).strip()


# ---------------------------------------------------------------------------
# 3. Парсер вывода сегментатора (разделители, НЕ JSON)
# ---------------------------------------------------------------------------
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SEG_STEP_RE = re.compile(r"###STEP###\s*(.*?)\s*(?:###ANSWER###|###END###|$)",
                          re.DOTALL | re.IGNORECASE)
_SEG_ANSWER_RE = re.compile(r"###ANSWER###\s*(.*?)\s*(?:###END###|$)",
                            re.DOTALL | re.IGNORECASE)


def _reattach_answer(step: str, answer: Optional[str]) -> str:
    """Гарантирует, что \\boxed{answer} присутствует в шаге, чтобы его увидел
    extract_answer (и роутер после commit ушёл на верификацию)."""
    if not answer:
        return step
    if extract_answer(step):
        return step
    boxed = answer if "\\boxed" in answer else f"\\boxed{{{answer}}}"
    return f"{step}\n\nFinal answer: {boxed}".strip() if step else f"Final answer: {boxed}"


def _parse_segmenter_response(content: str, raw_fallback: str,
                              allow_answer: bool = True) -> Tuple[str, Optional[str], bool]:
    """Разбирает ответ сегментатора формата ###STEP### / ###ANSWER###.

    Разделители вместо JSON выбраны намеренно: вывод сегментатора — сплошной
    LaTeX (\\frac, \\sqrt, \\boxed), а именно он ломает JSON-парсинг из-за
    неэкранированных бэкслэшей. Строчные маркеры такой проблемы лишены.

    allow_answer=False запрещает приклеивать финальный ответ к шагу. Нужно на
    малой глубине: qwen4b решает задачу целиком, сегментатор исправно возвращает
    первый шаг, но заодно рапортует найденный в черновике \\boxed — и пайплайн
    схлопывается в один шаг (де-факто обычный CoT). См. min_steps_before_answer.

    Возвращает (step_text, final_answer|None, reliable). При отсутствии маркеров
    деградируем к обычному экстрактору по <step> — сначала на самом ответе
    сегментатора, затем на сыром черновике генератора.
    """
    if not content:
        return _extract_step_content(raw_fallback), None, False

    text = _THINK_BLOCK_RE.sub("", content)
    m = _SEG_STEP_RE.search(text)
    if not m:
        # Маркеры не пришли — не теряем шаг, чистим чем есть.
        fallback = _extract_step_content(text) or _extract_step_content(raw_fallback)
        return fallback, (extract_answer(fallback) if allow_answer else None), False

    step = _STEP_ANY_TAG_RE.sub("", m.group(1)).strip()

    answer: Optional[str] = None
    a = _SEG_ANSWER_RE.search(text)
    if a:
        cand = a.group(1).strip()
        if cand and cand.upper() != "NONE":
            answer = extract_answer(cand) or cand

    if not allow_answer:
        # Шаг оставляем как есть (математику не портим), но ответ не приклеиваем.
        return step, None, True

    step = _reattach_answer(step, answer)
    return step, answer, True


# ---------------------------------------------------------------------------
# 4. Парсеры ответов оценщика и верификатора (JSON с лечением LaTeX)
# ---------------------------------------------------------------------------
_MD_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _iter_json_objects(text: str):
    """Перебирает сбалансированные {...} с конца текста к началу.

    Reasoning-модели кладут JSON в самый конец, а в начале полно латеховых
    \\frac{a}{b}. Идём с конца — нужный объект там.
    """
    opens = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(opens):
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break


def _extract_json_dict(content: str, expected_keys: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    cleaned = _THINK_BLOCK_RE.sub("", content)

    candidates = [m.group(1) for m in _MD_JSON_RE.finditer(cleaned)]
    candidates.extend(_iter_json_objects(cleaned))

    for candidate in candidates:
        for attempt in (candidate, re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", candidate)):
            try:
                parsed = json.loads(attempt, strict=False)
            except Exception:
                continue
            if isinstance(parsed, dict) and any(k in parsed for k in expected_keys):
                return parsed
    return None


# Между маркером и значением допускаем «обёртку»: модель копирует угловые
# скобки из плейсхолдера промпта и пишет «###VALID###\n<true>» вместо «true».
# Именно на этом терялся вердикт верификатора (aime26 task 30): регулярка
# натыкалась на «<» и вердикт уходил в «не распарсилось».
_WRAP = r"""[\s<\[\("'*`]*"""
_SEG_SCORE_RE = re.compile(rf"###SCORE###{_WRAP}([0-9]*\.?[0-9]+)", re.IGNORECASE)
_SEG_VALID_RE = re.compile(rf"###VALID###{_WRAP}(true|false|yes|no)", re.IGNORECASE)
# Хвостовой псевдотег вида </true> после обоснования — тот же скопированный
# плейсхолдер, в текст вердикта он попадать не должен.
_TRAILING_PSEUDOTAG_RE = re.compile(r"\s*</[^>\n]{1,40}>\s*$")
_SEG_RATIONALE_MARK_RE = re.compile(r"###RATIONALE###", re.IGNORECASE)
_SEG_END_RE = re.compile(r"###END###", re.IGNORECASE)


def _parse_delimited(content: str) -> Optional[Tuple[Optional[float], Optional[bool], str]]:
    """Разбирает формат ###SCORE###/###VALID### + ###RATIONALE###.

    Зачем не JSON: обоснование оценщика — сплошной LaTeX (\\angle, \\frac,
    \\cap), а бэкслеш внутри строки JSON невалиден. На прогоне aime24 14%
    ответов (17 из 123) не парсились как JSON и спасались регуляркой, которая
    режет rationale по первой кавычке. Со строчными маркерами этой проблемы нет
    — ровно та же причина, по которой их использует сегментатор.
    """
    if not content:
        return None
    text = _THINK_BLOCK_RE.sub("", content)
    score_all = list(_SEG_SCORE_RE.finditer(text))
    valid_all = list(_SEG_VALID_RE.finditer(text))
    if not score_all and not valid_all:
        return None
    score_m = score_all[-1] if score_all else None
    valid_m = valid_all[-1] if valid_all else None
    rat_marks = list(_SEG_RATIONALE_MARK_RE.finditer(text))
    if rat_marks:
        tail = text[rat_marks[-1].end():]
        end = _SEG_END_RE.search(tail)
        rationale = (tail[: end.start()] if end else tail).strip()
        rationale = _TRAILING_PSEUDOTAG_RE.sub("", rationale)
    else:
        rationale = "No rationale provided"
    score = None
    if score_m:
        try:
            score = max(0.0, min(1.0, float(score_m.group(1))))
        except ValueError:
            score = None
    valid = None
    if valid_m:
        valid = valid_m.group(1).lower() in ("true", "yes")
    return score, valid, rationale


def _parse_eval_response(content: str) -> Tuple[float, str, bool]:
    """Разбирает ответ оценщика. Возвращает (score, rationale, is_reliable)."""
    if not content:
        return 0.0, "Empty response from evaluator", False

    # Основной путь — разделители (LaTeX-безопасны). JSON ниже оставлен, чтобы
    # модуль продолжал понимать старые промпты и прогоны.
    delim = _parse_delimited(content)
    if delim is not None and delim[0] is not None:
        return delim[0], delim[2], True

    parsed = _extract_json_dict(content, ("score", "rationale"))
    if parsed is not None and "score" in parsed:
        try:
            score = max(0.0, min(1.0, float(parsed["score"])))
            rationale = str(parsed.get("rationale", "No rationale provided"))
            return score, rationale, True
        except (TypeError, ValueError):
            pass

    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
    json_str = md_match.group(1) if md_match else content
    if not md_match:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)

    try:
        result_dict = json.loads(json_str, strict=False)
        score = max(0.0, min(1.0, float(result_dict.get("score", 0.0))))
        rationale = str(result_dict.get("rationale", "No rationale provided"))
        return score, rationale, True
    except Exception as e_first:
        try:
            repaired_str = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', json_str)
            result_dict = json.loads(repaired_str, strict=False)
            score = max(0.0, min(1.0, float(result_dict.get("score", 0.0))))
            rationale = str(result_dict.get("rationale", "No rationale provided"))
            return score, rationale, True
        except Exception:
            pass

    score_match = re.search(r'"score"\s*:\s*([0-1](?:\.[0-9]+)?)', content, re.IGNORECASE)
    rat_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
    if score_match:
        try:
            score = max(0.0, min(1.0, float(score_match.group(1))))
            rationale = rat_match.group(1) if rat_match else f"Extracted via regex (JSON parse failed: {e_first})"
            print(f"  ℹ️ [JSON RECOVERY] Парсер спас оценку score={score:.4f} регуляркой!")
            return score, rationale, True
        except ValueError:
            pass

    return 0.0, f"Invalid JSON from evaluator (raw preview: {content[:400]!r})", False


def _parse_verify_response(content: str) -> Tuple[bool, str, bool]:
    """Аналогично _parse_eval_response, но для схемы верификатора."""
    if not content:
        return False, "Empty response from verifier", False

    delim = _parse_delimited(content)
    if delim is not None and delim[1] is not None:
        return delim[1], delim[2], True

    parsed = _extract_json_dict(content, ("is_valid", "rationale"))
    if parsed is not None and "is_valid" in parsed:
        return (
            bool(parsed["is_valid"]),
            str(parsed.get("rationale", "No rationale provided")),
            True,
        )

    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
    json_str = md_match.group(1) if md_match else content
    if not md_match:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)

    try:
        result_dict = json.loads(json_str, strict=False)
        is_valid = bool(result_dict.get("is_valid", False))
        rationale = str(result_dict.get("rationale", "No rationale provided"))
        return is_valid, rationale, True
    except Exception as e_first:
        try:
            repaired_str = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', json_str)
            result_dict = json.loads(repaired_str, strict=False)
            is_valid = bool(result_dict.get("is_valid", False))
            rationale = str(result_dict.get("rationale", "No rationale provided"))
            return is_valid, rationale, True
        except Exception:
            pass

    valid_match = re.search(r'"is_valid"\s*:\s*(true|false)', content, re.IGNORECASE)
    rat_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
    if valid_match:
        is_valid = (valid_match.group(1).lower() == "true")
        rationale = rat_match.group(1) if rat_match else f"Extracted via regex (JSON parse failed: {e_first})"
        print(f"  ℹ️ [JSON RECOVERY] Верификатор спас вердикт is_valid={is_valid} регуляркой!")
        return is_valid, rationale, True

    return False, f"Invalid JSON from verifier (raw preview: {content[:400]!r})", False


# ---------------------------------------------------------------------------
# 5. State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    problem: str
    steps: Annotated[List[str], operator.add]

    # Сырые генерации до сегментации и очищенные кандидаты после неё.
    candidate_raw: List[str]
    candidate_steps: List[str]
    candidate_scores: List[float]

    k_branches: int
    score_threshold: float
    branch_mode: str
    base_temperature: Optional[float]

    tokens_used: Annotated[int, operator.add]
    token_budget: int
    in_recovery: bool
    max_recoveries: int
    total_recovery_events: Annotated[int, operator.add]
    # Попытка собрать ответ из принятых шагов делается ровно один раз за задачу,
    # иначе роутеры зациклятся на finish -> give_up -> finish.
    finish_attempted: bool

    stuck_streak: int
    max_stuck_steps: int
    skip_verifier: bool
    max_step_attempts: int
    finish_at_fraction: float

    unreliable_eval_streak: int
    max_unreliable_evals: int
    eval_history: Annotated[List[Dict[str, Any]], operator.add]
    thinking_overruns: Annotated[int, operator.add]
    # Сколько раз пришлось звать LLM-сегментатор (быстрый путь по тегам не сработал).
    segmenter_calls: Annotated[int, operator.add]
    # Сколько сегментаций вернулись без распознанных маркеров (фоллбек-экстракция).
    segmenter_unreliable: Annotated[int, operator.add]
    # Сетевые сбои (таймаут/обрыв) за задачу. Ненулевое значение означает, что
    # часть шагов и оценок потеряна по вине инфраструктуры, а не модели.
    api_errors: Annotated[int, operator.add]

    # Разрешено ли генератору звать python_exec (флаг --no-tools выключает).
    use_tools: bool
    # Сколько раз инструмент реально исполнялся за задачу.
    tool_calls: Annotated[int, operator.add]
    # Сколько вызовов пришлось разбирать клиентом (сервер не распознал формат).
    # Ненулевое значение = на сервере неподходящий --tool-call-parser.
    tool_salvaged: Annotated[int, operator.add]

    # Минимум уже принятых шагов, прежде чем финальный ответ будет засчитан.
    # 1 = запрещаем ответ на глубине 0 (ровно то, что декларируют промпты).
    # 0 = прежнее поведение: ответ принимается сразу, пайплайн вырождается в CoT.
    min_steps_before_answer: int
    # Сколько раз ответ был отвергнут как преждевременный.
    premature_answers: Annotated[int, operator.add]
    # ПАССИВНЫЙ замер: сколько раз боксированное значение не встречалось в тексте
    # шага до самого бокса. Ни на что не влияет, нужен для статистики по режиму
    # «вывели одно число, забоксили другое».
    answers_not_in_step_text: Annotated[int, operator.add]
    # Глубина (число ранее принятых шагов), на которой ответ всё-таки засчитан.
    answer_depth: Optional[int]

    final_answer: Optional[str]
    is_valid: bool
    verifier_rationale: str
    gave_up: bool
    gave_up_reason: str

    step_recovery_attempts: int
    max_step_attempts: int


# ---------------------------------------------------------------------------
# 6. Узлы графа
# ---------------------------------------------------------------------------
def _generate_one(role: Role, context: str, temp: float,
                  depth: Optional[int] = None, branch: Optional[int] = None,
                  use_tools: bool = False
                  ) -> Tuple[str, bool, int, bool, int, int]:
    """Один вызов генератора с откатом при обрыве размышлений.

    У thinking-моделей размышления идут в тот же num_predict, что и ответ. Если
    их не хватило, сервер возвращает finish_reason='length' и пустой content —
    шаг теряется целиком. Повторяем один раз с выключенными размышлениями:
    лучше шаг без ризонинга, чем пустой шаг.

    Возвращает (raw_content, overran, total_tokens, api_failed, tool_calls, salvaged).
    """
    messages = [
        {"role": "system", "content": role.system_prompt},
        {"role": "user", "content": role.user_template.format(context=context)},
    ]

    if use_tools:
        result, tokens, n_calls, n_salv = _chat_with_tools(
            messages, temperature=temp, num_predict=role.num_predict,
            enable_thinking=role.enable_thinking, stage="generate",
            depth=depth, branch=branch)
        return result.text, False, tokens, bool(result.error), n_calls, n_salv

    result = _chat(messages, temperature=temp, num_predict=role.num_predict,
                   json_format=role.json_format, enable_thinking=role.enable_thinking,
                   stage="generate", depth=depth, branch=branch)

    truncated_empty = (not result.content.strip()) and result.finish_reason == "length"
    if truncated_empty and role.enable_thinking:
        print(f"      [THINKING OVERRUN] Размышления съели весь лимит "
              f"({role.num_predict} токенов), ответ пуст. Повтор без размышлений.")
        retry = _chat(messages, temperature=temp, num_predict=role.num_predict,
                      json_format=role.json_format, enable_thinking=False,
                      stage="generate", depth=depth, branch=branch,
                      record_extra={"thinking_retry": True})
        # Учитываем и токены оборванной первой попытки, и токены повтора.
        return retry.text, True, result.tokens + retry.tokens, bool(retry.error), 0, 0
    return result.text, False, result.tokens, bool(result.error), 0, 0


def generate_step(state: AgentState):
    current_depth = len(state.get('steps', []))
    print(f"\n[Node: Generate] Depth: {current_depth} | Tokens used: {state.get('tokens_used', 0)}")
    multi = state.get('branch_mode') == 'multi' or state.get('in_recovery')
    k = state.get('k_branches', 3) if multi else 1

    remaining = state.get('token_budget', 10**9) - state.get('tokens_used', 0)
    role_cost = int((ROLES["generator"].num_predict or DEFAULT_MAX_TOKENS)
                    * (TOOL_LOOP_TOKEN_FACTOR if state.get("use_tools") else 1.0))
    if role_cost > 0 and remaining < role_cost * k:
        affordable = max(1, remaining // role_cost)
        if affordable < k:
            print(f"  ⚠️  [BUDGET] Осталось {remaining} токенов, одна ветка стоит до "
                  f"{role_cost} — сокращаю {k} -> {affordable} ветк(и).")
            k = affordable
    print(f"  -> Generating {k} candidate(s).")

    role = ROLES["generator"]
    raw_candidates: List[str] = []
    total_tokens = 0
    overruns = 0
    api_errors = 0
    tool_calls = 0
    tool_salvaged = 0
    context = _build_context(state['problem'], state.get('steps', []))
    use_tools = bool(state.get("use_tools"))

    base_temp = state.get('base_temperature')
    if base_temp is None:
        base_temp = role.temperature

    for i in range(k):
        attempt = state.get('step_recovery_attempts', 0)
        temp = min(base_temp + 0.15 * i + 0.1 * attempt, 1.1)
        raw_text, overran, tks, api_failed, n_calls, n_salv = _generate_one(
            role, context, temp, depth=current_depth, branch=i + 1, use_tools=use_tools)
        overruns += overran
        api_errors += api_failed
        tool_calls += n_calls
        tool_salvaged += n_salv
        total_tokens += tks
        raw_candidates.append(raw_text)
        preview = re.sub(r"\s+", " ", raw_text).strip()[:200]
        fail_note = " ⚠️ [СЕТЕВОЙ СБОЙ, ветка потеряна]" if api_failed else ""
        tool_note = f", tool-вызовов: {n_calls}" if n_calls else ""
        print(f"    - Branch {i+1} raw generated (temp: {temp:.2f}, tokens: {tks}{tool_note}){fail_note}")
        print(f"      Raw preview: {preview}{'...' if len(raw_text) > 200 else ''}")

    return {"candidate_raw": raw_candidates, "tokens_used": total_tokens,
            "thinking_overruns": overruns, "api_errors": api_errors,
            "tool_calls": tool_calls, "tool_salvaged": tool_salvaged}


def segment_step(state: AgentState):
    """НОВАЯ стадия: из каждого сырого ответа генератора вырезает ровно один шаг.

    Быстрый путь: если в сыром ответе уже есть аккуратный одиночный <step>, берём
    его без вызова модели. Иначе зовём роль segmenter (размышления выключены),
    которая возвращает чистый шаг и, если он есть, финальный ответ.
    """
    raw_candidates = state.get('candidate_raw') or []
    depth = len(state.get('steps', []))
    min_before = state.get('min_steps_before_answer', 0)
    allow_answer = depth >= min_before
    print(f"\n[Node: Segment] Сегментирую {len(raw_candidates)} сырых кандидата(ов) "
          f"(глубина {depth}, ответ {'разрешён' if allow_answer else 'ЗАПРЕЩЁН'})...")
    role = ROLES["segmenter"]
    context = _build_context(state['problem'], state.get('steps', []))

    steps: List[str] = []
    total_tokens = 0
    seg_calls = 0
    seg_unreliable = 0
    api_errors = 0

    for i, raw in enumerate(raw_candidates):
        if not (raw or "").strip():
            steps.append("")
            print(f"    - Candidate {i+1}: пустая генерация, пропускаю сегментацию.")
            continue

        fast_path = _has_clean_single_step(raw)
        if fast_path:
            step = _extract_step_content(raw)
            reliable, answer = True, None
            print(f"    - Candidate {i+1}: чистый <step> найден — быстрый путь без сегментатора.")
        else:
            seg_calls += 1
            trimmed = _trim_for_segmenter(raw)
            if len(trimmed) < len(raw):
                print(f"      [SEGMENT TRIM] Черновик {len(raw)} -> {len(trimmed)} симв.")
            messages = [
                {"role": "system", "content": role.system_prompt},
                {"role": "user", "content": role.user_template.format(context=context,
                                                                      raw=trimmed)},
            ]
            res = _chat(messages, temperature=role.temperature, num_predict=role.num_predict,
                        json_format=role.json_format, enable_thinking=role.enable_thinking,
                        stage="segment", depth=depth, branch=i + 1)
            total_tokens += res.tokens
            api_errors += bool(res.error)
            step, answer, reliable = _parse_segmenter_response(res.text, raw,
                                                              allow_answer=allow_answer)
            seg_unreliable += (not reliable)
            tag = "" if reliable else " ⚠️ [маркеры не распознаны, фоллбек-экстракция]"
            ans_note = f" | answer={answer}" if answer else ""
            print(f"    - Candidate {i+1}: сегментатор вернул шаг ({res.tokens} ток.){ans_note}{tag}")

        # Итог сегментации ветки: именно этот текст уйдёт оценщику.
        RECORDER.record(
            stage="segment_result", depth=depth, branch=i + 1,
            content=step, fast_path=fast_path, reliable=reliable,
            answer=answer, answer_allowed=allow_answer,
        )
        steps.append(step)
        print(f"      Step:\n{step}\n")

    return {"candidate_steps": steps, "tokens_used": total_tokens,
            "segmenter_calls": seg_calls, "segmenter_unreliable": seg_unreliable,
            "api_errors": api_errors}


def evaluate_steps(state: AgentState):
    candidates = state['candidate_steps']
    print(f"\n[Node: Evaluate] Checking {len(candidates)} candidate(s)...")
    role = ROLES["evaluator"]
    scores: List[float] = []
    seen: Dict[str, Tuple[float, str]] = {}
    total_tokens = 0
    any_reliable = False
    api_errors = 0
    tool_calls = 0
    tool_salvaged = 0
    use_tools = bool(state.get("use_tools"))
    context = _build_context(state['problem'], state.get('steps', []))

    for i, step in enumerate(candidates):
        key = _normalize_step_text(step)

        if not key:
            score = 0.0
            rationale = "Step is entirely empty. Generator/segmenter produced nothing usable."
            seen[key] = (score, rationale)
            scores.append(score)
            print(f"    - Candidate {i+1} Score: {score:.4f} | Rationale: {rationale}")
            continue
        if key in seen:
            score, rationale = seen[key]
            scores.append(score)
            print(f"    - Candidate {i+1} Score: {score:.4f} | Rationale: {rationale} "
                  f"♻️ [ДУБЛИКАТ шага, оценщик повторно не вызывался]")
            continue

        # Мелкая глубина: принимаем не спрашивая. Пустой шаг сюда не попадёт —
        # он отсеян выше, иначе агент коммитил бы пустоту.
        if len(state.get('steps', [])) < EVALUATOR_MIN_DEPTH:
            score = 1.0
            rationale = (f"ПРИНЯТО БЕЗ ОЦЕНКИ: глубина "
                         f"{len(state.get('steps', []))} < {EVALUATOR_MIN_DEPTH}")
            any_reliable = True
            seen[key] = (score, rationale)
            scores.append(score)
            RECORDER.record(
                stage="evaluate_result", depth=len(state.get('steps', [])), branch=i + 1,
                content=rationale, score=score, reliable=True, step_text=step,
            )
            print(f"    - Candidate {i+1} Score: {score:.4f} | {rationale}")
            continue

        # Контрольный режим: балл вместо вердикта модели. Пустой шаг выше всё
        # равно получает 0 — это не суждение о качестве, а защита от коммита
        # пустоты, и без неё сравнивались бы разные пайплайны, а не разные
        # оценщики. Всё остальное ниже по графу не меняется.
        if EVALUATOR_MODE == "random":
            score = _random_eval_score()
            rationale = f"СЛУЧАЙНЫЙ ОЦЕНЩИК (контрольный режим), балл {score:.2f}"
            any_reliable = True
            seen[key] = (score, rationale)
            scores.append(score)
            RECORDER.record(
                stage="evaluate_result", depth=len(state.get('steps', [])), branch=i + 1,
                content=rationale, score=score, reliable=True, step_text=step,
            )
            print(f"    - Candidate {i+1} Score: {score:.4f} | {rationale}")
            continue

        messages = [
            {"role": "system", "content": role.system_prompt},
            {"role": "user", "content": role.user_template.format(context=context, step=step)},
        ]
        depth_now = len(state.get('steps', []))
        if use_tools:
            res, tks, n_calls, n_salv = _chat_with_tools(
                messages, temperature=role.temperature, num_predict=role.num_predict,
                enable_thinking=role.enable_thinking, stage="evaluate",
                depth=depth_now, branch=i + 1,
                max_hops=EVAL_TOOL_HOPS, max_calls=EVAL_TOOL_CALLS)
            tool_calls += n_calls
            tool_salvaged += n_salv
        else:
            res = _chat(messages, temperature=role.temperature, num_predict=role.num_predict,
                        json_format=role.json_format, enable_thinking=role.enable_thinking,
                        stage="evaluate", depth=depth_now, branch=i + 1)
            tks = res.tokens
        total_tokens += tks
        api_errors += bool(res.error)

        score, rationale, reliable = _parse_eval_response(res.text)

        if not reliable and not res.error:
            print(f"    - Candidate {i+1}: вердикт не разобран — переспрашиваю "
                  f"без инструментов.")
            retry = _chat(
                messages + [{"role": "user", "content": EVAL_REASK}],
                temperature=role.temperature, num_predict=EVAL_REASK_NUM_PREDICT,
                json_format=False, enable_thinking=False,
                stage="evaluate", depth=depth_now, branch=i + 1,
                record_extra={"reask": True},
            )
            total_tokens += retry.tokens
            api_errors += bool(retry.error)
            r_score, r_rationale, r_reliable = _parse_eval_response(retry.text)
            if r_reliable:
                score, rationale, reliable = r_score, r_rationale, r_reliable
                print(f"    - Candidate {i+1}: переспрос дал {score:.1f}.")

        if res.error:
            # Иначе сетевой сбой выглядит как «оценщик выдал мусор».
            rationale = f"СЕТЕВОЙ СБОЙ ({res.error}) — оценка не получена, не вина модели."
        any_reliable = any_reliable or reliable
        seen[key] = (score, rationale)
        scores.append(score)

        # Разобранный вердикт отдельно от сырого ответа: виден и балл, и почему.
        RECORDER.record(
            stage="evaluate_result", depth=depth_now, branch=i + 1,
            content=rationale, score=score, reliable=reliable, step_text=step,
        )

        tag = "" if reliable else (" ⚠️ [СЕТЕВОЙ СБОЙ]" if res.error else " ⚠️ [ОЦЕНКА НЕНАДЁЖНА]")
        print(f"    - Candidate {i+1} Score: {score:.4f}{tag} | Rationale: {rationale}")

    unreliable_streak = 0 if any_reliable else state.get('unreliable_eval_streak', 0) + 1
    if unreliable_streak > 0:
        print(f"  ⚠️  [ОЦЕНЩИК] Ни один ответ в раунде не распарсился — подряд "
              f"{unreliable_streak}/{state.get('max_unreliable_evals', 3)} ненадёжных раундов.")

    return {
        "candidate_scores": scores,
        "tokens_used": total_tokens,
        "api_errors": api_errors,
        "tool_calls": tool_calls,
        "tool_salvaged": tool_salvaged,
        "unreliable_eval_streak": unreliable_streak,
        "eval_history": [{
            "depth": len(state.get('steps', [])),
            "scores": scores,
            "reliable": any_reliable,
            "in_recovery": bool(state.get('in_recovery')),
        }],
    }


def trigger_recovery(state: AgentState):
    new_attempts = state.get('step_recovery_attempts', 0) + 1
    total_so_far = state.get('total_recovery_events', 0) + 1
    print(f"\n[Node: Recovery] Triggering k-branch recovery for the current step. "
          f"(Event {total_so_far}/{state['max_recoveries']} for the entire run)")
    return {
        "in_recovery": True,
        "step_recovery_attempts": new_attempts,
        "total_recovery_events": 1,
    }


# Инструкция для узла finish_answer. Намеренно короткая и без размышлений:
# вывод уже сделан в принятых шагах, нужен только сам ответ.
FINISH_INSTRUCTION = (
    "The accepted steps above already contain the derivation. Do NOT start a new "
    "line of reasoning and do NOT introduce new notation.\n"
    "Combine what the steps have established and state the single quantity the "
    "problem asks for, boxed.\n"
    "Reply with one short sentence followed by \\boxed{...} and nothing else. "
    "If the steps genuinely do not determine the answer, reply with the bare word "
    "UNDETERMINED."
)
FINISH_NUM_PREDICT = int(os.getenv("FINISH_NUM_PREDICT", "4000"))


def finish_answer(state: AgentState):
    """Последняя попытка собрать ответ из уже принятых шагов.

    Зачем. Замер по прогонам v9: из 8 сдач семь — «recovery budget exhausted», и
    у этих задач на момент сдачи было от 1 до 5 ПРИНЯТЫХ шагов, а ответ всё
    равно уходил как null. Классический режим 4B: вывод в шагах есть, финал не
    забоксили. Один короткий вызов дешевле (~4k токенов), чем ещё круг recovery
    (~11k) или чем ноль за задачу.

    Узел не выдумывает математику: генератору запрещено начинать новую линию
    рассуждения, и предусмотрен честный отказ (UNDETERMINED).
    """
    steps = state.get('steps', [])
    depth = len(steps)
    if not steps:
        print("\n[Node: Finish] Принятых шагов нет — собирать ответ не из чего.")
        return {"finish_attempted": True}

    print(f"\n[Node: Finish] Собираю ответ из {depth} принятых шагов "
          f"(вместо сдачи с пустым ответом).")
    role = ROLES["generator"]
    context = _build_context(state['problem'], steps)
    messages = [
        {"role": "system", "content": role.system_prompt},
        {"role": "user", "content": f"{context}\n\n{FINISH_INSTRUCTION}"},
    ]
    res = _chat(messages, temperature=0.2, num_predict=FINISH_NUM_PREDICT,
                enable_thinking=False, stage="finish", depth=depth)
    answer = extract_answer(res.text)
    RECORDER.record(stage="finish_result", depth=depth, content=res.text,
                    answer=answer, tokens={"total": res.tokens})
    if answer:
        print(f"  -> Ответ собран: {answer}")
        return {"final_answer": answer, "tokens_used": res.tokens,
                "finish_attempted": True}
    print("  -> Ответ собрать не удалось (UNDETERMINED или пусто).")
    return {"tokens_used": res.tokens, "finish_attempted": True}


def give_up(state: AgentState):
    if state.get('stuck_streak', 0) >= state.get('max_stuck_steps', 2):
        reason = (
            f"No-progress detected: {state['stuck_streak'] + 1} committed steps in a row did not add new "
            f"content. Stopping instead of grinding through the token budget "
            f"({state.get('tokens_used', 0)}/{state.get('token_budget', 0)})."
        )
    elif state.get('unreliable_eval_streak', 0) >= state.get('max_unreliable_evals', 3):
        if state.get('api_errors'):
            # Не приписываем модели то, что сломала сеть: при таймаутах ответ
            # оценщика пуст, и это неотличимо от «выдал не-JSON», хотя лечится
            # это --timeout, а не промптом.
            reason = (
                f"Aborted after {state['unreliable_eval_streak']} unusable evaluator rounds, but the "
                f"real cause looks like infrastructure: {state['api_errors']} network failure(s) "
                f"(timeout/reset) during this task, {state.get('tokens_used', 0)} tokens actually spent. "
                f"Raise --timeout or reduce --workers; this is not a model-quality result."
            )
        else:
            reason = (
                f"Evaluator returned unparseable JSON {state['unreliable_eval_streak']} rounds in a row — "
                f"likely a format/server issue, not a step-quality problem. Stopping "
                f"({state.get('tokens_used', 0)}/{state.get('token_budget', 0)} tokens used)."
            )
    elif state.get('tokens_used', 0) >= state.get('token_budget', 0):
        reason = (
            f"The token budget is exhausted ({state.get('tokens_used', 0)}/{state.get('token_budget', 0)}), "
            f"a clear \\boxed{{}} was not received."
        )
    else:
        reason = (
            f"Recovery budget for the entire run ({state.get('max_recoveries', 0)}) is exhausted and the "
            f"best candidate is still below score_threshold — stopping rather than committing low-quality steps."
        )
    print(f"\n[Node: Give Up] {reason}")
    return {
        "final_answer": None,
        "is_valid": False,
        "verifier_rationale": reason,
        "gave_up": True,
        "gave_up_reason": reason,
    }


# Числа в тексте шага: целые и десятичные, без индексов вида x_2 и без кусков
# LaTeX-команд. Нужны только для ПАССИВНОЙ диагностики (см. _answer_shape).
# Хвостовая точка — это конец предложения, а не часть числа: без отдельной
# оговорки «105.» не распознавалось вообще (точка попадала под запрет справа).
_NUMBER_RE = re.compile(r"(?<![\w\\.])(-?\d+(?:\.\d+)?)(?!\.?\d)(?!\w)")


def _answer_shape(step: str, answer: Optional[str], prior_steps: List[str]) -> Dict[str, Any]:
    """Записывает СЫРЫЕ факты о том, как боксированное значение соотносится с
    текстом шага. Это не детектор и не правило: ничего не блокирует, ни на что
    не влияет, никакой интерпретации здесь не делается.

    Зачем. На разных бенчмарках повторяется режим «вывели одно число, забоксили
    другое» (hmmt задача 13: вывели 105, ответили 266; aime24 задача 87: 699
    против 700; hmmt задача 12: посчитали 29, забоксили 29+1=30). Какой именно
    признак его отличает от нормального финального шага — заранее неизвестно:
    в задаче 12, например, само число 30 в тексте шага присутствует, так что
    простая проверка «ответ встречался раньше» его не ловит.

    Поэтому функция не угадывает признак, а складывает данные, по которым его
    можно искать офлайн. Никаких ключевых слов и никаких предположений о
    формате ответа или о бенчмарке здесь нет и быть не должно.
    """
    if not answer:
        return {}
    head = step.split("\\boxed", 1)[0]
    in_step = [m.group(1) for m in _NUMBER_RE.finditer(head)]
    prior_nums = [m.group(1) for m in _NUMBER_RE.finditer(" ".join(prior_steps))]
    ans = answer.strip()
    return {
        "boxed_value": ans,
        # Встречалось ли значение ответа в тексте шага до самого бокса.
        "boxed_seen_before_box": ans in in_step,
        # Встречалось ли оно в уже принятых шагах.
        "boxed_seen_in_prior_steps": ans in prior_nums,
        "last_number_before_box": in_step[-1] if in_step else None,
        "distinct_numbers_in_step": len(set(in_step)),
        # Сами числа (с потолком, чтобы не раздувать траекторию) — по ним и
        # ведётся офлайн-поиск признака.
        "numbers_before_box": in_step[-12:],
    }


def commit_step(state: AgentState):
    scores = state['candidate_scores']
    steps = state['candidate_steps']
    best_score = max(scores)
    tied = [i for i, s in enumerate(scores) if s == best_score]

    if len(tied) > 1:
        with_answer = [i for i in tied if extract_answer(steps[i])]
        best_idx = with_answer[0] if with_answer else tied[0]
    else:
        best_idx = tied[0]

    best_step = state['candidate_steps'][best_idx]
    best_score = state['candidate_scores'][best_idx]

    print(f"\n[Node: Commit] Selected best candidate (Score: {best_score:.4f}). Appending to steps.")

    prior_steps = state.get('steps', [])
    no_progress = bool(prior_steps) and _normalize_step_text(best_step) == _normalize_step_text(prior_steps[-1])
    stuck_streak = (state.get('stuck_streak', 0) + 1) if no_progress else 0
    if no_progress:
        print(f"  ⚠️  [NO PROGRESS] Принятый шаг не отличается по содержанию от предыдущего "
              f"— подряд {stuck_streak}/{state.get('max_stuck_steps', 2)}.")

    answer = extract_answer(best_step)

    min_before = state.get('min_steps_before_answer', 0)
    premature = 0
    if answer and len(prior_steps) < min_before:
        print(f"  ⚠️  [PREMATURE ANSWER] Ответ {answer!r} получен на глубине "
              f"{len(prior_steps)}, требуется минимум {min_before} принятых шаг(ов). "
              f"Не засчитываю, продолжаем вывод.")
        answer = None
        premature = 1
    elif answer:
        print(f"  -> Explicit answer found: {answer}")

    shape = _answer_shape(best_step, answer, prior_steps)
    if shape and not shape["boxed_seen_before_box"]:
        print(f"  ℹ️  [ANSWER SHAPE] Ответ {answer!r} не встречается в тексте шага до "
              f"бокса; последнее число перед ним — {shape['last_number_before_box']!r}. "
              f"Только замер, на решение не влияет.")

    RECORDER.record(
        stage="commit", depth=len(prior_steps), branch=best_idx + 1,
        content=best_step, score=best_score, answer=answer,
        premature=bool(premature), no_progress=no_progress,
        all_scores=list(scores), **shape,
    )

    result = {
        "steps": [best_step],
        "final_answer": answer if answer else "",
        "in_recovery": False,
        "candidate_raw": [],
        "candidate_steps": [],
        "candidate_scores": [],
        "stuck_streak": stuck_streak,
        "step_recovery_attempts": 0,
        "premature_answers": premature,
    }
    if answer:
        result["answer_depth"] = len(prior_steps)
        result["answers_not_in_step_text"] = 0 if shape.get(
            "boxed_seen_before_box") else 1
    return result


def verify_solution(state: AgentState):
    if state.get('skip_verifier'):
        print("\n[Node: Verify] Пропущен (--no-verify-step).")
        return {"is_valid": False,
                "verifier_rationale": "Верификатор отключён флагом --no-verify-step."}
    print("\n[Node: Verify] Running verifier...")
    role = ROLES["verifier"]
    context = _build_context(state['problem'], state.get('steps', []))
    messages = role.build_messages(context=context)
    res = _chat(messages, json_format=role.json_format, temperature=role.temperature,
                num_predict=role.num_predict, enable_thinking=role.enable_thinking,
                stage="verify", depth=len(state.get('steps', [])))

    is_valid, rationale, reliable = _parse_verify_response(res.text)
    if not reliable:
        print(f"  ⚠️  [НЕНАДЁЖНЫЙ ВЕРДИКТ] {rationale}")

    RECORDER.record(
        stage="verify_result", depth=len(state.get('steps', [])),
        content=rationale, is_valid=is_valid, reliable=reliable,
    )
    print(f"  -> Valid: {is_valid} | Rationale: {rationale}")
    return {
        "is_valid": is_valid,
        "verifier_rationale": rationale,
        "tokens_used": res.tokens,
        "api_errors": int(bool(res.error)),
    }


# ---------------------------------------------------------------------------
# 7. Роутеры
# ---------------------------------------------------------------------------
def route_after_eval(state: AgentState):
    scores = state.get('candidate_scores') or []
    if not scores:
        print("\n[Router] Кандидатов нет — генерация не дала результата. Giving up.")
        return "give_up"
    best_score = max(scores)
    tokens_used = state.get('tokens_used', 0)
    token_budget = state.get('token_budget', 10**9)
    attempts = state.get("step_recovery_attempts", 0)
    max_attempts = state.get("max_step_attempts", 3)

    total_recoveries = state.get('total_recovery_events', 0)
    max_recoveries = state.get('max_recoveries', 5)

    if tokens_used >= token_budget:
        print(f"\n[Router] Token budget exhausted ({tokens_used}/{token_budget}). "
              f"Commit the best available option.")
        return "commit"

    unreliable_streak = state.get('unreliable_eval_streak', 0)
    max_unreliable = state.get('max_unreliable_evals', 3)
    if unreliable_streak >= max_unreliable:
        print(f"\n[Router] Evaluator unparseable for {unreliable_streak} rounds in a row "
              f"(limit {max_unreliable}). Giving up.")
        return "give_up"
    def _out_of_attempts(why: str) -> str:
        if state.get('steps') and not state.get('finish_attempted'):
            print(f"\n[Router] {why} Принятые шаги есть — пробую собрать ответ из них.")
            return "finish"
        print(f"\n[Router] {why} Giving up.")
        return "give_up"

    # 1. Первичное срабатывание восстановления (шли в один поток).
    if (best_score < state['score_threshold']
            and not state['in_recovery']
            and state['branch_mode'] == 'single'):
        if total_recoveries < max_recoveries:
            print(f"\n[Router] Best score {best_score:.4f} < Threshold {state['score_threshold']}. Initiating recovery.")
            return "recover"
        return _out_of_attempts(
            f"Score {best_score:.4f} below threshold {state['score_threshold']}, "
            f"but global recovery budget ({max_recoveries}) exhausted.")

    # 2. Повторные попытки (уже в режиме восстановления).
    if best_score < state['score_threshold'] and state['in_recovery']:
        if attempts < max_attempts and total_recoveries < max_recoveries:
            print(f"\n[Router] All {state.get('k_branches', 3)} branches failed. "
                  f"Triggering recovery attempt {attempts + 1}/{max_attempts} "
                  f"(Global recoveries used: {total_recoveries}/{max_recoveries}).")
            return "recover"
        if total_recoveries >= max_recoveries:
            return _out_of_attempts(
                f"Global recovery budget ({max_recoveries}) exhausted during retries.")
        return _out_of_attempts(
            f"All {max_attempts} recovery attempts for this step failed.")

    # 3. Успех.
    print(f"\n[Router] Score {best_score:.4f} meets threshold. Committing.")
    return "commit"


def route_after_commit(state: AgentState):
    if state.get("final_answer"):
        return "verify"
    steps = state.get('steps') or []
    can_finish = bool(steps) and not state.get('finish_attempted')
    used = state.get('tokens_used', 0)
    budget = state.get('token_budget', 10**9)
    finish_at = float(state.get('finish_at_fraction', 0.85))
    if used >= budget:
        if can_finish:
            print("\n[Router] Бюджет исчерпан, но шаги есть — собираю ответ из них.")
            return "finish"
        return "give_up"
    if can_finish and budget < 10**9 and used >= finish_at * budget:
        print(f"\n[Router] Израсходовано {used:,}/{budget:,} ({used/budget:.0%} ≥ "
              f"{finish_at:.0%}) при {len(steps)} принятых шагах — собираю ответ, "
              f"пока бюджет на это есть.")
        return "finish"
    if state.get('stuck_streak', 0) >= state.get('max_stuck_steps', 2):
        print(f"\n[Router] {state['stuck_streak'] + 1} committed steps in a row added no new content.")
        return "finish" if can_finish else "give_up"
    return "generate"


def route_after_finish(state: AgentState):
    return "verify" if state.get("final_answer") else "give_up"


# ---------------------------------------------------------------------------
# 8. Сборка графа
# ---------------------------------------------------------------------------
def build_solver_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_step", generate_step)
    workflow.add_node("segment_step", segment_step)          
    workflow.add_node("evaluate_steps", evaluate_steps)
    workflow.add_node("trigger_recovery", trigger_recovery)
    workflow.add_node("commit_step", commit_step)
    workflow.add_node("verify_solution", verify_solution)
    workflow.add_node("finish_answer", finish_answer)
    workflow.add_node("give_up", give_up)

    workflow.set_entry_point("generate_step")

    workflow.add_edge("generate_step", "segment_step")       # generate -> segment
    workflow.add_edge("segment_step", "evaluate_steps")      # segment  -> evaluate
    workflow.add_conditional_edges(
        "evaluate_steps", route_after_eval,
        {"recover": "trigger_recovery", "commit": "commit_step",
         "finish": "finish_answer", "give_up": "give_up"},
    )
    workflow.add_edge("trigger_recovery", "generate_step")
    workflow.add_conditional_edges(
        "commit_step", route_after_commit,
        {"verify": "verify_solution", "generate": "generate_step",
         "finish": "finish_answer", "give_up": "give_up"},
    )

    workflow.add_conditional_edges(
        "finish_answer", route_after_finish,
        {"verify": "verify_solution", "give_up": "give_up"},
    )
    workflow.add_edge("verify_solution", END)
    workflow.add_edge("give_up", END)

    return workflow.compile()


def make_initial_state(problem: str, args=None, **overrides) -> Dict[str, Any]:
    """Начальное состояние. При переданном args (argparse Namespace из раннера)
    берёт из него параметры поиска; иначе — разумные дефолты. Форму состояния
    _solve_with_graph из agent_benchmark_runner получает именно отсюда."""
    state: Dict[str, Any] = {
        "problem": problem,
        "steps": [],
        "candidate_raw": [],
        "candidate_steps": [],
        "candidate_scores": [],
        "k_branches": 3,
        "score_threshold": 0.8,
        "branch_mode": "single",
        "base_temperature": None,
        "tokens_used": 0,
        "token_budget": 250000,
        "in_recovery": False,
        "max_recoveries": 15,
        "total_recovery_events": 0,
        "finish_attempted": False,
        "stuck_streak": 0,
        "max_stuck_steps": 2,
        "unreliable_eval_streak": 0,
        "max_unreliable_evals": 3,
        "eval_history": [],
        "thinking_overruns": 0,
        "segmenter_calls": 0,
        "segmenter_unreliable": 0,
        "api_errors": 0,
        "use_tools": False,
        "tool_calls": 0,
        "tool_salvaged": 0,
        "min_steps_before_answer": 0,
        "premature_answers": 0,
        "answers_not_in_step_text": 0,
        "answer_depth": None,
        "final_answer": None,
        "is_valid": False,
        "verifier_rationale": "",
        "gave_up": False,
        "gave_up_reason": "",
        "step_recovery_attempts": 0,
        "max_step_attempts": 3,
        "skip_verifier": False,
        "finish_at_fraction": 0.85,
    }
    if args is not None:
        state.update({
            "k_branches": args.k_branches,
            "score_threshold": args.score_threshold,
            "branch_mode": args.branch_mode,
            "base_temperature": args.temperature,
            "token_budget": args.token_budget,
            "max_recoveries": args.max_recoveries,
            "max_stuck_steps": args.max_stuck_steps,
            "max_unreliable_evals": args.max_unreliable_evals,
            "min_steps_before_answer": getattr(args, "min_steps_before_answer", 0),
            "use_tools": not getattr(args, "no_tools", False),
            "max_step_attempts": getattr(args, "max_step_attempts", 3),
            "skip_verifier": bool(getattr(args, "no_verify_step", False)),
            "finish_at_fraction": getattr(args, "finish_at", 0.85),
        })
    state.update(overrides)
    return state


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "conf/base/prompts/agent-step-qwen4b-v1.yml"


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        from dotenv import load_dotenv
        load_dotenv()
        MODEL_NAME = os.getenv("MODEL", MODEL_NAME)
        BASE_URL = os.getenv("OPENAI_BASE_URL", BASE_URL)
        API_KEY = os.getenv("OPENAI_API_KEY", API_KEY)
    except ImportError:
        pass

    load_prompts_from_yaml(DEFAULT_PROMPT_PATH)
    print(f"[smoke] MODEL={MODEL_NAME} BASE_URL={BASE_URL}")

    graph = build_solver_graph()
    problem = (
        "Let $x$ and $y$ be positive integers such that $x + y = 20$ and "
        "$x \\cdot y$ is as large as possible. Find $x \\cdot y$."
    )
    final = graph.invoke(make_initial_state(problem, token_budget=120000))
    print("\n================ RESULT ================")
    print(f"steps: {len(final.get('steps', []))}")
    for i, s in enumerate(final.get('steps', []), 1):
        print(f"  Step {i}: {s}")
    print(f"final_answer: {final.get('final_answer')!r}")
    print(f"is_valid: {final.get('is_valid')} | gave_up: {final.get('gave_up')}")
    print(f"tokens_used: {final.get('tokens_used')} | segmenter_calls: {final.get('segmenter_calls')}")
