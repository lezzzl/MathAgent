"""Пошаговый солвер под Qwen3-4B — вариант langgraph_math_solver БЕЗ инструментов
и с дополнительной стадией сегментации.

Зачем отдельный файл
--------------------
Основной пайплайн (langgraph_math_solver.py) писался под Qwen 7B/9B, которые
достаточно дисциплинированы, чтобы заворачивать один шаг в <step>...</step> и
финальный ответ в \\boxed{}. Qwen3-4B этой инструкции почти не слушается: он
вываливает в один ответ сразу всё решение вперемешку с рассуждениями. Прежний
экстрактор _extract_step_content в этом случае откатывался к "весь сырой текст
как шаг", и оценщик получал не атомарный шаг, а мусор — отсюда и провальные
результаты на AIME24/25/26.

Что изменено относительно оригинала
-----------------------------------
1. Убраны инструменты (тулы) целиком: все роли ходят в модель через простой
   HTTP-хелпер _chat, без langchain-субграфа и без python_exec. Тулы вернём
   позже отдельным шагом.
2. Добавлена НОВАЯ стадия-узел `segment_step` между генерацией и оценкой. Она
   берёт сырой ответ генератора и вырезает из него ровно один следующий шаг с
   помощью дешёвой роли `segmenter` (размышления выключены). Так дисциплина
   форматирования обеспечивается отдельным проходом, а не выпрашивается у 4B.
3. Промпты берутся из conf/base/prompts/agent-step-qwen4b-v1.yml (там же живёт
   роль segmenter).

Публичный интерфейс (ROLES, load_prompts_from_yaml, build_solver_graph,
MODEL_NAME/BASE_URL/API_KEY/... и форма AgentState) намеренно совместим с
langgraph_math_solver.py, чтобы существующий agent_benchmark_runner мог
использовать этот модуль как drop-in.
"""

import json
import operator
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, NamedTuple, Optional, Tuple

import requests
import yaml
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from answer_utils import extract_answer, iter_boxed


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
    # НОВАЯ роль: вырезает один шаг из сырого ответа генератора.
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
            ROLES[role_name] = Role(
                name=role_name,
                system_prompt=role_cfg.get("system", defaults["system"]),
                user_template=role_cfg.get("user_template", defaults["user_template"]),
                temperature=float(role_cfg.get("temperature", defaults["temperature"])),
                json_format=bool(role_cfg.get("json_format", defaults["json_format"])),
                num_predict=role_cfg.get("num_predict", defaults["num_predict"]),
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

# Если шаг из <step> длиннее этого — подозреваем, что модель втиснула туда сразу
# несколько шагов, и всё равно прогоняем через сегментатор.
MAX_STEP_CHARS = int(os.getenv("MAX_STEP_CHARS", "1500"))


class ChatResult(NamedTuple):
    content: str
    reasoning: str
    tokens: int
    finish_reason: Optional[str]

    @property
    def text(self) -> str:
        """content, а при его пустоте — reasoning_content.

        Reasoning-парсер сервера кладёт размышления в reasoning_content, а сам
        ответ — в content. Обычно нам нужен content; reasoning — только фоллбек
        на случай, когда сервер не отдал content отдельно.
        """
        return (self.content or "").strip() or (self.reasoning or "").strip()


def _chat(messages, temperature=0.2, seed=None, num_predict=None, json_format=False,
          enable_thinking=None) -> ChatResult:
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
    if seed is not None:
        payload["seed"] = seed
    if json_format:
        payload["response_format"] = {"type": "json_object"}
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices")
        if not choices:
            print(f"[WARN] No choices in response: {data}")
            return ChatResult("", "", 0, None)

        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content", "") or ""
        reasoning = message.get("reasoning_content", "") or message.get("reasoning", "") or ""
        if not content and not reasoning:
            print(f"[WARN] Empty content and reasoning in response: {data}")

        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        finish_reason = choice.get("finish_reason")
        return ChatResult(content, reasoning, tokens_used, finish_reason)

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        if 'response' in locals() and response is not None:
            print(f"Response content: {response.text}")
        return ChatResult("", "", 0, None)


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


def _has_clean_single_step(raw: str) -> bool:
    """Можно ли доверять быстрому пути без вызова сегментатора.

    True, если в сыром ответе есть закрытый <step>, извлечённое содержимое
    умещается в MAX_STEP_CHARS и содержит не больше одного \\boxed{}. Иначе —
    отдаём на сегментацию (тегов нет, или в один блок втиснули всё решение).
    """
    if not _STEP_TAG_RE.search(raw or ""):
        return False
    step = _extract_step_content(raw)
    if not step or len(step) > MAX_STEP_CHARS:
        return False
    if len(list(iter_boxed(step))) > 1:
        return False
    return True


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


def _parse_segmenter_response(content: str, raw_fallback: str) -> Tuple[str, Optional[str], bool]:
    """Разбирает ответ сегментатора формата ###STEP### / ###ANSWER###.

    Разделители вместо JSON выбраны намеренно: вывод сегментатора — сплошной
    LaTeX (\\frac, \\sqrt, \\boxed), а именно он ломает JSON-парсинг из-за
    неэкранированных бэкслэшей. Строчные маркеры такой проблемы лишены.

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
        return fallback, extract_answer(fallback), False

    step = _STEP_ANY_TAG_RE.sub("", m.group(1)).strip()

    answer: Optional[str] = None
    a = _SEG_ANSWER_RE.search(text)
    if a:
        cand = a.group(1).strip()
        if cand and cand.upper() != "NONE":
            answer = extract_answer(cand) or cand

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


def _parse_eval_response(content: str) -> Tuple[float, str, bool]:
    """Разбирает ответ оценщика. Возвращает (score, rationale, is_reliable)."""
    if not content:
        return 0.0, "Empty response from evaluator", False

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
    base_temperature: float

    tokens_used: Annotated[int, operator.add]
    token_budget: int
    in_recovery: bool
    max_recoveries: int
    total_recovery_events: Annotated[int, operator.add]

    stuck_streak: int
    max_stuck_steps: int

    unreliable_eval_streak: int
    max_unreliable_evals: int
    eval_history: Annotated[List[Dict[str, Any]], operator.add]
    thinking_overruns: Annotated[int, operator.add]
    # Сколько раз пришлось звать LLM-сегментатор (быстрый путь по тегам не сработал).
    segmenter_calls: Annotated[int, operator.add]

    final_answer: Optional[str]
    is_valid: bool
    verifier_rationale: str
    gave_up: bool
    gave_up_reason: str

    step_recovery_attempts: int
    max_step_attempts: int
    # Оставлено для совместимости формы состояния с основным раннером; в этом
    # пайплайне тулов нет, значение игнорируется.
    use_tools: bool


# ---------------------------------------------------------------------------
# 6. Узлы графа
# ---------------------------------------------------------------------------
def _generate_one(role: Role, context: str, temp: float) -> Tuple[str, bool, int]:
    """Один вызов генератора с откатом при обрыве размышлений.

    У thinking-моделей размышления идут в тот же num_predict, что и ответ. Если
    их не хватило, сервер возвращает finish_reason='length' и пустой content —
    шаг теряется целиком. Повторяем один раз с выключенными размышлениями:
    лучше шаг без ризонинга, чем пустой шаг.

    Возвращает (raw_content, overran, total_tokens).
    """
    messages = [
        {"role": "system", "content": role.system_prompt},
        {"role": "user", "content": role.user_template.format(context=context)},
    ]

    result = _chat(messages, temperature=temp, num_predict=role.num_predict,
                   json_format=role.json_format, enable_thinking=role.enable_thinking)

    truncated_empty = (not result.content.strip()) and result.finish_reason == "length"
    if truncated_empty and role.enable_thinking:
        print(f"      [THINKING OVERRUN] Размышления съели весь лимит "
              f"({role.num_predict} токенов), ответ пуст. Повтор без размышлений.")
        retry = _chat(messages, temperature=temp, num_predict=role.num_predict,
                      json_format=role.json_format, enable_thinking=False)
        # Учитываем и токены оборванной первой попытки, и токены повтора.
        return retry.text, True, result.tokens + retry.tokens
    if truncated_empty:
        print(f"      [TRUNCATED] Ответ пуст, finish_reason=length при лимите "
              f"{role.num_predict}. Поднимите num_predict генератора в yaml.")
    return result.text, False, result.tokens


def generate_step(state: AgentState):
    current_depth = len(state.get('steps', []))
    print(f"\n[Node: Generate] Depth: {current_depth} | Tokens used: {state.get('tokens_used', 0)}")
    multi = state.get('branch_mode') == 'multi' or state.get('in_recovery')
    k = state.get('k_branches', 3) if multi else 1
    print(f"  -> Generating {k} candidate(s).")

    role = ROLES["generator"]
    raw_candidates: List[str] = []
    total_tokens = 0
    overruns = 0
    context = _build_context(state['problem'], state.get('steps', []))

    base_temp = state.get('base_temperature')
    if base_temp is None:
        base_temp = role.temperature

    for i in range(k):
        attempt = state.get('step_recovery_attempts', 0)
        temp = min(base_temp + 0.15 * i + 0.1 * attempt, 1.1)
        raw_text, overran, tks = _generate_one(role, context, temp)
        overruns += overran
        total_tokens += tks
        raw_candidates.append(raw_text)
        preview = re.sub(r"\s+", " ", raw_text).strip()[:200]
        print(f"    - Branch {i+1} raw generated (temp: {temp:.2f}, tokens: {tks})")
        print(f"      Raw preview: {preview}{'...' if len(raw_text) > 200 else ''}")

    return {"candidate_raw": raw_candidates, "tokens_used": total_tokens,
            "thinking_overruns": overruns}


def segment_step(state: AgentState):
    """НОВАЯ стадия: из каждого сырого ответа генератора вырезает ровно один шаг.

    Быстрый путь: если в сыром ответе уже есть аккуратный одиночный <step>, берём
    его без вызова модели. Иначе зовём роль segmenter (размышления выключены),
    которая возвращает чистый шаг и, если он есть, финальный ответ.
    """
    raw_candidates = state.get('candidate_raw') or []
    print(f"\n[Node: Segment] Сегментирую {len(raw_candidates)} сырых кандидата(ов)...")
    role = ROLES["segmenter"]
    context = _build_context(state['problem'], state.get('steps', []))

    steps: List[str] = []
    total_tokens = 0
    seg_calls = 0

    for i, raw in enumerate(raw_candidates):
        if not (raw or "").strip():
            steps.append("")
            print(f"    - Candidate {i+1}: пустая генерация, пропускаю сегментацию.")
            continue

        if _has_clean_single_step(raw):
            step = _extract_step_content(raw)
            print(f"    - Candidate {i+1}: чистый <step> найден — быстрый путь без сегментатора.")
        else:
            seg_calls += 1
            messages = [
                {"role": "system", "content": role.system_prompt},
                {"role": "user", "content": role.user_template.format(context=context, raw=raw)},
            ]
            res = _chat(messages, temperature=role.temperature, num_predict=role.num_predict,
                        json_format=role.json_format, enable_thinking=role.enable_thinking)
            total_tokens += res.tokens
            step, answer, reliable = _parse_segmenter_response(res.text, raw)
            tag = "" if reliable else " ⚠️ [маркеры не распознаны, фоллбек-экстракция]"
            ans_note = f" | answer={answer}" if answer else ""
            print(f"    - Candidate {i+1}: сегментатор вернул шаг ({res.tokens} ток.){ans_note}{tag}")

        steps.append(step)
        print(f"      Step:\n{step}\n")

    return {"candidate_steps": steps, "tokens_used": total_tokens,
            "segmenter_calls": seg_calls}


def evaluate_steps(state: AgentState):
    candidates = state['candidate_steps']
    print(f"\n[Node: Evaluate] Checking {len(candidates)} candidate(s)...")
    role = ROLES["evaluator"]
    scores: List[float] = []
    seen: Dict[str, Tuple[float, str]] = {}
    total_tokens = 0
    any_reliable = False
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

        messages = [
            {"role": "system", "content": role.system_prompt},
            {"role": "user", "content": role.user_template.format(context=context, step=step)},
        ]
        res = _chat(messages, temperature=role.temperature, num_predict=role.num_predict,
                    json_format=role.json_format, enable_thinking=role.enable_thinking)
        total_tokens += res.tokens

        score, rationale, reliable = _parse_eval_response(res.text)
        any_reliable = any_reliable or reliable
        seen[key] = (score, rationale)
        scores.append(score)

        tag = "" if reliable else " ⚠️ [ОЦЕНКА НЕНАДЁЖНА]"
        print(f"    - Candidate {i+1} Score: {score:.4f}{tag} | Rationale: {rationale}")

    unreliable_streak = 0 if any_reliable else state.get('unreliable_eval_streak', 0) + 1
    if unreliable_streak > 0:
        print(f"  ⚠️  [ОЦЕНЩИК] Ни один ответ в раунде не распарсился — подряд "
              f"{unreliable_streak}/{state.get('max_unreliable_evals', 3)} ненадёжных раундов.")

    return {
        "candidate_scores": scores,
        "tokens_used": total_tokens,
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


def give_up(state: AgentState):
    if state.get('stuck_streak', 0) >= state.get('max_stuck_steps', 2):
        reason = (
            f"No-progress detected: {state['stuck_streak'] + 1} committed steps in a row did not add new "
            f"content. Stopping instead of grinding through the token budget "
            f"({state.get('tokens_used', 0)}/{state.get('token_budget', 0)})."
        )
    elif state.get('unreliable_eval_streak', 0) >= state.get('max_unreliable_evals', 3):
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
    if answer:
        print(f"  -> Explicit answer found: {answer}")

    return {
        "steps": [best_step],
        "final_answer": answer if answer else "",
        "in_recovery": False,
        "candidate_raw": [],
        "candidate_steps": [],
        "candidate_scores": [],
        "stuck_streak": stuck_streak,
        "step_recovery_attempts": 0,
    }


def verify_solution(state: AgentState):
    print("\n[Node: Verify] Running verifier...")
    role = ROLES["verifier"]
    context = _build_context(state['problem'], state.get('steps', []))
    messages = role.build_messages(context=context)
    res = _chat(messages, json_format=role.json_format, temperature=role.temperature,
                num_predict=role.num_predict, enable_thinking=role.enable_thinking)

    is_valid, rationale, reliable = _parse_verify_response(res.text)
    if not reliable:
        print(f"  ⚠️  [НЕНАДЁЖНЫЙ ВЕРДИКТ] {rationale}")

    print(f"  -> Valid: {is_valid} | Rationale: {rationale}")
    return {
        "is_valid": is_valid,
        "verifier_rationale": rationale,
        "tokens_used": res.tokens,
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

    # 1. Первичное срабатывание восстановления (шли в один поток).
    if (best_score < state['score_threshold']
            and not state['in_recovery']
            and state['branch_mode'] == 'single'):
        if total_recoveries < max_recoveries:
            print(f"\n[Router] Best score {best_score:.4f} < Threshold {state['score_threshold']}. Initiating recovery.")
            return "recover"
        print(f"\n[Router] Score {best_score:.4f} below threshold {state['score_threshold']}, but recovery "
              f"budget ({max_recoveries}) exhausted. Giving up.")
        return "give_up"

    # 2. Повторные попытки (уже в режиме восстановления).
    if best_score < state['score_threshold'] and state['in_recovery']:
        if attempts < max_attempts and total_recoveries < max_recoveries:
            print(f"\n[Router] All {state.get('k_branches', 3)} branches failed. "
                  f"Triggering recovery attempt {attempts + 1}/{max_attempts} "
                  f"(Global recoveries used: {total_recoveries}/{max_recoveries}).")
            return "recover"
        if total_recoveries >= max_recoveries:
            print(f"\n[Router] Global recovery budget ({max_recoveries}) exhausted during retries. Giving up.")
            return "give_up"
        print(f"\n[Router] All {max_attempts} recovery attempts for this step failed. Giving up.")
        return "give_up"

    # 3. Успех.
    print(f"\n[Router] Score {best_score:.4f} meets threshold. Committing.")
    return "commit"


def route_after_commit(state: AgentState):
    if state.get("final_answer"):
        return "verify"
    if state.get('tokens_used', 0) >= state.get('token_budget', 10**9):
        return "give_up"
    if state.get('stuck_streak', 0) >= state.get('max_stuck_steps', 2):
        print(f"\n[Router] {state['stuck_streak'] + 1} committed steps in a row added no new content. Giving up.")
        return "give_up"
    return "generate"


# ---------------------------------------------------------------------------
# 8. Сборка графа
# ---------------------------------------------------------------------------
def build_solver_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_step", generate_step)
    workflow.add_node("segment_step", segment_step)          # НОВЫЙ узел
    workflow.add_node("evaluate_steps", evaluate_steps)
    workflow.add_node("trigger_recovery", trigger_recovery)
    workflow.add_node("commit_step", commit_step)
    workflow.add_node("verify_solution", verify_solution)
    workflow.add_node("give_up", give_up)

    workflow.set_entry_point("generate_step")

    workflow.add_edge("generate_step", "segment_step")       # generate -> segment
    workflow.add_edge("segment_step", "evaluate_steps")      # segment  -> evaluate
    workflow.add_conditional_edges(
        "evaluate_steps", route_after_eval,
        {"recover": "trigger_recovery", "commit": "commit_step", "give_up": "give_up"},
    )
    workflow.add_edge("trigger_recovery", "generate_step")
    workflow.add_conditional_edges(
        "commit_step", route_after_commit,
        {"verify": "verify_solution", "generate": "generate_step", "give_up": "give_up"},
    )
    workflow.add_edge("verify_solution", END)
    workflow.add_edge("give_up", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 9. Начальное состояние + smoke-test
# ---------------------------------------------------------------------------
def make_initial_state(problem: str, args=None, **overrides) -> Dict[str, Any]:
    """Начальное состояние. При переданном args (argparse Namespace из раннера)
    берёт из него параметры поиска; иначе — разумные дефолты. Форма совместима с
    _solve_with_graph из agent_benchmark_runner (use_tools здесь игнорируется)."""
    state: Dict[str, Any] = {
        "problem": problem,
        "steps": [],
        "candidate_raw": [],
        "candidate_steps": [],
        "candidate_scores": [],
        "k_branches": 3,
        "score_threshold": 0.8,
        "branch_mode": "multi",
        "base_temperature": None,
        "tokens_used": 0,
        "token_budget": 250000,
        "in_recovery": False,
        "max_recoveries": 5,
        "total_recovery_events": 0,
        "stuck_streak": 0,
        "max_stuck_steps": 2,
        "unreliable_eval_streak": 0,
        "max_unreliable_evals": 3,
        "eval_history": [],
        "thinking_overruns": 0,
        "segmenter_calls": 0,
        "final_answer": None,
        "is_valid": False,
        "verifier_rationale": "",
        "gave_up": False,
        "gave_up_reason": "",
        "step_recovery_attempts": 0,
        "max_step_attempts": 3,
        "use_tools": False,
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
            "use_tools": not getattr(args, "no_tools", False),
        })
    state.update(overrides)
    return state


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "conf/base/prompts/agent-step-qwen4b-v1.yml"


if __name__ == "__main__":
    # Автономный smoke-test: решает одну задачу против сервера qwen4b.
    # MODEL / OPENAI_BASE_URL / OPENAI_API_KEY берутся из окружения.
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
