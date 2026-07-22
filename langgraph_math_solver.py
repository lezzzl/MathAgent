import json
import operator
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import requests
import yaml
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from answer_utils import extract_answer
from tool_generator_subgraph import count_chain_tokens, generator_with_tools, make_llm
from tools import reset_calculator_state

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
        """Собирает messages для _chat() из system_prompt + user_template.
        kwargs — именованные значения для плейсхолдеров шаблона (например,
        context=..., step=...). Лишние kwargs, которых нет в шаблоне, просто
        игнорируются — так один и тот же вызов подходит и для ролей, которым
        нужен только context, и для тех, кому нужен ещё и step."""
        try:
            user_content = self.user_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Роль '{self.name}': user_template ссылается на плейсхолдер {e}, "
                f"которого нет среди переданных аргументов ({sorted(kwargs)}). "
                f"Проверьте user_template в agent-step-v1.yml для этой роли."
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
        "num_predict": 4000,
    },
    "evaluator": {
        "system": _FALLBACK_SYSTEM,
        "user_template": "{context}\nGive a score of the new step:\n{step}",
        "temperature": 0.6,
        "json_format": False,
        "num_predict": 4000,
    },
    "verifier": {
        "system": _FALLBACK_SYSTEM,
        "user_template": "{context}",
        "temperature": 0.6,
        "json_format": False,
        "num_predict": 3000,
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


MODEL_NAME = os.getenv("MODEL", "Qwen/Qwen3.5-4B")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "40000"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

def _chat(messages, temperature=0.2, seed=None, num_predict=None, json_format=False,
          enable_thinking=None):
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }

    payload["max_tokens"] = num_predict if num_predict is not None else DEFAULT_MAX_TOKENS

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
            return "", 0

        message = choices[0].get("message")
        if not message:
            print(f"[WARN] No message in response: {data}")
            return "", 0

        content = message.get("content", "")
        if not content:
            content = message.get("reasoning_content", "") or message.get("reasoning", "")

        if not content:
            print(f"[WARN] Empty content and reasoning in response: {data}")

        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return content, tokens_used

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        if 'response' in locals() and response is not None:
            print(f"Response content: {response.text}")
        return "", 0

_STEP_TAG_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL | re.IGNORECASE)
_STEP_OPEN_TAG_RE = re.compile(r"<step>", re.IGNORECASE)
_STEP_ANY_TAG_RE = re.compile(r"</?step\s*>", re.IGNORECASE)
_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
def _extract_step_content(raw: str) -> str:
    """
    Извлекает содержимое <step>...</step>.
    Если модель вынесла \boxed{ за пределы тегов, автоматически
    захватывает весь хвост от \boxed{ и приклеивает его обратно к шагу.
    """
    if not raw:
        return ""

    # Берём ПОСЛЕДНИЙ закрытый блок: reasoning-модель перебирает черновики
    # раньше, а финальный шаг пишет последним (и это согласовано с выбором
    # последнего \boxed ниже). Прежний .search брал первый и возвращал черновик.
    closed = list(_STEP_TAG_RE.finditer(raw))
    if closed:
        step = closed[-1].group(1).strip()
    else:
        # Незакрытый <step> (обрыв по лимиту) — берём от последнего открытия.
        opens = list(_STEP_OPEN_TAG_RE.finditer(raw))
        step = raw[opens[-1].end():].strip() if opens else raw.strip()

    matches = list(_BOXED_START_RE.finditer(raw))
    if matches:
        last_match = matches[-1]
        if last_match.group(0) not in step:
            # Хвост от \boxed{ до конца, но обрезанный по первому же step-тегу:
            # иначе открывающий <step> следующего черновика протекал в шаг и
            # дальше в контекст ("блоки <step> в логах").
            tail = raw[last_match.start():]
            clean_tail = _STEP_ANY_TAG_RE.split(tail)[0].strip()
            if step:
                step += f"\n\nFinal answer: {clean_tail}"
            else:
                step = f"Final answer: {clean_tail}"

    # Страховка: срезаем любые уцелевшие теги, чтобы они не попали в контекст.
    return _STEP_ANY_TAG_RE.sub("", step).strip()


def _build_context(problem: str, steps: List[str]) -> str:
    """Собирает контекст, обрезая середину решения при переполнении.

    Условие задачи и последние шаги нужны модели всегда, а самые ранние шаги
    обычно уже «впитаны» в последующие выкладки — поэтому при нехватке места
    выбрасываем их с начала, а не обрезаем текст по символам.
    """
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


_LEAKED_TOOL_CALL_RE = re.compile(r"<tool_call>|<\|tool_call\|>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MD_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _iter_json_objects(text: str):
    """Перебирает сбалансированные {...} с конца текста к началу.

    Reasoning-модели (Qwen3, R1) пишут длинную цепочку рассуждений, а JSON
    кладут в самый конец. Прежний поиск `\\{.*\\}` жадно захватывал от ПЕРВОЙ
    скобки — а ей обычно оказывалась латеховая \\frac{a}{b} в рассуждениях,
    из-за чего разбор гарантированно падал. Идём с конца: нужный объект там.
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
                    yield text[start : i + 1]
                    break


def _extract_json_dict(content: str, expected_keys: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """Достаёт из ответа модели JSON-объект с одним из ожидаемых ключей."""
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


def _diagnose_non_json(content: str, role: str) -> Optional[str]:
    """Распознаёт типовые причины, по которым ответ не является JSON.

    Главная из них — вызов инструмента, оборванный лимитом токенов: парсер
    hermes не может разобрать незакрытый <tool_call>, и сырой текст утекает в
    content. Раньше это выглядело как «модель выдала мусор», хотя лечится
    увеличением num_predict.
    """
    if _LEAKED_TOOL_CALL_RE.search(content or ""):
        return (
            f"{role}: в ответе сырой <tool_call> вместо JSON. Сервер не распознал "
            f"вызов инструмента — обычно это несовпадение формата с "
            f"--tool-call-parser (Qwen3.x пишет <function=...><parameter=...>, "
            f"а hermes ждёт JSON), реже обрыв по лимиту токенов. Если инструменты "
            f"не нужны, запускайте с --no-tools: тогда они вообще не отдаются модели."
        )
    return None


def _parse_eval_response(content: str) -> Tuple[float, str, bool]:
    """
    Разбирает ответ оценщика с 3 уровнями защиты от типичных сбоев LLM
    (неэкранированные слэши LaTeX, markdown-блоки, битый синтаксис JSON).
    Возвращает (score, rationale, is_reliable).
    """
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

    # 4. РЕГЕКС-ФОЛЛБЕК: Если JSON разрушен, спасаем score и rationale регуляркой
    score_match = re.search(r'"score"\s*:\s*([0-1](?:\.[0-9]+)?)', content, re.IGNORECASE)
    rat_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
    
    if score_match:
        try:
            score = max(0.0, min(1.0, float(score_match.group(1))))
            rationale = rat_match.group(1) if rat_match else f"Extracted via regex (JSON parse failed: {e_first})"
            print(f"  ℹ️ [JSON RECOVERY] Парсер спас оценку score={score:.4f} регулярным выражением!")
            return score, rationale, True
        except ValueError:
            pass

    diagnosis = _diagnose_non_json(content, "оценщик")
    if diagnosis:
        return 0.0, diagnosis, False

    return 0.0, f"Invalid JSON from evaluator (raw preview: {content[:400]!r})", False


def _parse_verify_response(content: str) -> Tuple[bool, str, bool]:
    """Аналогично _parse_eval_response, но для схемы верификатора с лечением LaTeX и фоллбеком."""
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
        print(f"  ℹ️ [JSON RECOVERY] Верификатор спас вердикт is_valid={is_valid} регулярным выражением!")
        return is_valid, rationale, True

    diagnosis = _diagnose_non_json(content, "верификатор")
    if diagnosis:
        return False, diagnosis, False

    return False, f"Invalid JSON from verifier (raw preview: {content[:400]!r})", False


# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    problem: str
    steps: Annotated[List[str], operator.add]  
    
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

    final_answer: Optional[str]
    is_valid: bool
    verifier_rationale: str
    gave_up: bool         
    gave_up_reason: str    

    step_recovery_attempts: int  
    max_step_attempts: int     
    use_tools: bool


# ---------------------------------------------------------------------------
# 2. Graph Nodes
# ---------------------------------------------------------------------------
def _finish_reason(message: Any) -> Optional[str]:
    return (getattr(message, "response_metadata", None) or {}).get("finish_reason")


def _message_text(message: Any) -> str:
    """content сообщения, а при его пустоте — reasoning_content.

    Qwen3.5 не всегда уважает enable_thinking=false и иногда всё равно уходит в
    размышления. reasoning-parser сервера вырезает их в reasoning_content
    (langchain кладёт его в additional_kwargs), и content приходит пустым.
    Раньше evaluate_steps/generate_step видели пустоту и репортили
    'Empty response', хотя JSON или шаг лежали в reasoning_content. _chat для
    верификатора уже читал этот фоллбек — теперь он общий для всех ролей.
    """
    content = (getattr(message, "content", "") or "").strip()
    if content:
        return content
    extra = getattr(message, "additional_kwargs", None) or {}
    return (extra.get("reasoning_content") or extra.get("reasoning") or "").strip()


def _generate_one(role: Role, context: str, temp: float, use_tools: bool):
    """Один вызов генератора с откатом при обрыве на размышлениях.

    У thinking-моделей размышления идут в тот же num_predict, что и ответ. Если
    их не хватило, сервер возвращает finish_reason='length' и ПУСТОЙ content —
    шаг теряется целиком. Recovery тут не помогает: он меняет температуру, а
    причина в бюджете токенов, поэтому все k веток приходят одинаково пустыми.
    В этом случае повторяем вызов один раз с выключенными размышлениями: лучше
    шаг без размышлений, чем пустой шаг и сожжённая попытка восстановления.
    """
    def call(enable_thinking):
        llm = make_llm(temp, model_name=MODEL_NAME, base_url=BASE_URL, api_key=API_KEY,
                       max_tokens=role.num_predict, with_tools=use_tools,
                       enable_thinking=enable_thinking)
        messages = [SystemMessage(content=role.system_prompt), HumanMessage(content=context)]
        if use_tools:
            return generator_with_tools.invoke(
                {"messages": messages, "max_hops": 5},
                config={"configurable": {"llm": llm}},
            )
        return {"messages": messages + [llm.invoke(messages)]}

    result = call(role.enable_thinking)
    last = result["messages"][-1]
    truncated_empty = (
        not _message_text(last)
        and _finish_reason(last) == "length"
    )
    if truncated_empty and role.enable_thinking:
        print(f"      [THINKING OVERRUN] Размышления съели весь лимит "
              f"({role.num_predict} токенов), ответ пуст. Повтор без размышлений.")
        # Токены оборванного вызова (обычно весь num_predict) нужно учесть:
        # иначе token_budget и метрика tokens_used недосчитывают расход, а при
        # частых обрывах на IMO это тысячи токенов на задачу.
        overrun_tokens = count_chain_tokens(result["messages"])
        return call(False), True, overrun_tokens
    if truncated_empty:
        print(f"      [TRUNCATED] Ответ пуст, finish_reason=length при лимите "
              f"{role.num_predict}. Поднимите num_predict генератора в yaml.")
    return result, False, 0


def generate_step(state: AgentState):
    if not state.get('steps'):
        reset_calculator_state()
    current_depth = len(state.get('steps', []))
    print(f"\n[Node: Generate] Depth: {current_depth} | Tokens used: {state.get('tokens_used', 0)}")    
    multi = state.get('branch_mode') == 'multi' or state.get('in_recovery')
    k = state.get('k_branches', 3) if multi else 1
    print(f"  -> Generating {k} candidate(s).")

    role = ROLES["generator"]
    candidates = []
    total_tokens = 0
    overruns = 0
    context = _build_context(state['problem'], state.get('steps', []))
    
    use_tools = state.get("use_tools", True)
    base_temp = state.get('base_temperature')
    if base_temp is None:
        base_temp = role.temperature
    for i in range(k):
        attempt = state.get('step_recovery_attempts', 0)
        temp = min(base_temp + 0.15 * i + 0.1 * attempt, 1.1)
        result, overran, overrun_tokens = _generate_one(role, context, temp, use_tools)
        overruns += overran

        for m in result["messages"]:
            if isinstance(m, ToolMessage):
                print(f"      [tool_result] {m.content[:200]}")
        final_msg = result["messages"][-1]
        raw_text = _message_text(final_msg)
        step_text = _extract_step_content(raw_text)
        candidates.append(step_text)

        # overrun_tokens — расход оборванной первой попытки при откате (0 без него).
        tks = count_chain_tokens(result["messages"]) + overrun_tokens
        total_tokens += tks

        stripped_note = ""
        if step_text != (raw_text or "").strip():
            stripped_note = f" | ⚠️ отброшено {len(raw_text) - len(step_text)} симв. текста вне <step> тегов"
        n_tool_calls = sum(1 for m in result["messages"] if getattr(m, "tool_calls", None))
        fr = _finish_reason(final_msg)
        fr_note = f", finish={fr}" if fr and fr != "stop" else ""
        print(f"    - Branch {i+1} generated (temp: {temp:.2f}, tokens: {tks}, "
              f"tool-вызовов: {n_tool_calls}{fr_note}{stripped_note})")
        print(f"      Step:\n{step_text}\n")

    return {"candidate_steps": candidates, "tokens_used": total_tokens,
            "thinking_overruns": overruns}


def evaluate_steps(state: AgentState):
    candidates = state['candidate_steps']
    print(f"\n[Node: Evaluate] Checking {len(candidates)} candidate(s)...")
    role = ROLES["evaluator"]
    scores: List[float] = []
    seen: Dict[str, Tuple[float, str]] = {}
    total_tokens = 0
    any_reliable = False
    use_tools = state.get("use_tools", True)
    context = _build_context(state['problem'], state.get('steps', []))

    for i, step in enumerate(candidates):
        key = _normalize_step_text(step)
        
        if not key:
            score = 0.0
            rationale = "Step is entirely empty. Generator produced whitespace or failed to output tags."
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
        
        user_content = role.user_template.format(context=context, step=step)
        llm = make_llm(role.temperature, model_name=MODEL_NAME, base_url=BASE_URL, api_key=API_KEY,
                max_tokens=role.num_predict, with_tools=use_tools,
                enable_thinking=role.enable_thinking)
        messages = [
            SystemMessage(content=role.system_prompt),
            HumanMessage(content=user_content),
        ]
        if use_tools:
            result = generator_with_tools.invoke(
                {"messages": messages, "max_hops": 4, "max_total_tool_calls": 8},
                config={"configurable": {"llm": llm}},
            )
        else:
            result = {"messages": messages + [llm.invoke(messages)]}
        final_msg = result["messages"][-1]
        content = _message_text(final_msg)

        tks = count_chain_tokens(result["messages"])
        total_tokens += tks

        score, rationale, reliable = _parse_eval_response(content)
        any_reliable = any_reliable or reliable
        seen[key] = (score, rationale)
        scores.append(score)

        n_eval_tools = sum(1 for m in result["messages"] if getattr(m, "tool_calls", None))
        tool_note = f" (калькулятор вызван {n_eval_tools} раз)" if n_eval_tools > 0 else ""

        tag = "" if reliable else " ⚠️ [ОЦЕНКА НЕНАДЁЖНА]"
        print(f"    - Candidate {i+1} Score: {score:.4f}{tag}{tool_note} | Rationale: {rationale}")

    unreliable_streak = 0 if any_reliable else state.get('unreliable_eval_streak', 0) + 1
    if unreliable_streak > 0:
        print(f"  ⚠️  [ОЦЕНЩИК] Ни один ответ в этом раунде не распарсился — подряд "
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
            f"content (differed only in step number/wording). Stopping now instead of grinding through "
            f"the rest of the token budget ({state.get('tokens_used', 0)}/{state.get('token_budget', 0)})."
        )
    elif state.get('unreliable_eval_streak', 0) >= state.get('max_unreliable_evals', 3):
        reason = (
            f"Evaluator returned unparseable JSON {state['unreliable_eval_streak']} rounds in a row — "
            f"likely a server-side issue (e.g. response_format=json_object needs a guided-decoding backend "
            f"on vLLM), not a step-quality problem. Stopping instead of committing on noise "
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
            f"best candidate is still below score_threshold — stopping rather than committing low-quality "
            f"steps indefinitely until the token budget dies."
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
              f"(разница только в формулировке/номере шага) — подряд {stuck_streak}/{state.get('max_stuck_steps', 2)}.")

    answer = extract_answer(best_step)
    if answer:
        print(f"  -> Explicit answer found: {answer}")
        
    return {
        "steps": [best_step],       
        "final_answer": answer if answer else "",
        "in_recovery": False,       
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
    content, tks = _chat(messages, json_format=role.json_format, temperature=role.temperature,
                         num_predict=role.num_predict, enable_thinking=role.enable_thinking)

    is_valid, rationale, reliable = _parse_verify_response(content)
    if not reliable:
        print(f"  ⚠️  [НЕНАДЁЖНЫЙ ВЕРДИКТ] {rationale}")

    print(f"  -> Valid: {is_valid} | Rationale: {rationale}")
    
    return {
        "is_valid": is_valid,
        "verifier_rationale": rationale,
        "tokens_used": tks
    }


# ---------------------------------------------------------------------------
# 3. Conditional Edge Routers
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
        print(f"\n[Router] The token budget is exhausted ({tokens_used}/{token_budget}). "
              f"Commit the best available option without further attempts.")
        return "commit"

    unreliable_streak = state.get('unreliable_eval_streak', 0)
    max_unreliable = state.get('max_unreliable_evals', 3)
    if unreliable_streak >= max_unreliable:
        print(f"\n[Router] Evaluator has been unparseable for {unreliable_streak} rounds in a row "
              f"(limit {max_unreliable}). Giving up — this looks like a broken JSON mode, not a model quality issue.")
        return "give_up"


    # 1. Первичное срабатывание восстановления (если мы шли в один поток)
    if (best_score < state['score_threshold'] 
        and not state['in_recovery'] 
        and state['branch_mode'] == 'single'):
        
        if total_recoveries < max_recoveries:
            print(f"\n[Router] Best score {best_score:.4f} < Threshold {state['score_threshold']}. Initiating recovery.")
            return "recover"
        else:
            print(f"\n[Router] Score {best_score:.4f} still below threshold {state['score_threshold']}, but the "
                  f"recovery budget for the entire run ({max_recoveries}) is exhausted "
                  f"({total_recoveries} used). Giving up instead of committing low-quality steps indefinitely.")
            return "give_up"

    # 2. Логика повторных попыток (когда мы УЖЕ в режиме восстановления)
    if best_score < state['score_threshold'] and state['in_recovery']:
        
        # Если есть еще локальные попытки И глобальный бюджет не исчерпан
        if attempts < max_attempts and total_recoveries < max_recoveries:
            print(f"\n[Router] All {state.get('k_branches', 3)} branches failed. "
                  f"Triggering recovery attempt {attempts + 1}/{max_attempts} "
                  f"(Global recoveries used: {total_recoveries}/{max_recoveries}).")
            return "recover"
            
        # Если исчерпан глобальный бюджет на всю задачу
        elif total_recoveries >= max_recoveries:
            print(f"\n[Router] Global recovery budget ({max_recoveries}) exhausted during retries. Giving up.")
            return "give_up"
            
        # Если исчерпан лимит попыток для конкретно этого шага
        else:
            print(f"\n[Router] All {max_attempts} recovery attempts for this step failed. "
                  f"Giving up to prevent context poisoning.")
            return "give_up" 
            
    # 3. Успех
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
# 4. Graph Construction
# ---------------------------------------------------------------------------
def build_solver_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_step", generate_step)
    workflow.add_node("evaluate_steps", evaluate_steps)
    workflow.add_node("trigger_recovery", trigger_recovery)
    workflow.add_node("commit_step", commit_step)
    workflow.add_node("verify_solution", verify_solution)
    workflow.add_node("give_up", give_up)

    workflow.set_entry_point("generate_step")

    workflow.add_edge("generate_step", "evaluate_steps")
    workflow.add_conditional_edges("evaluate_steps", route_after_eval, {"recover": "trigger_recovery", "commit": "commit_step", "give_up": "give_up"})
    workflow.add_edge("trigger_recovery", "generate_step")
    workflow.add_conditional_edges("commit_step", route_after_commit, {"verify": "verify_solution", "generate": "generate_step", "give_up": "give_up"})
    workflow.add_edge("verify_solution", END)
    workflow.add_edge("give_up", END)

    return workflow.compile()