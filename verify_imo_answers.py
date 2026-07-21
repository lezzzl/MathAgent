"""Гибридная верификация ответов IMO-AnswerBench: math-verify + LLM-судья.

Зачем отдельный скрипт, а не verify_answers.py: у IMO-AnswerBench ~40% ответов
символьные ($2^{u-2}$), множественные или функции. Числовая сверка math-verify
их проваливает по одной причине — расхождение имён переменных: эталон
$\\lfloor\\log_2 a\\rfloor+1$ и ответ модели с буквой n для sympy не равны, хотя
это одна формула. Понять эквивалентность может только семантический судья, что
и делают авторы бенчмарка.

Пайплайн на задачу:
  1. math-verify (parse + verify) — числа, множества в LaTeX, эквивалентные
     алгебраические формы решаются точно и бесплатно;
  2. если не подтвердилось — LLM-судья решает об эквивалентности.

Метод сверки пишется в поле verification_method, чтобы точность можно было
разложить на «алгоритмическую» и «судейскую» части.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import requests

from answer_utils import extract_boxed
from math_verify import parse, verify

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


JUDGE_SYSTEM = """You are a strict grader for a mathematics competition. You are given a REFERENCE answer and a STUDENT answer. Decide whether they are mathematically EQUIVALENT.

EQUIVALENT — they denote the same mathematical object, even if written differently:
- different free-variable names: floor(log2(a))+1 is equivalent to floor(log2(n))+1
- algebraically equal forms: 2^(u-2) is equivalent to (2^u)/4
- the same set of values in any order: {-2/3, 0, 2/3} is equivalent to {0, 2/3, -2/3}
- the same functions up to naming: g(x)=2x^3+c is equivalent to f(t)=2t^3+C

NOT EQUIVALENT — they denote different values, a different set of solutions (any solution missing or extra), or a different function.

Do NOT try to solve any problem. Compare ONLY the two answers as written.
Respond with a single JSON object and nothing else: {"equivalent": true} or {"equivalent": false}"""

JUDGE_USER = "REFERENCE answer: {reference}\nSTUDENT answer: {student}"

_EQUIV_RE = re.compile(r'"?equivalent"?\s*[:=]\s*"?(true|false)"?', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Ступень 1: math-verify
# ---------------------------------------------------------------------------
def _for_parse(s: str) -> str:
    """Оборачивает голый LaTeX в $...$ для надёжного распознавания.

    math-verify парсит '2\\sqrt{2}' как False, но '$2\\sqrt{2}$' — верно:
    содержимое \\boxed извлекается без обёртки и теряет распознавание как LaTeX.
    Если обёртка ($ или \\boxed) уже есть, не трогаем.
    """
    s = s.strip()
    if "$" in s or "\\boxed" in s:
        return s
    return f"${s}$"


def math_verify_equal(reference: str, student: str, timeout: Optional[int]) -> Optional[bool]:
    """True/False по math-verify, либо None если хоть одна сторона не распарсилась.

    None означает «алгоритм не применим» — задача уходит к судье, а не считается
    неверной. Так мы не штрафуем ответ за то, что sympy не осилил его форму.
    """
    try:
        ref = parse(_for_parse(reference), parsing_timeout=timeout)
        ans = parse(_for_parse(student), parsing_timeout=timeout)
    except Exception:
        return None
    if not ref or not ans:
        return None
    try:
        return bool(verify(ref, ans, timeout_seconds=timeout))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ступень 2: LLM-судья
# ---------------------------------------------------------------------------
def llm_judge(reference: str, student: str, cfg: dict[str, Any]) -> tuple[Optional[bool], str]:
    """Спрашивает у модели-судьи об эквивалентности. (verdict, raw_or_error)."""
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(reference=reference, student=student)},
        ],
        "temperature": 0.0,
        "max_tokens": cfg["max_tokens"],
    }
    if cfg["thinking"] is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": cfg["thinking"]}

    try:
        resp = requests.post(
            f"{cfg['base_url'].rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {cfg['api_key']}"},
            timeout=cfg["timeout"],
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
    except Exception as exc:  # noqa: BLE001
        return None, f"judge request failed: {type(exc).__name__}: {exc}"

    text = (msg.get("content") or "").strip()
    if not text:
        text = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()

    match = _EQUIV_RE.search(text)
    if not match:
        return None, f"unparseable judge reply: {text[:200]!r}"
    return match.group(1).lower() == "true", text[:300]


# ---------------------------------------------------------------------------
# Одна запись
# ---------------------------------------------------------------------------
def verify_one(record: dict[str, Any], args, judge_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    reference = str(record.get(args.gt_key, "") or "")
    # solution — уже извлечённый финальный ответ раннера. Если вдруг это полный
    # текст с \boxed{}, достаём содержимое; иначе берём как есть.
    solution = str(record.get(args.answer_key) or "").strip()
    student = extract_boxed(solution) if "\\boxed" in solution else solution

    out["verification_method"] = None
    out["is_correct"] = False

    if not student or not reference:
        out["verification_method"] = "no_answer"
        return out

    # Ступень 1: math-verify
    mv = math_verify_equal(reference, student, args.timeout)
    if mv is True:
        out["is_correct"] = True
        out["verification_method"] = "math_verify"
        return out
    if mv is False and args.no_judge:
        out["verification_method"] = "math_verify"
        return out

    # Ступень 2: судья (для не-True при math-verify), если не отключён
    if args.no_judge:
        out["verification_method"] = "math_verify_unparsed"
        return out

    verdict, raw = llm_judge(reference, student, judge_cfg)
    out["judge_raw"] = raw
    if verdict is None:
        out["verification_method"] = "judge_unreliable"
        out["is_correct"] = False
    else:
        out["is_correct"] = verdict
        out["verification_method"] = "llm_judge"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--answer-key", default="solution")
    ap.add_argument("--gt-key", default="ground_truth")
    ap.add_argument("--workers", type=int, default=8, help="Параллельных judge-запросов")
    ap.add_argument("--no-judge", action="store_true", help="Только math-verify, без судьи")
    ap.add_argument(
        "--timeout", type=int, default=None,
        help="Таймаут парсинга/сверки math-verify. На Windows принудительно "
             "отключается (иначе math-verify не парсит ничего), на Linux 5с.",
    )
    ap.add_argument("--judge-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8002/v1"))
    ap.add_argument("--judge-model", default=os.getenv("MODEL", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--judge-api-key", default=os.getenv("OPENAI_API_KEY", "token-abc123"))
    ap.add_argument("--judge-timeout", type=float, default=300.0)
    ap.add_argument("--judge-max-tokens", type=int, default=2000)
    ap.add_argument(
        "--judge-thinking", choices=["on", "off", "none"], default="off",
        help="Размышления судьи. 'off' быстро и обычно достаточно; 'on' точнее на "
             "сложных формах, но дороже; 'none' — не трогать (для не-thinking моделей).",
    )
    args = ap.parse_args()

    if args.timeout is None and sys.platform == "win32":
        args.timeout = None  # на Windows math-verify timeout нерабочий
    elif args.timeout is None:
        args.timeout = 5

    judge_cfg = {
        "base_url": args.judge_base_url,
        "model": args.judge_model,
        "api_key": args.judge_api_key,
        "timeout": args.judge_timeout,
        "max_tokens": args.judge_max_tokens,
        "thinking": {"on": True, "off": False, "none": None}[args.judge_thinking],
    }

    records = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    output = args.output or args.input.with_name(args.input.stem + "_imoverified.jsonl")

    if not args.no_judge:
        print(f"[judge] модель {judge_cfg['model']} @ {judge_cfg['base_url']} "
              f"| thinking={args.judge_thinking} | параллельно {args.workers}")

    # math-verify — быстрая последовательная ступень; judge-запросы параллелим.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda r: verify_one(r, args, judge_cfg), records))

    with output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    from collections import Counter
    by_method = Counter(r["verification_method"] for r in results)
    solved_by = Counter(r["verification_method"] for r in results if r["is_correct"])

    print("=" * 60)
    print(f"Точность: {correct}/{total} = {correct/total*100:.1f}%" if total else "нет записей")
    print("Разбивка по методу сверки (все / из них верных):")
    for method in by_method:
        print(f"  {method:22} {by_method[method]:3} / {solved_by.get(method,0)}")
    print(f"Результат сохранён: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
