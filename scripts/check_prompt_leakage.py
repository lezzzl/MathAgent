"""Ищет утечку задач бенчмарка в промпты ролей.

    python scripts/check_prompt_leakage.py                       # все yml
    python scripts/check_prompt_leakage.py conf/.../my-v4.yml    # конкретный

Зачем. Примеры в промптах писались «по мотивам» реальных задач, и часть из них
оказалась дословными постановками из бенчмарков: разбор палиндромов с суммой
цифр 13 — это AIME26 task 2, уравнение $D/v = D/(v+2)+1$ — AIME26 task 1, а
«$\\cos\\theta = 29/36$ при вопросе про $m+n$» — AIME26 task 5 целиком, вместе
с промежуточным значением и целью. Все затронутые задачи решались верно, то
есть замер был испорчен.

Скрипт проверяет два вида утечки:
  1. ЧИСЛА: любое число из промпта, совпадающее с эталоном какой-либо задачи;
  2. ТЕКСТ: редкие слова и формулы, общие у промпта и условия задачи.

Эталоны и условия берутся из results/**/*_verified.jsonl и файлов траекторий —
то есть из того, что реально прогонялось.

Код возврата 1, если найдена утечка: годится для запуска в CI или руками перед
прогоном.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Слова, которые встречаются в любом математическом тексте и ничего не выдают.
STOPWORDS = {
    "problem", "number", "numbers", "integer", "integers", "positive", "find",
    "step", "steps", "value", "values", "answer", "solution", "example", "let",
    "then", "there", "which", "where", "given", "such", "that", "with", "from",
    "digits", "digit", "sum", "total", "equation", "constant", "point", "points",
    "the", "and", "for", "are", "all", "one", "two", "each", "into", "when",
    "these", "this", "have", "been", "will", "your", "must", "only", "score",
    "rationale", "boxed", "frac", "sqrt", "cdot", "theta", "final", "check",
}


def load_benchmark_data() -> tuple[set[str], list[tuple[str, str, str]]]:
    """Возвращает (множество эталонов, список (бенч, id, условие))."""
    truths: set[str] = set()
    problems: list[tuple[str, str, str]] = []

    for path in glob.glob(str(ROOT / "results" / "*" / "*verified*.jsonl")):
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    gt = str(rec.get("ground_truth") or "").strip()
                    if gt:
                        truths.add(gt)
        except Exception:  # noqa: BLE001 — битый файл не должен ронять проверку
            continue

    for path in glob.glob(str(ROOT / "results" / "*" / "*_trajectory.json")):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        bench = (data.get("run") or {}).get("benchmark", Path(path).parent.name)
        for task in data.get("tasks", []):
            gt = str(task.get("ground_truth") or "").strip()
            if gt:
                truths.add(gt)
            problem = task.get("problem") or ""
            if problem:
                problems.append((bench, str(task.get("task_id")), problem))
    return truths, problems


# Длина совпадающей цепочки слов, начиная с которой это уже не совпадение
# случайной лексики, а дословная переписанная постановка задачи.
NGRAM = 5


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    """Цепочки из n значимых слов подряд.

    Мешок слов не годится: четыре общих слова вроде «determine/exactly/more/than»
    есть в любом условии, и проверка тонет в ложных срабатываниях. Дословная же
    цепочка «palindromes whose digits add up» встречается только там, откуда её
    скопировали.
    """
    seq = [w for w in re.findall(r"[A-Za-z]+", text.lower()) if w not in STOPWORDS]
    return {tuple(seq[i:i + n]) for i in range(max(0, len(seq) - n + 1))}


def check(path: Path, truths: set[str], problems) -> int:
    cfg = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("roles", {})
    findings = 0
    print(f"\n=== {path.name} ===")

    for role, spec in cfg.items():
        text = "\n".join(str(spec.get(k, "")) for k in ("system", "user_template"))
        body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))

        # 1. Числа, совпадающие с эталонами. Однозначные пропускаем: они есть в
        # любом тексте и дали бы сплошной шум.
        nums = {m.group(1) for m in re.finditer(r"(?<![\d.])(\d{2,4})(?![\d.])", body)}
        hits = sorted(n for n in nums if n in truths)
        if hits:
            findings += len(hits)
            print(f"  [ЧИСЛО] {role}: {hits} — совпадает с эталоном задачи бенчмарка")

        # 2. Дословные фрагменты условий.
        pg = ngrams(body)
        for bench, tid, problem in problems:
            common = pg & ngrams(problem)
            if common:
                findings += 1
                frag = " ".join(sorted(common)[0])
                print(f"  [ТЕКСТ] {role}: дословный фрагмент из {bench} task {tid}: "
                      f"...{frag}...")
    if not findings:
        print("  утечки не найдено")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="*", type=Path,
                    help="yml с промптами (по умолчанию — все qwen4b-версии)")
    args = ap.parse_args()

    paths = args.prompts or sorted(
        (ROOT / "conf" / "base" / "prompts").glob("agent-step-qwen4b-v*.yml"))
    truths, problems = load_benchmark_data()
    print(f"Эталонов собрано: {len(truths)}, условий задач: {len(problems)}")

    total = sum(check(Path(p), truths, problems) for p in paths)
    print(f"\nИТОГО находок: {total}")
    if total:
        print("Утечка означает, что промпт подсказывает ответ или постановку "
              "конкретной задачи бенчмарка — замер на таком промпте недостоверен.")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
