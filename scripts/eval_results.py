"""Оценка качества бейзлайна по JSONL-результатам бенчмарков.

Считает accuracy (доля верных ответов) по файлам, которые пишет
run_all_benchmarks / run_*.py. Ответ модели сравнивается с эталоном:
берётся чистое поле `gold` (если runner его сохранил), иначе из `ground_truth`
извлекается содержимое последнего \\boxed{...} / \\framebox{...}.

Сравнение простое: из ответа модели и из эталона берётся содержимое \\boxed{...},
нормализуется (чистка LaTeX) и сверяется как строка (или как число).

Использование:
    python scripts/eval_results.py results/                 # все .jsonl рекурсивно
    python scripts/eval_results.py results/math500/*.jsonl  # конкретные файлы
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_boxed(text: str) -> str | None:
    r"""Вернуть содержимое последнего \boxed{...} / \framebox{...} / \fbox{...}."""
    for macro in (r"\boxed", r"\framebox", r"\fbox"):
        start = text.rfind(macro)
        if start == -1:
            continue
        brace = text.find("{", start)
        if brace == -1:
            continue
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace + 1 : i]
    return None


def normalize(answer: str) -> str:
    r"""Грубая нормализация LaTeX-ответа для сравнения строк."""
    s = answer.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("$", "").replace(" ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.rstrip(".")
    return s


def as_number(s: str) -> float | None:
    """Попытаться распарсить строку как число (для сравнения 480 vs 480.0 и т.п.)."""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def answers_equal(pred: str, gold: str) -> bool:
    """Совпадают ли \\boxed-ответ модели и эталон (строкой или числом)."""
    np, ng = normalize(pred), normalize(gold)
    if np == ng:
        return True
    xp, xg = as_number(np), as_number(ng)
    if xp is not None and xg is not None:
        return abs(xp - xg) < 1e-6
    return False


def gold_answer(record: dict[str, Any]) -> str | None:
    """Достать эталонный ответ: сперва чистое поле, иначе из ground_truth."""
    for key in ("gold", "answer"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    gt = record.get("ground_truth")
    if not gt:
        return None
    return extract_boxed(gt) or gt


def pred_answer(record: dict[str, Any]) -> str:
    """Ответ модели: если есть \\boxed — берём его, иначе весь solution."""
    solution = str(record.get("solution") or "")
    return extract_boxed(solution) or solution


def reasoning_fraction(solution: str) -> tuple[float, bool]:
    """Доля reasoning (<think>…</think>) в ответе и флаг обрыва думанья.

    Возвращает (доля_символов_в_think, оборвано_ли). Оборвано = открыт <think>,
    но нет </think> (модель упёрлась в max_tokens посреди рассуждения) — тогда
    считаем reasoning всем выходом.
    """
    if not solution or "<think>" not in solution:
        return 0.0, False
    start = solution.find("<think>") + len("<think>")
    end = solution.find("</think>", start)
    if end == -1:
        return 1.0, True
    return (end - start) / len(solution), False


def iter_jsonl(paths: list[Path]):
    """Пройтись по всем .jsonl из переданных файлов/директорий.

    При обходе папок пропускаем *.graded.jsonl (их пишет grade_results.py),
    иначе записи задвоятся: оригинал + сгрейженная копия.
    """
    for path in paths:
        if path.is_dir():
            files = [f for f in sorted(path.rglob("*.jsonl"))
                     if not f.name.endswith(".graded.jsonl")]
        else:
            files = [path]
        for file in files:
            with file.open(encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Accuracy по результатам бенчмарков")
    parser.add_argument("paths", nargs="+", type=Path, help="файлы .jsonl или папки")
    args = parser.parse_args()

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0, "correct": 0, "errors": 0, "latency": 0.0,
            # выходные токены (генерация), отдельно по верным/неверным ответам
            "out_sum": 0, "out_n": 0,
            "out_ok_sum": 0, "out_ok_n": 0,
            "out_bad_sum": 0, "out_bad_n": 0,
            # токены на reasoning (<think>…</think>) и обрывы думанья
            "think_sum": 0.0, "cut": 0,
        }
    )

    for record in iter_jsonl(args.paths):
        bench = record.get("benchmark_name", "unknown")
        row = stats[bench]
        row["n"] += 1
        meta = record.get("metadata") or {}
        if meta.get("error"):
            row["errors"] += 1
        row["latency"] += float(meta.get("latency_seconds") or 0.0)

        gold = gold_answer(record)
        correct = gold is not None and answers_equal(pred_answer(record), gold)
        if correct:
            row["correct"] += 1

        frac, cut = reasoning_fraction(str(record.get("solution") or ""))
        if cut:
            row["cut"] += 1
        out = (meta.get("usage") or {}).get("output_tokens") or 0
        if out:
            row["out_sum"] += out
            row["out_n"] += 1
            row["think_sum"] += out * frac
            key = "out_ok" if correct else "out_bad"
            row[f"{key}_sum"] += out
            row[f"{key}_n"] += 1

    if not stats:
        print("Не найдено ни одной записи.")
        return 1

    def mean(total: float, count: int) -> float:
        return total / count if count else 0.0

    print("Сравнение: по \\boxed-ответу (нормализация строки / числа)")
    print("out — сред. выходные токены (out✓/out✗ — на верных/неверных)")
    print("think — сред. токены на reasoning (<think>…</think>); "
          "% — доля от out; cut — задач с оборванным думаньем\n")
    header = (
        f"{'benchmark':<12} {'n':>5} {'correct':>8} {'accuracy':>9} {'errors':>7} "
        f"{'avg_s':>7} {'out':>8} {'out✓':>8} {'out✗':>8} {'think':>8} {'think%':>7} {'cut':>5}"
    )
    print(header)
    print("-" * len(header))
    total_n = total_c = total_e = 0
    for bench in sorted(stats):
        row = stats[bench]
        n, c, e = row["n"], row["correct"], row["errors"]
        think_avg = mean(row["think_sum"], row["out_n"])
        think_pct = row["think_sum"] / row["out_sum"] if row["out_sum"] else 0.0
        print(
            f"{bench:<12} {n:>5} {c:>8} {mean(c, n):>8.1%} {e:>7} "
            f"{mean(row['latency'], n):>7.2f} "
            f"{mean(row['out_sum'], row['out_n']):>8.0f} "
            f"{mean(row['out_ok_sum'], row['out_ok_n']):>8.0f} "
            f"{mean(row['out_bad_sum'], row['out_bad_n']):>8.0f} "
            f"{think_avg:>8.0f} {think_pct:>6.0%} {row['cut']:>5}"
        )
        total_n += n
        total_c += c
        total_e += e
    print("-" * len(header))
    print(f"{'ИТОГО':<12} {total_n:>5} {total_c:>8} {mean(total_c, total_n):>8.1%} {total_e:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
