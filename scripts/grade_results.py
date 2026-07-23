"""Отдельный шаг проверки ответов после решения агента (детерминированный).

Разделяет solve (дорого — гоняем модель) и grade (дёшево — сверка строк).
Читает solved-JSONL, для каждой задачи извлекает \\boxed-ответ, сверяет с
эталоном и пишет рядом graded-файл, где к записи добавлены поля:
    predicted_answer — извлечённый ответ модели
    gold_answer      — эталон
    correct          — True/False

В конце печатает сводную таблицу по бенчмаркам: accuracy, среднее число
выходных токенов на задачу и среднее время. Логика сравнения переиспользуется
из eval_results.py (одна точка правды).

Использование:
    python scripts/grade_results.py results/hmmt25/RUN.jsonl
    python scripts/grade_results.py results/              # все .jsonl рекурсивно

Для входного X.jsonl пишется X.graded.jsonl. Уже готовые *.graded.jsonl
пропускаются (чтобы не грейдить повторно).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # чтобы найти eval_results

from eval_results import answers_equal, gold_answer, pred_answer

GRADED_SUFFIX = ".graded.jsonl"


def graded_path(source: Path) -> Path:
    """results/hmmt25/RUN.jsonl -> results/hmmt25/RUN.graded.jsonl."""
    return source.parent / (source.stem + GRADED_SUFFIX)


def grade_file(source: Path, stats: dict[str, dict[str, Any]]) -> int:
    """Проверить один файл, записать *.graded.jsonl, обновить stats. Вернуть n."""
    n = 0
    with source.open(encoding="utf-8") as src, graded_path(source).open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n += 1

            row = stats[record.get("benchmark_name", "unknown")]
            row["n"] += 1
            meta = record.get("metadata") or {}
            if meta.get("error"):
                row["errors"] += 1
            row["latency"] += float(meta.get("latency_seconds") or 0.0)
            out_tokens = (meta.get("usage") or {}).get("output_tokens") or 0
            if out_tokens:
                row["out_sum"] += out_tokens
                row["out_n"] += 1

            gold = gold_answer(record)
            pred = pred_answer(record)
            is_correct = gold is not None and answers_equal(pred, gold)
            if is_correct:
                row["correct"] += 1

            record["predicted_answer"] = pred
            record["gold_answer"] = gold
            record["correct"] = is_correct
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return n


def collect_files(paths: list[Path]) -> list[Path]:
    """Собрать .jsonl из файлов/папок, пропуская уже сгрейженные."""
    files: list[Path] = []
    for path in paths:
        found = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        files.extend(f for f in found if not f.name.endswith(GRADED_SUFFIX))
    return files


def print_table(stats: dict[str, dict[str, Any]]) -> None:
    """Сводка по бенчмаркам: accuracy, сред. токенов на задачу, сред. время."""
    def mean(total: float, count: int) -> float:
        return total / count if count else 0.0

    header = (
        f"{'benchmark':<12} {'n':>5} {'correct':>8} {'accuracy':>9} "
        f"{'errors':>7} {'avg_tok':>8} {'avg_s':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    tot_n = tot_c = tot_e = 0
    for bench in sorted(stats):
        row = stats[bench]
        n, c, e = row["n"], row["correct"], row["errors"]
        print(
            f"{bench:<12} {n:>5} {c:>8} {mean(c, n):>8.1%} {e:>7} "
            f"{mean(row['out_sum'], row['out_n']):>8.0f} {mean(row['latency'], n):>8.1f}"
        )
        tot_n += n
        tot_c += c
        tot_e += e
    print("-" * len(header))
    print(f"{'ИТОГО':<12} {tot_n:>5} {tot_c:>8} {mean(tot_c, tot_n):>8.1%} {tot_e:>7}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка ответов (grade) после solve")
    parser.add_argument("paths", nargs="+", type=Path, help="файлы .jsonl или папки")
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("Не найдено ни одного .jsonl для проверки.")
        return 1

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "errors": 0, "latency": 0.0, "out_sum": 0, "out_n": 0}
    )
    for source in files:
        grade_file(source, stats)
        print(f"graded: {source}  ->  {graded_path(source).name}")

    print_table(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
