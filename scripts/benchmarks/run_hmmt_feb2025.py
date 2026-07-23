"""Запускает агента на задачах HMMT February 2025 через OpenAI-совместимый API.

Формат ответов смешанный: 14 из 30 — целые, остальные точные символьные
константы ($\\frac{1}{576}$, $\\frac{9\\sqrt{23}}{23}$, $2^{25}\\cdot 26!$),
встречается и множественный ответ. В отличие от IMO-AnswerBench это конкретные
числа, а не выражения со свободным параметром, поэтому sympy сравнивает их
надёжно — фильтр по формату здесь не нужен, но сверять результаты следует
гибридным verify_imo_answers.py, а не числовым verify_answers.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.agent_benchmark_runner import (
    BenchmarkConfig,
    parse_benchmark_args,
    run_benchmark,
)

CONFIG = BenchmarkConfig(
    name="HMMT_Feb2025",
    dataset_name="MathArena/hmmt_feb_2025",
    split="train",
    task_id_field="problem_idx",
    output_directory="hmmt_feb2025",
    metadata_fields=("problem_type",),
    # problem_field/ground_truth_field — дефолтные problem/answer, как у AIME.
)


def main() -> int:
    """Читает аргументы и передаёт конфигурацию HMMT общему runner."""
    args = parse_benchmark_args(__doc__ or "", default_model="Qwen/Qwen3.5-9B")
    return run_benchmark(CONFIG, args)


if __name__ == "__main__":
    raise SystemExit(main())
