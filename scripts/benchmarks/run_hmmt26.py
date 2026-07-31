"""Запускает агента на задачах HMMT February 2026.

Раньше этот скрипт был подключён к benchmark_runner — раннеру для обычного
CoT-режима, у которого нет ни --pipeline, ни --trajectory, ни --token-budget.
Переведён на agent_benchmark_runner, как run_hmmt_feb2025 и все AIME-скрипты,
иначе пошаговый пайплайн из него не запускается.
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
    name="HMMT26",
    dataset_name="MathArena/hmmt_feb_2026",
    split="train",
    task_id_field="problem_idx",
    output_directory="hmmt26",
    ground_truth_field="answer",
    metadata_fields=("problem_type",),
)


def main() -> int:
    """Читает аргументы и передаёт конфигурацию HMMT26 общему runner."""
    args = parse_benchmark_args(__doc__ or "")
    return run_benchmark(CONFIG, args)


if __name__ == "__main__":
    raise SystemExit(main())
