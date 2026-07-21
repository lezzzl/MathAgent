"""Запускает модель на задачах HMMT February 2026 через OpenAI-совместимый API."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.benchmark_runner import (
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
