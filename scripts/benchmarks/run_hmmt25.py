"""Запускает модель на задачах HMMT February 2025 через OpenAI-совместимый API."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.benchmark_runner import (
    BenchmarkConfig,
    parse_benchmark_args,
    run_benchmark,
)

# MathArena/hmmt_feb_2025: поля problem_idx / problem / answer / problem_type,
# split train, 30 задач, ответы короткие. Авторского решения нет.
CONFIG = BenchmarkConfig(
    name="HMMT25",
    dataset_name="MathArena/hmmt_feb_2025",
    split="train",
    task_id_field="problem_idx",
    output_directory="hmmt25",
    metadata_fields=("problem_type",),
)


def main() -> int:
    """Читает аргументы и передаёт конфигурацию HMMT25 общему runner."""
    args = parse_benchmark_args(__doc__ or "", default_model="qwen3.5:4b")
    return run_benchmark(CONFIG, args)


if __name__ == "__main__":
    raise SystemExit(main())
