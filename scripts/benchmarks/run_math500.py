"""Запускает модель на задачах MATH500 через OpenAI-совместимый API."""

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
    name="MATH500",
    dataset_name="HuggingFaceH4/MATH-500",
    split="test",
    task_id_field="unique_id",
    output_directory="math500",
    metadata_fields=("subject", "level"),
)


def main() -> int:
    """Читает аргументы и передаёт конфигурацию MATH500 общему runner."""
    args = parse_benchmark_args(__doc__ or "", default_model="qwen3.5:4b")
    return run_benchmark(CONFIG, args)


if __name__ == "__main__":
    raise SystemExit(main())
