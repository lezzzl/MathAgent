"""Запускает модель на задачах AIME 2025 через OpenAI-совместимый API."""

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
    name="AIME25",
    dataset_name="Sunny8781/AIME2025_w_solution",
    split="test",
    task_id_field="id",
    output_directory="aime25",
)


def main() -> int:
    """Читает аргументы и передаёт конфигурацию AIME25 общему runner."""
    args = parse_benchmark_args(__doc__ or "")
    return run_benchmark(CONFIG, args)


if __name__ == "__main__":
    raise SystemExit(main())
