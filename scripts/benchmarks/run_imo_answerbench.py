"""Запускает агента на задачах IMO AnswerBench через OpenAI-совместимый API."""

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
    name="IMOAnswerBench",
    dataset_name="OpenEvals/IMO-AnswerBench",
    split="train",
    task_id_field="Problem ID",
    output_directory="imo_answerbench",
    problem_field="Problem",
    ground_truth_field="Short Answer",
    metadata_fields=("Category", "Subcategory", "Source"),
)


def main() -> int:
    """Читает аргументы и передаёт конфигурацию IMO AnswerBench общему runner."""
    args = parse_benchmark_args(__doc__ or "", default_model="Qwen/Qwen3.5-9B")
    return run_benchmark(CONFIG, args)


if __name__ == "__main__":
    raise SystemExit(main())
