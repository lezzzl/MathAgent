"""Запускает агента на задачах GSM8K."""

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
    name="GSM8K",
    dataset_name="openai/gsm8k",
    split="test",                     # стандартный оценочный сплит
    task_id_field=None,               # в датасете нет поля id – будет использован индекс
    output_directory="gsm8k",
)


def main() -> int:
    args = parse_benchmark_args(__doc__ or "", default_model="llama3.2")
    return run_benchmark(CONFIG, args)


if __name__ == "__main__":
    raise SystemExit(main())