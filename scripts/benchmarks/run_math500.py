"""Запускает агента на задачах MATH-500 через OpenAI-совместимый API.

Формат ответов самый разнообразный из бенчмарков проекта: 311 целых, а среди
остальных 189 — дроби, углы ($90^\\circ$), координаты, символьные выражения
($p-q$) и даже текст ($\\text{Evelyn}$). math-verify надёжно закрывает целые и
большинство символьных (само-сверка проходит на 449/500), но текст, единицы и
координаты требуют семантического судьи. Поэтому результаты следует сверять
гибридным verify_imo_answers.py (math-verify + LLM-судья), а не числовым
verify_answers.py.

Датасет большой (500 задач). С reasoning-моделью полный прогон измеряется
часами — начинайте с --limit и запускайте в tmux с --resume.
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
    name="MATH500",
    dataset_name="HuggingFaceH4/MATH-500",
    split="test",
    task_id_field="unique_id",
    output_directory="math500",
    metadata_fields=("subject", "level"),
    # problem_field/ground_truth_field — дефолтные problem/answer.
)


def main() -> int:
    """Читает аргументы и передаёт конфигурацию MATH-500 общему runner."""
    args = parse_benchmark_args(__doc__ or "", default_model="Qwen/Qwen3.5-9B")
    return run_benchmark(CONFIG, args)


if __name__ == "__main__":
    raise SystemExit(main())
