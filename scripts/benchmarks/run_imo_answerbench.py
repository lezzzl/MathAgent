"""Запускает агента на ЧИСЛОВОМ подмножестве IMO AnswerBench.

IMO-AnswerBench смешанный по формату ответа: ~57% — целые числа, а остальное
символьные выражения с параметром ($2^{u-2}$), точные константы, множественные
ответы и функции ($P(x)=-1, P(x)=x+1$). Числовая сверка math-verify корректно
работает только на первой категории, поэтому по умолчанию скрипт прогоняет
только её — иначе ~40% задач получили бы "неверно" из-за формата эталона, а не
из-за качества решения.

Символьную часть правильно оценивать LLM-судьёй (как в оригинальном OpenEvals),
это отдельный пайплайн — здесь он не реализован.

Флаг --all-answers снимает фильтр и прогоняет весь датасет (например, чтобы
позже оценить символьные ответы вручную).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.agent_benchmark_runner import (
    BenchmarkConfig,
    parse_benchmark_args,
    run_benchmark,
)

_INT_ANSWER_RE = re.compile(r"^-?\d+$")


def is_integer_answer(short_answer: str) -> bool:
    """True, если Short Answer — одно целое число (в $...$ и с пробелами).

    Намеренно строгий: отсекает символьные ($2^{u-2}$), дроби, корни и
    множественные ответы ('3, 4') — всё, что числовая сверка не потянет
    надёжно. Пропускает большие целые (1012, 2026 и т.п.), диапазон не
    ограничивается, поэтому годится для IMO, а не только для AIME 0..999.
    """
    core = short_answer.strip().strip("$").strip().rstrip(".").strip()
    return bool(_INT_ANSWER_RE.match(core))


CONFIG = BenchmarkConfig(
    name="IMOAnswerBench",
    dataset_name="OpenEvals/IMO-AnswerBench",
    split="train",
    task_id_field="Problem ID",
    output_directory="imo_answerbench",
    problem_field="Problem",
    ground_truth_field="Short Answer",
    metadata_fields=("Category", "Subcategory", "Source"),
    answer_filter=is_integer_answer,
    answer_filter_name="integer-only",
)

CONFIG_ALL = BenchmarkConfig(
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
    args = parse_benchmark_args(
        __doc__ or "",
        default_model="Qwen/Qwen3.5-9B",
        extra_flags=[
            (
                ["--all-answers"],
                {
                    "action": "store_true",
                    "help": "Прогнать ВЕСЬ датасет, включая символьные и "
                            "множественные ответы (метрика math-verify на них "
                            "недостоверна). По умолчанию только целочисленные.",
                },
            )
        ],
    )
    config = CONFIG_ALL if args.all_answers else CONFIG
    return run_benchmark(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
