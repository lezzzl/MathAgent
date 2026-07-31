"""Запускает агента на IMO AnswerBench (по умолчанию — на ВСЕХ 400 задачах).

IMO-AnswerBench смешанный по формату ответа: ~57% — целые числа (228 задач), а
остальное символьные выражения с параметром ($2^{u-2}$), точные константы,
множественные ответы и функции ($P(x)=-1, P(x)=x+1$).

Раньше скрипт по умолчанию оставлял только числовое подмножество, потому что
math-verify надёжно сверяет лишь его. Но это делало прогон несравнимым с
результатами коллег, у которых в manifest.json стоит total_tasks=400, поэтому
теперь по умолчанию гоняется весь датасет.

Флаг --integer-only возвращает старое поведение (228 числовых задач) — полезно,
когда нужна чистая метрика без шума LLM-судьи.

Учтите при чтении результатов: на символьной части вердикт ставит LLM-судья, и
он заметно менее надёжен, чем math-verify. В сводке такие задачи помечены
verification_method = llm_judge / judge_unreliable.
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
        extra_flags=[
            (
                ["--integer-only"],
                {
                    "action": "store_true",
                    "help": "Прогнать только целочисленное подмножество (228 задач "
                            "из 400) — там math-verify надёжен и метрика чистая. "
                            "По умолчанию гоняется весь датасет.",
                },
            ),
            (
                ["--all-answers"],
                {
                    "action": "store_true",
                    "help": "Устаревший флаг: полный датасет теперь и так по "
                            "умолчанию. Оставлен, чтобы не ломать старые команды.",
                },
            ),
        ],
    )
    if args.all_answers:
        print("[config] --all-answers больше не нужен: весь датасет гоняется по "
              "умолчанию. Для старого поведения используйте --integer-only.")
    config = CONFIG if args.integer_only else CONFIG_ALL
    return run_benchmark(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
