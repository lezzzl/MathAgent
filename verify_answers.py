"""
verify_answers.py

Проверяет ответы, полученные от агента, против ground truth ответов
из бенчмарков с единственным числовым ответом (AIME24/25 и подобные),
используя библиотеку math-verify (https://github.com/huggingface/Math-Verify).

Ожидаемый формат ВХОДНОГО json-файла — список словарей, например:

[
  {
    "id": "2024-I-1",
    "model": "qwen2.5-32b",
    "pipeline": "cot_baseline",
    "benchmark": "aime24",
    "ground_truth": "204",
    "model_answer": "... рассуждения ... Итоговый ответ: \\boxed{204}"
  },
  ...
]

Названия полей можно переопределить через аргументы командной строки
(--id-key, --gt-key, --answer-key), если в вашем pipeline они называются
иначе.

ВЫХОДНОЙ json — тот же список словарей с добавленными полями:
  - is_correct            (bool)
  - parsed_model_answer   (строковое представление распарсенного ответа модели)
  - parsed_ground_truth   (строковое представление распарсенного ground truth)
  - verification_error    (добавляется только если во время парсинга/сверки
                            произошла ошибка)
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from math_verify import parse, verify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_and_verify(model_answer: str, ground_truth, timeout_seconds) -> dict:
    """
    Парсит ответ модели (ожидается \\boxed{...} где-то в тексте) и ground truth,
    затем сверяет их через math_verify.verify.

    Возвращает dict:
      is_correct, parsed_answer, parsed_ground_truth, error
    """
    result = {
        "is_correct": False,
        "parsed_answer": None,
        "parsed_ground_truth": None,
        "error": None,
    }

    model_answer = model_answer if isinstance(model_answer, str) else str(model_answer)
    gt_str = str(ground_truth)

    boxed_matches = list(re.finditer(r'\\boxed{', model_answer))
    if boxed_matches:
        last_boxed_idx = boxed_matches[-1].start()
        model_answer = model_answer[last_boxed_idx:]

    try:
        parsed_gt = parse(gt_str, parsing_timeout=timeout_seconds)
        parsed_answer = parse(model_answer, parsing_timeout=timeout_seconds)

        result["parsed_ground_truth"] = str(parsed_gt)
        result["parsed_answer"] = str(parsed_answer)

        if not parsed_answer:
            # math_verify не смог найти \boxed{}/выражение в ответе модели
            result["error"] = "no_answer_extracted_from_model_output"
            return result

        if not parsed_gt:
            result["error"] = "no_answer_extracted_from_ground_truth"
            return result

        result["is_correct"] = bool(
            verify(parsed_gt, parsed_answer, timeout_seconds=timeout_seconds)
        )

    except Exception as e:  
        result["error"] = f"{type(e).__name__}: {e}"
        logger.warning("Ошибка верификации: %s", result["error"])

    return result


def process_file(
    input_path: Path,
    output_path: Path,
    id_key: str,
    answer_key: str,
    gt_key: str,
    group_by_keys: list[str],
    timeout_seconds,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Ожидался список словарей (список задач) в корне json-файла, "
            f"получен {type(data).__name__}."
        )

    total = len(data)
    correct = 0
    errors = 0
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Элемент #%d не является словарём, пропускаю.", i)
            continue

        model_answer = item.get(answer_key)
        ground_truth = item.get(gt_key)
        item_id = item.get(id_key, f"index_{i}")

        if model_answer is None:
            logger.warning("id=%s: отсутствует поле '%s' с ответом модели.", item_id, answer_key)
        if ground_truth is None:
            logger.warning("id=%s: отсутствует поле '%s' с ground truth.", item_id, gt_key)

        verification = extract_and_verify(model_answer or "", ground_truth, timeout_seconds)

        item["is_correct"] = verification["is_correct"]
        item["parsed_model_answer"] = verification["parsed_answer"]
        item["parsed_ground_truth"] = verification["parsed_ground_truth"]
        if verification["error"]:
            item["verification_error"] = verification["error"]
            errors += 1

        if verification["is_correct"]:
            correct += 1

        for key in group_by_keys:
            group_val = item.get(key, f"<нет поля '{key}'>")
            group_stats[(key, group_val)]["total"] += 1
            if verification["is_correct"]:
                group_stats[(key, group_val)]["correct"] += 1

        logger.info(
            "[%d/%d] id=%s correct=%s%s",
            i + 1,
            total,
            item_id,
            verification["is_correct"],
            f" ({verification['error']})" if verification["error"] else "",
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    accuracy = correct / total if total else 0.0
    logger.info("=" * 60)
    logger.info("Готово. Итоговая точность: %d/%d = %.2f%%", correct, total, accuracy * 100)
    if errors:
        logger.warning(
            "%d элементов с ошибками верификации (см. поле 'verification_error').", errors
        )

    if group_by_keys:
        logger.info("Разбивка по группам:")
        for (key, val), stats in sorted(group_stats.items()):
            g_acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
            logger.info(
                "  %s=%s: %d/%d = %.2f%%", key, val, stats["correct"], stats["total"], g_acc * 100
            )

    logger.info("Результат сохранён в: %s", output_path)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Верификация ответов агента против ground truth через math-verify."
    )
    arg_parser.add_argument("input", type=Path, help="Путь к входному json-файлу от агента.")
    arg_parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Путь к выходному json (по умолчанию: <input>_verified.json).",
    )
    arg_parser.add_argument("--id-key", default="id", help="Название поля с id задачи.")
    arg_parser.add_argument(
        "--answer-key", default="model_answer",
        help="Название поля с сырым ответом модели (должен содержать \\boxed{...}).",
    )
    arg_parser.add_argument(
        "--gt-key", default="ground_truth", help="Название поля с ground truth ответом."
    )
    arg_parser.add_argument(
        "--group-by", default="model,pipeline,benchmark",
        help="Список полей через запятую для разбивки точности по группам "
             "(например: model,benchmark). Пусто — без разбивки.",
    )
    arg_parser.add_argument(
        "--timeout", type=int, default=5,
        help="Таймаут парсинга/сверки в секундах на один ответ (по умолчанию 5).",
    )
    arg_parser.add_argument(
        "--no-timeout", action="store_true",
        help="Отключить таймаут полностью (на Windows это избавляет от накладных "
             "расходов multiprocessing, но без защиты от зависания на патологическом вводе).",
    )
    args = arg_parser.parse_args()

    output_path = args.output or args.input.with_name(args.input.stem + "_verified.json")
    group_by_keys = [k.strip() for k in args.group_by.split(",") if k.strip()]
    timeout_seconds = None if args.no_timeout else args.timeout

    process_file(
        input_path=args.input,
        output_path=output_path,
        id_key=args.id_key,
        answer_key=args.answer_key,
        gt_key=args.gt_key,
        group_by_keys=group_by_keys,
        timeout_seconds=timeout_seconds,
    )


if __name__ == "__main__":
    main()
