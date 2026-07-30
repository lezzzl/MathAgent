

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator
import re

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

        result['extracted_text'] = str(parsed_answer) if parsed_answer is not None else None

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


def process_sample(
    item: dict[str, Any],
    id_key: str,
    answer_key: str,
    gt_key: str,
    timeout_seconds,
) -> dict[str, Any]:
    """
    Обрабатывает одну запись бенчмарка (одну задачу): извлекает ответ модели,
    сверяет с ground truth через math_verify и возвращает новую запись с
    добавленными полями результата.

    item — dict в формате, который пишет benchmark_runner.build_record().
    Исходный dict не мутируется — возвращается копия с
    дополнительными полями, все остальные поля (benchmark_name, model_name,
    metadata и т.д.) сохраняются как есть.
    """
    result_item = dict(item)

    model_answer = item.get(answer_key)
    ground_truth = item.get(gt_key)
    item_id = item.get(id_key, "<no id>")

    if model_answer is None:
        logger.warning("id=%s: отсутствует поле '%s' с ответом модели.", item_id, answer_key)
    if ground_truth is None:
        logger.warning("id=%s: отсутствует поле '%s' с ground truth.", item_id, gt_key)

    verification = extract_and_verify(model_answer or "", ground_truth, timeout_seconds)
    
    result_item["is_correct"] = verification["is_correct"]
    result_item["extracted_answer_text"] = verification["extracted_text"]
    result_item["parsed_model_answer"] = verification["parsed_answer"]
    result_item["parsed_ground_truth"] = verification["parsed_ground_truth"]
    if verification["error"]:
        result_item["verification_error"] = verification["error"]

    logger.info(
        "id=%s correct=%s%s",
        item_id,
        verification["is_correct"],
        f" ({verification['error']})" if verification["error"] else "",
    )

    return result_item


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """
    Построчно читает записи входного файла. Формат определяется по расширению:
      .jsonl -> построчный JSON, одна запись на строку 
      иначе  -> JSON-список словарей
    """
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Некорректный JSON в строке {line_no} файла {path}: {e}"
                    ) from e
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                "Ожидался список словарей (список задач) в корне json-файла, "
                f"получен {type(data).__name__}."
            )
        yield from data


def process_file(
    input_path: Path,
    output_path: Path,
    id_key: str,
    answer_key: str,
    gt_key: str,
    group_by_keys: list[str],
    timeout_seconds,
) -> None:
    """
    Проходит по всем записям input_path, сверяет каждую через process_sample
    и пишет результат в output_path.

    Формат вывода определяется расширением output_path: .jsonl -> построчно
    (с flush() после каждой записи, как в benchmark_runner.py — результаты не
    теряются при обрыве прогона), иначе -> JSON-список.
    """
    output_is_jsonl = output_path.suffix.lower() == ".jsonl"

    total = 0
    correct = 0
    errors = 0
    group_stats: dict[tuple[str, Any], dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    json_list_buffer: list[dict[str, Any]] = []

    jsonl_file = output_path.open("w", encoding="utf-8") if output_is_jsonl else None
    try:
        for i, item in enumerate(iter_records(input_path)):
            if not isinstance(item, dict):
                logger.warning("Запись #%d не является объектом (dict), пропускаю.", i)
                continue

            total += 1
            processed = process_sample(item, id_key, answer_key, gt_key, timeout_seconds)

            if processed.get("is_correct"):
                correct += 1
            if processed.get("verification_error"):
                errors += 1

            for key in group_by_keys:
                group_val = processed.get(key, f"<нет поля '{key}'>")
                group_stats[(key, group_val)]["total"] += 1
                if processed.get("is_correct"):
                    group_stats[(key, group_val)]["correct"] += 1

            if jsonl_file is not None:
                jsonl_file.write(json.dumps(processed, ensure_ascii=False) + "\n")
                jsonl_file.flush()
            else:
                json_list_buffer.append(processed)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    if jsonl_file is None:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(json_list_buffer, f, ensure_ascii=False, indent=2)

    accuracy = correct / total if total else 0.0
    logger.info("=" * 60)
    logger.info("Готово. Итоговая точность: %d/%d = %.2f%%", correct, total, accuracy * 100)
    if errors:
        logger.warning(
            "%d элементов с ошибками верификации (см. поле 'verification_error').", errors
        )

    if group_by_keys:
        logger.info("Разбивка по группам:")
        for (key, val), stats in sorted(group_stats.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            g_acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
            logger.info(
                "  %s=%s: %d/%d = %.2f%%", key, val, stats["correct"], stats["total"], g_acc * 100
            )

    logger.info("Результат сохранён в: %s", output_path)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Верификация ответов агента против ground truth через math-verify."
    )
    arg_parser.add_argument(
        "input", type=Path,
        help="Путь к входному файлу от агента (.jsonl как у benchmark_runner.py, или .json со списком).",
    )
    arg_parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Путь к выходному файлу (по умолчанию: <input>_verified<расширение исходного файла>).",
    )
    arg_parser.add_argument(
        "--id-key", default="task_id", help="Название поля с id задачи (по умолчанию как в benchmark_runner.py)."
    )
    arg_parser.add_argument(
        "--answer-key", default="solution",
        help="Название поля с сырым ответом модели (должен содержать \\boxed{...}).",
    )
    arg_parser.add_argument(
        "--gt-key", default="ground_truth", help="Название поля с ground truth ответом."
    )
    arg_parser.add_argument(
        "--group-by", default="model_name,benchmark_name",
        help="Список полей через запятую для разбивки точности по группам "
             "(например: model_name,benchmark_name). Пусто — без разбивки.",
    )
    arg_parser.add_argument(
        "--timeout", type=int, default=None,
        help="Таймаут парсинга/сверки в секундах на один ответ. По умолчанию "
             "отключён на Windows (см. --no-timeout) и равен 5 на остальных ОС.",
    )
    arg_parser.add_argument(
        "--no-timeout", action="store_true",
        help="Отключить таймаут полностью. На Windows это ОБЯЗАТЕЛЬНО: math_verify "
             "реализует таймаут через multiprocessing, дочерние процессы падают с "
             "WinError, и parse() молча возвращает пустой результат — не парсится "
             "вообще ничего.",
    )
    args = arg_parser.parse_args()

    output_path = args.output or args.input.with_name(
        args.input.stem + "_verified" + args.input.suffix
    )
    group_by_keys = [k.strip() for k in args.group_by.split(",") if k.strip()]

    # Таймаут math_verify нерабочий на Windows: его multiprocessing-обёртка не
    # поднимает дочерний процесс, parse() возвращает [] на ЛЮБОМ входе, и весь
    # файл получает is_correct=False. Поэтому там он выключен по умолчанию.
    if args.no_timeout:
        timeout_seconds = None
    elif args.timeout is not None:
        timeout_seconds = args.timeout
    elif sys.platform == "win32":
        timeout_seconds = None
        logger.info(
            "Windows: таймаут парсинга отключён автоматически (иначе math_verify "
            "не распарсит ни одного ответа). Явно задать: --timeout N."
        )
    else:
        timeout_seconds = 5

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
