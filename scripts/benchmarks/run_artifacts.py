import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Стандартные каталоги артефактов относительно корня проекта
ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "results" / "runs"
LOGS_ROOT = ROOT / "logs"

# Допустимые символы защищают пути запуска от выхода за рабочие каталоги
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def utc_now() -> str:
    """Возвращает UTC-время для единообразных временных меток в manifest."""
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(model_name: str) -> str:
    """Создаёт уникальный и безопасный для пути run_id из модели и времени."""
    # Преобразуем название модели в безопасную часть имени каталога
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()

    # Временная метка различает последовательные запуски одной модели
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_slug}-{timestamp}"


def validate_run_id(run_id: str) -> str:
    """Запрещает символы, с которыми run_id мог бы выйти из каталога запуска."""
    # Отклоняем пробелы, слеши и другие символы до построения путей
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )
    return run_id


def get_manifest_path(run_id: str) -> Path:
    """Строит стандартный путь к manifest, чтобы все скрипты находили его одинаково."""
    return RUNS_ROOT / validate_run_id(run_id) / "manifest.json"


def get_benchmark_output_path(run_id: str, output_name: str) -> Path:
    """Строит путь к JSONL бенчмарка внутри каталога конкретного запуска."""
    return RUNS_ROOT / validate_run_id(run_id) / f"{output_name}.jsonl"


def configure_run_logger(run_id: str) -> logging.Logger:
    """Создаёт единый logger запуска, который пишет и в терминал, и в runner.log.

    Повторный вызов с тем же run_id возвращает уже настроенный logger, чтобы не
    дублировать строки при последовательном запуске нескольких бенчмарков.
    """
    logger = logging.getLogger(f"mathagent.benchmark.{validate_run_id(run_id)}")

    # Не добавляем повторные обработчики для одного run_id
    if logger.handlers:
        return logger

    # Все бенчмарки одного запуска дописывают события в общий runner.log
    log_path = LOGS_ROOT / run_id / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # Один обработчик выводит события в терминал, второй сохраняет их в файл
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Запрещаем повторный вывод через родительские logger
    logger.propagate = False
    return logger


def read_json(path: Path) -> dict[str, Any]:
    """Читает manifest как словарь для последующего обновления его состояния."""
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Безопасно заменяет manifest, не оставляя повреждённый JSON при прерывании.

    Сначала данные полностью записываются и синхронизируются во временный файл,
    после чего временный файл атомарно заменяет основной.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Сначала полностью записываем новое содержимое во временный файл
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    # Атомарная замена не оставляет частично записанный manifest
    temporary_path.replace(path)


def ensure_manifest(
    path: Path,
    expected: dict[str, Any],
    benchmark_name: str,
    benchmark_details: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    """Подготавливает manifest перед запуском отдельного бенчмарка.

    Для нового run_id функция создаёт manifest. При resume она проверяет, что
    модель, pipeline, prompt и generation-параметры не изменились, а затем
    отмечает выбранный бенчмарк и весь запуск как выполняющиеся.
    """
    if path.exists():
        # При продолжении проверяем параметры, влияющие на результат эксперимента
        manifest = read_json(path)
        for field in (
            "model",
            "pipeline",
            "prompt",
            "prompt_version",
            "role",
            "generation",
        ):
            if manifest.get(field) != expected.get(field):
                raise ValueError(
                    f"Cannot continue run: manifest field '{field}' does not match"
                )
        if (
            "pipeline_config" in manifest or "pipeline_config" in expected
        ) and manifest.get("pipeline_config") != expected.get("pipeline_config"):
            raise ValueError(
                "Cannot continue run: manifest field 'pipeline_config' does not match"
            )

        # Технические параметры разрешено менять между сегментами resume
        manifest["runtime"] = expected["runtime"]
        manifest["serving"] = expected["serving"]
    else:
        # Для нового run_id создаём пустой manifest со статусом running
        manifest = {
            "schema_version": 1,
            **expected,
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "benchmarks": {},
            "summary": {},
        }

    # Регистрируем текущий бенчмарк или обновляем сведения о нём
    benchmark = manifest.setdefault("benchmarks", {}).setdefault(benchmark_name, {})
    benchmark.update(benchmark_details)

    # Счётчик показывает сколько раз запуск продолжали через --resume
    if resume:
        benchmark["resume_count"] = int(benchmark.get("resume_count", 0)) + 1

    # Перед запросами отмечаем бенчмарк и весь run как выполняющиеся
    benchmark["status"] = "running"
    benchmark["updated_at"] = utc_now()
    manifest["status"] = "running"
    manifest["updated_at"] = utc_now()
    write_json_atomic(path, manifest)
    return manifest


def update_benchmark_manifest(
    path: Path,
    benchmark_name: str,
    status: str,
    summary: dict[str, Any],
    segment: dict[str, Any] | None = None,
) -> None:
    """Сохраняет статус и метрики бенчмарка, затем пересчитывает итог всего run.

    Это позволяет одному manifest содержать как результаты каждого датасета,
    так и суммарное число задач и токенов по всем уже запущенным бенчмаркам.
    Отдельные сегменты сохраняют длительность первоначального запуска и resume.
    """
    # Загружаем предыдущие сведения чтобы не потерять историю resume
    manifest = read_json(path)
    previous_benchmark = manifest["benchmarks"].get(benchmark_name, {})

    # Рассчитываем накопительный throughput текущего бенчмарка
    summary = dict(summary)
    wall_time = float(summary.get("wall_time_seconds", 0.0))
    processed_tasks = int(summary.get("successful_tasks", 0)) + int(
        summary.get("failed_tasks", 0)
    )
    summary["tasks_per_second"] = (
        round(processed_tasks / wall_time, 4) if wall_time else 0.0
    )

    # Добавляем метрики текущего запуска в историю отдельных сегментов
    segments = list(previous_benchmark.get("segments", []))
    if segment is not None:
        segments.append({**segment, "status": status, "updated_at": utc_now()})

    # Заменяем изменяемые данные и сохраняем постоянные поля датасета
    manifest["benchmarks"][benchmark_name] = {
        **{
            field: previous_benchmark[field]
            for field in ("dataset", "split", "total_tasks", "output")
            if field in previous_benchmark
        },
        "status": status,
        "updated_at": utc_now(),
        "summary": summary,
    }
    if segments:
        manifest["benchmarks"][benchmark_name]["segments"] = segments

    # Сохраняем счётчик resume после перестроения словаря бенчмарка
    if "resume_count" in previous_benchmark:
        manifest["benchmarks"][benchmark_name]["resume_count"] = previous_benchmark[
            "resume_count"
        ]

    # Пересчитываем общий summary по всем бенчмаркам текущего run_id
    benchmark_summaries = [
        item.get("summary", {}) for item in manifest["benchmarks"].values()
    ]
    manifest["summary"] = {
        "total_tasks": sum(item.get("total_tasks", 0) for item in benchmark_summaries),
        "successful_tasks": sum(
            item.get("successful_tasks", 0) for item in benchmark_summaries
        ),
        "failed_tasks": sum(
            item.get("failed_tasks", 0) for item in benchmark_summaries
        ),
        "remaining_tasks": sum(
            item.get("remaining_tasks", 0) for item in benchmark_summaries
        ),
        "input_tokens": sum(
            item.get("input_tokens", 0) for item in benchmark_summaries
        ),
        "output_tokens": sum(
            item.get("output_tokens", 0) for item in benchmark_summaries
        ),
        "total_tokens": sum(
            item.get("total_tokens", 0) for item in benchmark_summaries
        ),
        "wall_time_seconds": round(
            sum(item.get("wall_time_seconds", 0.0) for item in benchmark_summaries), 3
        ),
    }

    # Фиксируем актуальный статус и атомарно сохраняем manifest
    manifest["status"] = status
    manifest["updated_at"] = utc_now()
    write_json_atomic(path, manifest)


def finalize_run_manifest(path: Path, status: str) -> None:
    """Фиксирует итоговый статус run и рассчитывает общую скорость обработки.
    """
    # Ранняя ошибка могла произойти до создания manifest
    if not path.exists():
        return

    # Обновляем финальный статус всего запуска
    manifest = read_json(path)
    manifest["status"] = status
    manifest["updated_at"] = utc_now()
    summary = manifest.get("summary", {})

    # Общий throughput учитывает обработанные задачи всех бенчмарков
    wall_time = summary.get("wall_time_seconds", 0.0)
    processed_tasks = summary.get("successful_tasks", 0) + summary.get(
        "failed_tasks", 0
    )
    summary["tasks_per_second"] = (
        round(processed_tasks / wall_time, 4) if wall_time else 0.0
    )
    manifest["summary"] = summary
    write_json_atomic(path, manifest)


def is_successful_record(record: dict[str, Any]) -> bool:
    """Определяет завершённый API-вызов, который не нужно повторять при resume.

    Функция проверяет инфраструктурный успех записи, то есть отсутствие API-ошибки, и решение не None
    """
    # Здесь проверяется технический успех вызова, а не правильность ответа
    metadata = record.get("metadata") or {}
    return record.get("solution") is not None and metadata.get("error") is None


def load_resume_records(
    path: Path,
    benchmark_name: str,
    model_name: str,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    """Восстанавливает завершённые задачи и очищает JSONL перед resume.

    Функция проверяет benchmark/model, отбрасывает API-ошибки, незавершённую
    последнюю строку и старые дубликаты, после чего атомарно оставляет по одной
    актуальной записи на task_id. Возвращаемый словарь используется runner для
    пропуска уже обработанных задач.
    """
    # Отсутствующий JSONL означает что сохранённых задач пока нет
    if not path.exists():
        logger.info("resume_file_missing path=%s", path)
        return {}

    # Читаем файл построчно поскольку каждая строка содержит отдельную задачу
    lines = path.read_text(encoding="utf-8").splitlines()
    successful: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # Незавершённую последнюю строку можно безопасно повторить
            if index == len(lines) - 1:
                logger.warning("resume_discarded_incomplete_last_line path=%s", path)
                continue
            raise ValueError(f"Invalid JSONL at {path}:{index + 1}")

        # Запрещаем смешивать в одном JSONL разные бенчмарки и модели
        if record.get("benchmark_name") != benchmark_name:
            raise ValueError(f"Unexpected benchmark in resume file: {path}")
        if record.get("model_name") != model_name:
            raise ValueError(f"Unexpected model in resume file: {path}")

        # Ошибочные вызовы удаляем чтобы runner выполнил их заново
        if not is_successful_record(record):
            continue

        # Для повторяющегося task_id оставляем последнюю успешную запись
        task_id = str(record["task_id"])
        if task_id not in successful:
            order.append(task_id)
        successful[task_id] = record

    # Атомарно заменяем исходный JSONL очищенным набором успешных записей
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for task_id in order:
            stream.write(json.dumps(successful[task_id], ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)
    return successful


def append_jsonl_record(stream: Any, record: dict[str, Any]) -> None:
    """Надёжно добавляет результат задачи в JSONL сразу после её завершения.

    flush и fsync уменьшают риск потери уже полученных ответов при обрыве
    туннеля, остановке процесса или последующем resume.
    """
    # Записываем одну завершённую задачу и сразу синхронизируем файл с диском
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
