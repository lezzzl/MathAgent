"""Общая реализация запуска математических бенчмарков."""

# ruff: noqa: E402

import argparse
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from mathagent.agent.graph import (
    ModelConfig,
    create_code_agent_graph,
    create_solver_graph,
)
from scripts.benchmarks.run_artifacts import (
    append_jsonl_record,
    configure_run_logger,
    ensure_manifest,
    finalize_run_manifest,
    generate_run_id,
    get_benchmark_output_path,
    get_manifest_path,
    is_successful_record,
    load_resume_records,
    update_benchmark_manifest,
    validate_run_id,
)

DEFAULT_SOLVER_PROMPT = ROOT / "conf/base/prompts/solver-v0.yml"
DEFAULT_CODE_AGENT_PROMPT = ROOT / "conf/base/prompts/code-agent-v1.yml"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Хранит параметры, которые различаются у бенчмарков."""

    name: str
    dataset_name: str
    split: str
    task_id_field: str
    output_directory: str
    problem_field: str = "problem"
    ground_truth_field: str = "solution"
    metadata_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskOutcome:
    """Хранит готовую JSONL-запись и категорию ошибки вызова."""

    record: dict[str, Any]
    error_category: str | None


def parse_benchmark_args(
    description: str,
    *,
    include_output: bool = True,
) -> argparse.Namespace:
    """Создаёт общий CLI всех benchmark-скриптов и возвращает выбранные параметры.

    Отдельные бенчмарки используют одинаковые настройки
    модели, sampling, concurrency, логирования и resume.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=os.getenv("MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )

    # Дефолтные параметры взял из MathArena
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    #Ограничивает макимальную длшину ответа модели
    parser.add_argument("--max-tokens", type=int, default=2048)

    # Ограничивает время ожидания ответа модели, чтобы не зависать на одной задаче
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-retries", type=int, default=1)

    parser.add_argument(
        "--pipeline",
        choices=("solver", "code_agent"),
        default="solver",
    )
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--execution-timeout", type=float, default=10.0)
    parser.add_argument("--limit", type=int) # ограничивает число задач из датасета, чтобы быстро проверить работу runner

    parser.add_argument("--concurrency", type=int, default=8) # количество параллельных запросов к модели
    parser.add_argument("--max-consecutive-api-errors", type=int, default=3) # максимальное число подряд идущих ошибок API, после которого runner прерывает выполнение

    # Эти парметры не используются прямо и нужны просто для записи инфорации о моедли
    parser.add_argument("--reasoning-parser", default="qwen3")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-max-model-len", type=int, default=73728)


    if include_output:
        parser.add_argument("--output", type=Path)

    parser.add_argument("--run-id") # задаёт явный идентификатор запуска, чтобы можно было продолжить прерванный запуск с тем же run_id
    parser.add_argument("--resume", action="store_true") # добавляет возможность продолжить прерванный запуск с тем же run_id
    return parser.parse_args()


def build_record(
    run_id: str,
    config: BenchmarkConfig,
    model_name: str,
    task_id: str,
    solution: str | None,
    reasoning: str | dict[str, str] | None,
    ground_truth: str,
    metadata: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собирает унифицированную JSONL-запись для одной решённой задачи.

    Общая схема нужна, чтобы evaluator одинаково читал результаты разных
    бенчмарков и pipeline независимо от их исходного формата.
    """
    record = {
        "run_id": run_id,
        "benchmark_name": config.name,
        "model_name": model_name,
        "task_id": task_id,
        "solution": solution,
        "reasoning": reasoning,
        "ground_truth": ground_truth,
        "metadata": metadata,
    }
    if trace is not None:
        record["trace"] = trace
    return record


def resolve_output_path(
    config: BenchmarkConfig,
    output: Path | None,
    run_id: str,
) -> Path:
    """Выбирает явный --output либо стандартный JSONL внутри каталога run_id."""
    if output is not None:
        return output
    return get_benchmark_output_path(run_id, config.output_directory)

# Нужно для сохранения версии prompt-конфига в manifest, раньше брался просто из state графа
def load_prompt_version(prompt_path: Path) -> str:
    """Извлекает версию prompt-конфига для воспроизводимости результатов.
    """
    with prompt_path.open(encoding="utf-8") as stream:
        prompt_config = yaml.safe_load(stream)
    try:
        return str(prompt_config["version"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Prompt config has no version: {prompt_path}") from exc

# Вынес проверку всех аргументов в отдельную функцию
def validate_args(args: argparse.Namespace) -> None:
    """Отклоняет некорректные CLI-параметры до загрузки датасета и запуска модели."""
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.max_consecutive_api_errors < 1:
        raise ValueError("--max-consecutive-api-errors must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    if args.max_repairs < 0:
        raise ValueError("--max-repairs must be non-negative")
    if args.execution_timeout <= 0:
        raise ValueError("--execution-timeout must be positive")
    if args.vllm_max_num_seqs < 1:
        raise ValueError("--vllm-max-num-seqs must be positive")
    if args.vllm_max_model_len < 1:
        raise ValueError("--vllm-max-model-len must be positive")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires an explicit --run-id")


def create_manifest_config(
    run_id: str,
    args: argparse.Namespace,
    prompt_path: Path,
    prompt_version: str,
) -> dict[str, Any]:
    """Формирует конфигурацию эксперимента, которая будет сохранена в manifest.

    В неё входят параметры, влияющие на результат и производительность, но не
    попадают API-ключ и URL сервера, которые не нужны для сравнения запусков.
    """
    return {
        "run_id": run_id,
        "model": args.model,
        "pipeline": args.pipeline,
        "prompt": str(prompt_path),
        "prompt_version": prompt_version,
        "generation": {
            "thinking": args.thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
        "runtime": {
            "concurrency": args.concurrency,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "max_consecutive_api_errors": args.max_consecutive_api_errors,
        },
        "serving": {
            "engine": "vllm",
            "reasoning_parser": args.reasoning_parser,
            "max_num_seqs": args.vllm_max_num_seqs,
            "max_model_len": args.vllm_max_model_len,
        },
        **(
            {
                "pipeline_config": {
                    "max_repairs": args.max_repairs,
                    "execution_timeout": args.execution_timeout,
                }
            }
            if args.pipeline == "code_agent"
            else {}
        ),
    }


def create_graph(
    args: argparse.Namespace,
    prompt_path: Path,
) -> Any:
    """Собирает выбранный граф с параметрами текущего запуска."""
    model_config = ModelConfig(
        name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    if args.pipeline == "solver":
        return create_solver_graph(model_config, prompt_path)
    if args.pipeline == "code_agent":
        return create_code_agent_graph(
            model_config,
            prompt_path,
            max_repairs=args.max_repairs,
            execution_timeout=args.execution_timeout,
        )
    raise ValueError(f"Unknown pipeline: {args.pipeline}")


def exception_names(exception: Exception) -> set[str]:
    """Собирает типы исключения по всей цепочке cause/context для классификации.

    LangChain и HTTP-клиент могут оборачивать исходную API-ошибку в другие
    исключения, поэтому проверки только внешнего типа недостаточно.
    """
    names: set[str] = set()
    current: BaseException | None = exception
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        names.add(type(current).__name__)
        current = current.__cause__ or current.__context__
    return names

# нужно для классификации ошибок API, чтобы runner мог прерывать выполнение при fatal/infrastructure ошибках
def classify_exception(exception: Exception) -> str:
    """Определяет, должен ли runner остановиться или продолжить после ошибки.

    fatal означает неверную конфигурацию запроса, infrastructure — временную
    проблему API/туннеля, task — ошибку только текущей задачи.
    """
    names = exception_names(exception)
    fatal_names = {
        "AuthenticationError",
        "BadRequestError",
        "NotFoundError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
    }
    infrastructure_names = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutError",
    }
    if names & fatal_names:
        return "fatal"
    if names & infrastructure_names:
        return "infrastructure"
    return "task"


def solve_item(
    graph: Any,
    config: BenchmarkConfig,
    item: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    prompt_version: str,
) -> TaskOutcome:
    """Выполняет граф для одной задачи и превращает результат в JSONL-запись.

    Здесь измеряется latency и сохраняются общие выходные поля любого pipeline.
    Исключения преобразуются в metadata.error, чтобы результат задачи не
    потерялся и мог быть обработан общим циклом параллельного запуска.
    """
    task_id = str(item[config.task_id_field])
    started = time.perf_counter()
    solution: str | None = None
    reasoning: str | dict[str, str] | None = None
    usage: dict[str, Any] = {}
    error: str | None = None
    error_category: str | None = None
    trace: dict[str, Any] | None = None

    try:
        state = graph.invoke(
            {"problem": item[config.problem_field]},
            config={
                "run_name": f"{config.name}:{task_id}",
                "tags": [
                    f"benchmark:{config.name}",
                    f"pipeline:{args.pipeline}",
                    f"model:{args.model}",
                ],
                "metadata": {
                    "thread_id": f"{run_id}:{config.name}",
                    "experiment_run_id": run_id,
                    "benchmark_name": config.name,
                    "task_id": task_id,
                    "model_name": args.model,
                    "pipeline": args.pipeline,
                    "prompt_version": prompt_version,
                },
            },
        )
        solution = state["solution"]
        reasoning = state.get("reasoning")
        usage = state["usage"]
        prompt_version = state["prompt_version"]
        trace = state.get("trace")
    except Exception as exc:
        error_category = classify_exception(exc)
        error = f"{type(exc).__name__}: {exc}"

    metadata = {
        "dataset": config.dataset_name,
        **{field: item.get(field) for field in config.metadata_fields},
        "prompt_version": prompt_version,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "usage": usage,
        "error": error,
    }
    record = build_record(
        run_id=run_id,
        config=config,
        model_name=args.model,
        task_id=task_id,
        solution=solution,
        reasoning=reasoning,
        ground_truth=str(item[config.ground_truth_field]),
        metadata=metadata,
        trace=trace,
    )
    return TaskOutcome(record, error_category)

# Уже нужно будет если несколько узлов
def usage_totals(usage: dict[str, Any]) -> dict[str, int]:
    """Приводит плоский и вложенный token usage к общей сумме.

    Рекурсивный подсчёт поддерживает как один вызов модели, так и будущие графы,
    в которых usage будет разделён по нескольким узлам.
    """
    token_fields = ("input_tokens", "output_tokens", "total_tokens")
    if any(field in usage for field in token_fields):
        return {field: int(usage.get(field, 0) or 0) for field in token_fields}

    totals = {field: 0 for field in token_fields}
    for value in usage.values():
        if not isinstance(value, dict):
            continue
        child_totals = usage_totals(value)
        for field in token_fields:
            totals[field] += child_totals[field]
    return totals


def summarize_records(
    records: list[dict[str, Any]],
    total_tasks: int,
    wall_time: float,
) -> dict[str, Any]:
    """Рассчитывает сводные task, token и throughput-метрики по записям JSONL."""
    successful = sum(is_successful_record(record) for record in records)
    token_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = (record.get("metadata") or {}).get("usage") or {}
        totals = usage_totals(usage)
        for field in token_totals:
            token_totals[field] += totals[field]
    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful,
        "failed_tasks": len(records) - successful,
        "remaining_tasks": max(total_tasks - len(records), 0),
        **token_totals,
        "wall_time_seconds": round(wall_time, 3),
        "tasks_per_second": round(len(records) / wall_time, 4) if wall_time else 0.0,
    }


def run_concurrent_tasks(
    graph: Any,
    config: BenchmarkConfig,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    prompt_version: str,
    output_path: Path,
    existing_records: dict[str, dict[str, Any]],
    logger: Any,
) -> tuple[str, list[dict[str, Any]], float]:
    """Параллельно решает задачи, не превышая заданный --concurrency.

    Функция поддерживает заполненный пул запросов, записывает ответы по мере
    завершения и останавливает подачу новых задач при fatal/infrastructure
    ошибках. Она возвращает статус, все доступные записи и wall time сегмента.
    """
    # При resume исключаем уже сохранённые задачи, но включаем их записи в summary
    pending_items = [
        item
        for item in items
        if str(item[config.task_id_field]) not in existing_records
    ]

    # Счётчики описывают прогресс всего бенчмарка и состояние circuit breaker
    all_records = list(existing_records.values())
    completed = len(existing_records)
    total_tasks = len(items)
    consecutive_infrastructure_errors = 0
    abort_status: str | None = None
    started = time.perf_counter()

    logger.info(
        "benchmark_started benchmark=%s total=%d skipped=%d concurrency=%d output=%s",
        config.name,
        total_tasks,
        completed,
        args.concurrency,
        output_path,
    )

    # JSONL открыт в append-режиме, чтобы результат сохранялся после каждой задачи
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        # Параллельный запуск задач через ограниченный пул worker
        executor = ThreadPoolExecutor(max_workers=args.concurrency)

        # futures связывает выполняющийся запрос с task_id для логов и прогресса
        futures: dict[Future[TaskOutcome], str] = {}
        item_iterator = iter(pending_items)

        def submit_next() -> bool:
            """Занимает освободившийся worker следующей ещё не отправленной задачей."""
            try:
                item = next(item_iterator)
            except StopIteration:
                return False
            task_id = str(item[config.task_id_field])
            future = executor.submit(
                solve_item,
                graph,
                config,
                item,
                args,
                run_id,
                prompt_version,
            )
            futures[future] = task_id
            logger.info("task_submitted benchmark=%s task_id=%s", config.name, task_id)
            return True

        # Первичное заполнение пула запускает не больше --concurrency запросов.
        for _ in range(min(args.concurrency, len(pending_items))):
            submit_next()

        try:
            while futures:
                # Обрабатываем запросы сразу после завершения, не ожидая весь пул.
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                finished_slots = 0
                for future in done:
                    task_id = futures.pop(future)
                    outcome = future.result()
                    record = outcome.record
                    append_jsonl_record(output, record)
                    all_records.append(record)
                    completed += 1
                    finished_slots += 1

                    # Успех сбрасывает серию инфраструктурных ошибок, а fatal
                    # ошибка или достижение лимита останавливает запуск новых задач
                    metadata = record["metadata"]
                    usage = usage_totals(metadata.get("usage") or {})
                    if outcome.error_category is None:
                        consecutive_infrastructure_errors = 0
                        logger.info(
                            "task_completed benchmark=%s task_id=%s progress=%d/%d "
                            "latency=%.3f output_tokens=%d finish_reason=%s",
                            config.name,
                            task_id,
                            completed,
                            total_tasks,
                            metadata["latency_seconds"],
                            usage["output_tokens"],
                            (metadata.get("usage") or {}).get("finish_reason"),
                        )
                    else:
                        logger.error(
                            "task_failed benchmark=%s task_id=%s category=%s error=%s",
                            config.name,
                            task_id,
                            outcome.error_category,
                            metadata["error"],
                        )
                        if outcome.error_category == "fatal":
                            abort_status = "failed"
                        elif outcome.error_category == "infrastructure":
                            consecutive_infrastructure_errors += 1
                            if (
                                consecutive_infrastructure_errors
                                >= args.max_consecutive_api_errors
                            ):
                                abort_status = "interrupted"

                # Освободившиеся места заполняются только пока продолжение безопасно
                if abort_status is None:
                    for _ in range(finished_slots):
                        submit_next()
                else:
                    logger.error(
                        "circuit_breaker_open benchmark=%s status=%s remaining=%d",
                        config.name,
                        abort_status,
                        total_tasks - completed,
                    )
        except KeyboardInterrupt:
            abort_status = "interrupted"
            logger.warning("benchmark_interrupted benchmark=%s", config.name)
        finally:
            # Завершаем активные worker и отменяем задания, которые ещё не начались
            executor.shutdown(wait=True, cancel_futures=True)

    # Итоговый статус различает остановку запуска и ошибки отдельных задач
    wall_time = time.perf_counter() - started
    if abort_status is not None:
        return abort_status, all_records, wall_time
    if any(not is_successful_record(record) for record in all_records):
        return "completed_with_errors", all_records, wall_time
    return "completed", all_records, wall_time


def _run_benchmark(
    config: BenchmarkConfig,
    args: argparse.Namespace,
    run_id: str,
    logger: Any,
) -> int:
    """Выполняет полный жизненный цикл одного подготовленного бенчмарка.

    Функция выбирает prompt, загружает датасет, подготавливает manifest/resume,
    запускает графы, сохраняет summary и возвращает код завершения для run_all.
    """
    from datasets import load_dataset

    # Определяем prompt и JSONL-файл до запуска, чтобы заранее проверить конфигурацию
    default_prompts = {
        "solver": DEFAULT_SOLVER_PROMPT,
        "code_agent": DEFAULT_CODE_AGENT_PROMPT,
    }
    prompt_path = args.prompt or default_prompts[args.pipeline]
    prompt_path = prompt_path.resolve()
    prompt_version = load_prompt_version(prompt_path)
    output_path = resolve_output_path(config, getattr(args, "output", None), run_id)
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --resume with the same --run-id."
        )

    # Загружаем выбранный split датасета и применяем --limit для коротких прогонов
    dataset = load_dataset(config.dataset_name, split=config.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    items = [dict(item) for item in dataset]

    # Создаём новый manifest или проверяем совместимость параметров при resume
    manifest_path = get_manifest_path(run_id)
    manifest_config = create_manifest_config(
        run_id,
        args,
        prompt_path,
        prompt_version,
    )
    manifest = ensure_manifest(
        manifest_path,
        manifest_config,
        config.name,
        {
            "dataset": config.dataset_name,
            "split": config.split,
            "total_tasks": len(items),
            "output": str(output_path),
        },
        resume=args.resume,
    )

    # Накопленное время предыдущих сегментов нужно для корректного resume summary
    previous_wall_time = 0.0
    if args.resume:
        previous_wall_time = float(
            manifest.get("benchmarks", {})
            .get(config.name, {})
            .get("summary", {})
            .get("wall_time_seconds", 0.0)
        )

    # При resume очищаем JSONL и восстанавливаем задачи, которые не надо повторять
    existing_records = (
        load_resume_records(
            output_path,
            config.name,
            args.model,
            logger,
        )
        if args.resume
        else {}
    )
    for task_id in existing_records:
        logger.info("resume_skipped benchmark=%s task_id=%s", config.name, task_id)

    # Собираем solver-граф и передаём задачи диспетчеру параллельных запросов
    graph = create_graph(args, prompt_path)
    status, records, wall_time = run_concurrent_tasks(
        graph=graph,
        config=config,
        items=items,
        args=args,
        run_id=run_id,
        prompt_version=prompt_version,
        output_path=output_path,
        existing_records=existing_records,
        logger=logger,
    )

    # Формируем накопительный summary и отдельные метрики текущего сегмента
    cumulative_wall_time = previous_wall_time + wall_time
    summary = summarize_records(records, len(items), cumulative_wall_time)
    processed_in_segment = len(records) - len(existing_records)
    segment = {
        "resume": args.resume,
        "processed_tasks": processed_in_segment,
        "wall_time_seconds": round(wall_time, 3),
        "tasks_per_second": (
            round(processed_in_segment / wall_time, 4)
            if processed_in_segment and wall_time
            else 0.0
        ),
    }

    # Записываем итог бенчмарка в manifest и закрываем статус всего запуска
    update_benchmark_manifest(
        manifest_path,
        config.name,
        status,
        summary,
        segment,
    )
    finalize_run_manifest(manifest_path, status)
    logger.info(
        "benchmark_finished benchmark=%s status=%s successful=%d failed=%d "
        "remaining=%d wall_time=%.3f output=%s",
        config.name,
        status,
        summary["successful_tasks"],
        summary["failed_tasks"],
        summary["remaining_tasks"],
        summary["wall_time_seconds"],
        output_path,
    )

    # Код возврата позволяет run_all решить, можно ли запускать следующий бенчмарк
    if status in {"failed", "interrupted"}:
        return 2
    if status == "completed_with_errors":
        return 1
    return 0


def run_benchmark(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Публичная точка запуска одного бенчмарка с логированием и обработкой ошибок.

    Отдельные run_aime/run_hmmt/run_imo скрипты передают сюда только описание
    своего датасета и CLI-параметры, не дублируя общую реализацию runner.
    """
    validate_args(args)
    run_id = (
        validate_run_id(args.run_id) if args.run_id else generate_run_id(args.model)
    )
    args.run_id = run_id
    logger = configure_run_logger(run_id)
    manifest_path = get_manifest_path(run_id)
    logger.info(
        "run_started run_id=%s model=%s benchmark=%s", run_id, args.model, config.name
    )
    try:
        return _run_benchmark(config, args, run_id, logger)
    except KeyboardInterrupt:
        logger.warning("run_interrupted run_id=%s benchmark=%s", run_id, config.name)
        finalize_run_manifest(manifest_path, "interrupted")
        return 2
    except Exception:
        logger.exception("run_failed run_id=%s benchmark=%s", run_id, config.name)
        finalize_run_manifest(manifest_path, "failed")
        return 2
