"""Узлы пайплайна benchmarks.

Kedro здесь — оркестратор: узел читает гиперпараметры из conf/ (модель, sampling,
runtime, список бенчей) и запускает каждый бенчмарк отдельным подпроцессом с ОДНИМ
общим run_id — так же, как проверенный scripts/run_all_benchmarks.py. Изоляция
процессов защищает от падения одного бенча, а общий run_id объединяет JSONL,
manifest и runner.log в один эксперимент под results/runs/<run_id>/.

Сам раннер (scripts/benchmarks/*) не дублируется и не переписывается — мы только
транслируем конфиг в его CLI.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# nodes.py: src/mathagent/pipelines/benchmarks/nodes.py -> корень репозитория
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ключ бенчмарка (из params:benchmarks.select) -> его standalone-скрипт
BENCHMARK_SCRIPTS: dict[str, Path] = {
    "aime24": ROOT / "scripts/benchmarks/run_aime24.py",
    "aime25": ROOT / "scripts/benchmarks/run_aime25.py",
    "aime26": ROOT / "scripts/benchmarks/run_aime26.py",
    "hmmt25": ROOT / "scripts/benchmarks/run_hmmt25.py",
    "hmmt26": ROOT / "scripts/benchmarks/run_hmmt26.py",
    "imo_answerbench": ROOT / "scripts/benchmarks/run_imo_answerbench.py",
    "math500": ROOT / "scripts/benchmarks/run_math500.py",
}


def _build_command(
    script: Path, params: dict[str, Any], api_key: str, run_id: str
) -> list[str]:
    """Транслирует конфиг benchmarks в CLI одного benchmark-скрипта."""
    model = params["model"]
    gen = params["generation"]
    runtime = params["runtime"]
    serving = params["serving"]
    command = [
        sys.executable,
        str(script),
        "--model", str(model["name"]),
        "--base-url", str(model["base_url"]),
        "--api-key", api_key,
        "--temperature", str(gen["temperature"]),
        "--top-p", str(gen["top_p"]),
        "--top-k", str(gen["top_k"]),
        "--min-p", str(gen["min_p"]),
        "--presence-penalty", str(gen["presence_penalty"]),
        "--repetition-penalty", str(gen["repetition_penalty"]),
        "--seed", str(gen["seed"]),
        "--max-tokens", str(gen["max_tokens"]),
        "--timeout", str(runtime["timeout"]),
        "--max-retries", str(runtime["max_retries"]),
        "--concurrency", str(runtime["concurrency"]),
        "--max-consecutive-api-errors", str(runtime["max_consecutive_api_errors"]),
        "--run-id", run_id,
        "--reasoning-parser", str(serving["reasoning_parser"]),
        "--vllm-max-num-seqs", str(serving["vllm_max_num_seqs"]),
        "--vllm-max-model-len", str(serving["vllm_max_model_len"]),
        "--pipeline", str(params.get("pipeline", "solver")),
    ]
    command.append("--thinking" if gen["thinking"] else "--no-thinking")
    # prompt хранится относительно корня репозитория; подпроцесс работает с cwd=ROOT
    if params.get("prompt"):
        command += ["--prompt", str(params["prompt"])]
    if params.get("limit") is not None:
        command += ["--limit", str(params["limit"])]
    if params.get("resume"):
        command.append("--resume")
    return command


def run_benchmarks(params: dict[str, Any]) -> dict[str, Any]:
    """Гоняет выбранные бенчмарки под одним run_id и возвращает сводку запуска.

    params — секция benchmarks из conf/. Возвращаемый словарь Kedro сохраняет в
    датасет benchmark_run_summary (см. catalog.yml).
    """
    from scripts.benchmarks.run_artifacts import (
        configure_run_logger,
        finalize_run_manifest,
        generate_run_id,
        get_manifest_path,
        validate_run_id,
    )

    select = list(params["select"])
    unknown = [key for key in select if key not in BENCHMARK_SCRIPTS]
    if unknown:
        raise ValueError(
            f"Неизвестные бенчмарки в benchmarks.select: {unknown}. "
            f"Доступны: {sorted(BENCHMARK_SCRIPTS)}"
        )

    # Ключ API берём по ИМЕНИ env-переменной (в конфиге хранится имя, не сам ключ)
    api_key_env = params["model"].get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env, "EMPTY")

    run_id = (
        validate_run_id(params["run_id"])
        if params.get("run_id")
        else generate_run_id(params["model"]["name"])
    )
    logger = configure_run_logger(run_id)
    logger.info(
        "kedro_run_all_started run_id=%s model=%s benchmarks=%s",
        run_id,
        params["model"]["name"],
        ",".join(select),
    )

    statuses: dict[str, int] = {}
    failed: list[str] = []
    interrupted = False
    for key in select:
        command = _build_command(BENCHMARK_SCRIPTS[key], params, api_key, run_id)
        logger.info("benchmark_process_started benchmark=%s", key)
        result = subprocess.run(command, cwd=str(ROOT), check=False)
        statuses[key] = result.returncode
        if result.returncode:
            failed.append(key)
            interrupted = result.returncode == 2
            logger.error(
                "benchmark_process_failed benchmark=%s returncode=%d",
                key,
                result.returncode,
            )
            break  # следующий бенч не стартуем, как в run_all_benchmarks

    manifest_path = get_manifest_path(run_id)
    status = (
        "interrupted"
        if interrupted
        else "completed_with_errors"
        if failed
        else "completed"
    )
    finalize_run_manifest(manifest_path, status)
    logger.info("kedro_run_all_finished run_id=%s status=%s", run_id, status)

    return {
        "run_id": run_id,
        "status": status,
        "model": params["model"]["name"],
        "benchmarks": statuses,
        "failed": failed,
        "manifest": str(manifest_path),
    }


def publish_to_langsmith(
    summary: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Опционально публикует готовый прогон в LangSmith (инструмент из src/langsmith).

    Живой прогресс инференса и так в logs/<run_id>/runner.log; здесь — пост-фактум
    заливка решений/reasoning/оценки в LangSmith UI. По умолчанию выключено; сбой
    публикации НЕ роняет прогон бенчей (результаты уже сохранены на диск).
    """
    observability = params.get("observability") or {}
    if not observability.get("publish_langsmith"):
        return {"published": False, "reason": "disabled"}

    api_key_env = observability.get("langsmith_api_key_env", "LANGSMITH_API_KEY")
    if not os.getenv(api_key_env):
        return {
            "published": False,
            "reason": f"env-переменная {api_key_env} с ключом LangSmith не задана",
        }

    run_id = summary["run_id"]
    command = [
        sys.executable,
        str(ROOT / "src/langsmith/compare_runs.py"),
        "publish",
        "--run-id",
        run_id,
    ]
    result = subprocess.run(command, cwd=str(ROOT), check=False)
    return {
        "published": result.returncode == 0,
        "run_id": run_id,
        "returncode": result.returncode,
    }
