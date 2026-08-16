"""Run every standalone benchmark script with the same model parameters."""

# ruff: noqa: E402

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.pavel.benchmark_runner import parse_benchmark_args
from scripts.benchmarks.pavel.run_artifacts import (
    configure_run_logger,
    finalize_run_manifest,
    generate_run_id,
    get_manifest_path,
    read_json,
    validate_run_id,
)

BENCHMARK_SCRIPTS = (
    # ROOT / "scripts/benchmarks/pavel/run_aime25.py",
    ROOT / "scripts/benchmarks/pavel/run_aime26.py",
    ROOT / "scripts/benchmarks/pavel/run_hmmt26.py",
    ROOT / "scripts/benchmarks/pavel/run_imo_answerbench.py",
    # ROOT / "scripts/benchmarks/pavel/run_math500.py",
)
BENCHMARKS_BY_NAME = {
    script.stem.removeprefix("run_"): script for script in BENCHMARK_SCRIPTS
}


def parse_args() -> argparse.Namespace:
    """Считывает один набор параметров, который будет передан всем бенчмаркам."""
    return parse_benchmark_args(
        __doc__ or "",
        include_output=False,
        benchmark_choices=tuple(BENCHMARKS_BY_NAME),
    )


def build_command(script: Path, args: argparse.Namespace) -> list[str]:
    """Формирует команду benchmark-скрипта без потери параметров общего запуска.

    Отдельный процесс изолирует сбой конкретного бенчмарка, а одинаковый run_id
    объединяет JSONL, manifest и runner.log в один эксперимент.
    """
    command = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--min-p",
        str(args.min_p),
        "--presence-penalty",
        str(args.presence_penalty),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
        "--concurrency",
        str(args.concurrency),
        "--max-consecutive-api-errors",
        str(args.max_consecutive_api_errors),
        "--run-id",
        args.run_id,
        "--reasoning-parser",
        args.reasoning_parser,
        "--vllm-max-num-seqs",
        str(args.vllm_max_num_seqs),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--pipeline",
        args.pipeline,
        "--max-repairs",
        str(args.max_repairs),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--max-format-retries",
        str(args.max_format_retries),
        "--max-tool-repairs",
        str(args.max_tool_repairs),
        "--max-precheck-rejections",
        str(args.max_precheck_rejections),
        "--max-verification-rounds",
        str(args.max_verification_rounds),
        "--execution-timeout",
        str(args.execution_timeout),
    ]
    command.append("--thinking" if args.thinking else "--no-thinking")
    if args.planner_model is not None:
        command.extend(["--planner-model", args.planner_model])
    if args.planner_base_url is not None:
        command.extend(["--planner-base-url", args.planner_base_url])
    if args.planner_api_key is not None:
        command.extend(["--planner-api-key", args.planner_api_key])
    if args.prompt is not None:
        command.extend(["--prompt", str(args.prompt)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.resume:
        command.append("--resume")
    return command


def read_completed_process_status(manifest_path: Path) -> str | None:
    """Читает статус, записанный завершившимся benchmark-процессом."""
    if not manifest_path.exists():
        return None
    status = read_json(manifest_path).get("status")
    return status if isinstance(status, str) else None


def main() -> int:
    """Последовательно запускает список бенчмарков и управляет общим статусом run.

    Task-level ошибки дают completed_with_errors и не прерывают следующие
    датасеты. Только fatal/infrastructure остановка завершает цикл с кодом 2.
    """
    args = parse_args()
    args.run_id = (
        validate_run_id(args.run_id) if args.run_id else generate_run_id(args.model)
    )
    selected_scripts = tuple(BENCHMARKS_BY_NAME[name] for name in args.benchmarks)
    logger = configure_run_logger(args.run_id)
    logger.info(
        "run_all_started run_id=%s model=%s benchmarks=%s",
        args.run_id,
        args.model,
        ",".join(script.stem for script in selected_scripts),
    )
    failed: list[str] = []
    abort_status: str | None = None
    completed_with_errors = False
    manifest_path = get_manifest_path(args.run_id)
    for script in selected_scripts:
        logger.info("benchmark_process_started script=%s", script.stem)
        result = subprocess.run(build_command(script, args), cwd=ROOT, check=False)
        process_status = read_completed_process_status(manifest_path)

        # Код 1 поддерживается для совместимости со старыми benchmark scripts:
        # это частичные task errors, после которых следующий датасет безопасен
        if result.returncode == 1 or process_status == "completed_with_errors":
            completed_with_errors = True
            logger.warning(
                "benchmark_process_completed_with_errors script=%s",
                script.stem,
            )

        if result.returncode not in {0, 1}:
            failed.append(script.stem)
            logger.error(
                "benchmark_process_failed script=%s returncode=%d",
                script.stem,
                result.returncode,
            )
            abort_status = (
                process_status
                if process_status in {"failed", "interrupted"}
                else "failed"
            )
            break

    status = (
        abort_status
        if abort_status is not None
        else "completed_with_errors"
        if completed_with_errors
        else "completed"
    )
    finalize_run_manifest(manifest_path, status)
    if failed:
        logger.error("run_all_finished status=%s failed=%s", status, ",".join(failed))
        return 2
    if completed_with_errors:
        logger.warning(
            "run_all_finished status=completed_with_errors run_id=%s",
            args.run_id,
        )
        return 0
    logger.info("run_all_finished status=completed run_id=%s", args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
