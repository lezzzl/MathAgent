"""Run every standalone benchmark script with the same model parameters."""

# ruff: noqa: E402

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.benchmark_runner import parse_benchmark_args
from scripts.benchmarks.run_artifacts import (
    configure_run_logger,
    finalize_run_manifest,
    generate_run_id,
    get_manifest_path,
    validate_run_id,
)

BENCHMARK_SCRIPTS = (
    # ROOT / "scripts/benchmarks/run_aime25.py",
    ROOT / "scripts/benchmarks/run_aime26.py",
    ROOT / "scripts/benchmarks/run_hmmt26.py",
    ROOT / "scripts/benchmarks/run_imo_answerbench.py",
    # ROOT / "scripts/benchmarks/run_math500.py",
)


def parse_args() -> argparse.Namespace:
    """Считывает один набор параметров, который будет передан всем бенчмаркам."""
    return parse_benchmark_args(
        __doc__ or "",
        include_output=False,
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
        "--execution-timeout",
        str(args.execution_timeout),
    ]
    command.append("--thinking" if args.thinking else "--no-thinking")
    if args.prompt is not None:
        command.extend(["--prompt", str(args.prompt)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.resume:
        command.append("--resume")
    return command


def main() -> int:
    """Последовательно запускает список бенчмарков и управляет общим статусом run.

    Следующий датасет начинается только после предыдущего. При ошибке или обрыве
    цикл прекращается, manifest получает соответствующий статус, а код возврата
    позволяет shell/CI отличить успех, частичный результат и interruption.
    """
    args = parse_args()
    args.run_id = (
        validate_run_id(args.run_id) if args.run_id else generate_run_id(args.model)
    )
    logger = configure_run_logger(args.run_id)
    logger.info(
        "run_all_started run_id=%s model=%s benchmarks=%s",
        args.run_id,
        args.model,
        ",".join(script.stem for script in BENCHMARK_SCRIPTS),
    )
    failed: list[str] = []
    interrupted = False
    for script in BENCHMARK_SCRIPTS:
        logger.info("benchmark_process_started script=%s", script.stem)
        result = subprocess.run(build_command(script, args), cwd=ROOT, check=False)
        if result.returncode:
            failed.append(script.stem)
            logger.error(
                "benchmark_process_failed script=%s returncode=%d",
                script.stem,
                result.returncode,
            )
            interrupted = result.returncode == 2
            break

    manifest_path = get_manifest_path(args.run_id)
    status = (
        "interrupted"
        if interrupted
        else "completed_with_errors"
        if failed
        else "completed"
    )
    finalize_run_manifest(manifest_path, status)
    if failed:
        logger.error("run_all_finished status=%s failed=%s", status, ",".join(failed))
        return 2 if interrupted else 1
    logger.info("run_all_finished status=completed run_id=%s", args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
