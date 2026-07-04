"""Run every standalone benchmark script with the same model parameters."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.benchmark_runner import parse_benchmark_args

BENCHMARK_SCRIPTS = (
    ROOT / "scripts/benchmarks/run_aime24.py",
    ROOT / "scripts/benchmarks/run_aime25.py",
    ROOT / "scripts/benchmarks/run_math500.py",
)


def parse_args() -> argparse.Namespace:
    """Считывает общие параметры для запуска всех бенчмарков."""
    return parse_benchmark_args(
        __doc__ or "",
        default_model="qwen3.5:4b",
        include_output=False,
    )


def build_command(script: Path, args: argparse.Namespace) -> list[str]:
    """Формирует команду отдельного benchmark с общими параметрами."""
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
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--prompt",
        str(args.prompt),
        "--role",
        args.role,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def main() -> int:
    """Последовательно запускает AIME24 и MATH500 и возвращает общий статус."""
    args = parse_args()
    failed: list[str] = []
    for script in BENCHMARK_SCRIPTS:
        print(f"\n=== Running {script.stem} ===", flush=True)
        result = subprocess.run(build_command(script, args), cwd=ROOT, check=False)
        if result.returncode:
            failed.append(script.stem)

    if failed:
        print(f"\nFailed benchmarks: {', '.join(failed)}")
        return 1
    print("\nAll benchmarks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
