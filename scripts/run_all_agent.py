import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.agent_benchmark_runner import parse_benchmark_args

BENCHMARK_SCRIPTS = (
    ROOT / "scripts/benchmarks/run_aime24_agent.py",
    ROOT / "scripts/benchmarks/run_aime25_agent.py",
    ROOT / "scripts/benchmarks/run_math500_agent.py",
)


def parse_args() -> argparse.Namespace:
    return parse_benchmark_args(
        __doc__ or "",
        default_model="llama3.2",
        include_output=False,
    )


def build_command(script: Path, args: argparse.Namespace) -> list[str]:
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
        "--k-branches",
        str(args.k_branches),
        "--score-threshold",
        str(args.score_threshold),
        "--branch-mode",
        args.branch_mode,
        "--token-budget",
        str(args.token_budget),
        "--max-recoveries",
        str(args.max_recoveries),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def main() -> int:
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
    print("\nAll agent benchmarks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())