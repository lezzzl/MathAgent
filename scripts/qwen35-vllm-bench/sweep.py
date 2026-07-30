#!/usr/bin/env python3
"""Restart vLLM across server configs and run the streaming benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def wait_for_server(
    base_url: str,
    process: subprocess.Popen[Any],
    timeout_s: float,
    progress_interval_s: float,
) -> float:
    server_root = base_url.rstrip("/")
    if server_root.endswith("/v1"):
        server_root = server_root[:-3]
    health_url = f"{server_root}/health"
    started = time.monotonic()
    deadline = time.monotonic() + timeout_s
    next_report = started + progress_interval_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    elapsed = time.monotonic() - started
                    print(f"Server ready after {format_duration(elapsed)}.", flush=True)
                    return elapsed
        except (urllib.error.URLError, TimeoutError):
            pass
        now = time.monotonic()
        if progress_interval_s > 0 and now >= next_report:
            print(
                f"Waiting for server startup... {format_duration(now - started)} elapsed",
                flush=True,
            )
            next_report = now + progress_interval_s
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s")


def stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def write_csv(rows: list[dict[str, Any]], target: Path) -> None:
    if not rows:
        return
    columns = [
        "server_config",
        "concurrency",
        "output_tokens_target",
        "output_throughput_tok_s",
        "total_throughput_tok_s",
        "tpot_ms_p50",
        "ttft_ms_p50",
        "e2e_s_p50",
        "gpu_utilization_pct_mean",
        "gpu_memory_used_mib_max",
        "failed_requests",
        "exact_target_length",
        "result_file",
        "progress_log",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def append_event(target: Path, event: str, **fields: Any) -> None:
    """Append and flush one durable sweep event."""
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with target.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("sweep.example.json"))
    parser.add_argument("--output-tokens", type=int, help="override output_tokens in the JSON")
    parser.add_argument("--config-name", action="append", help="run only this server config; repeatable")
    parser.add_argument("--result-dir", type=Path, default=Path("results/sweep"))
    parser.add_argument("--server-start-timeout-s", type=float, default=1200)
    parser.add_argument(
        "--sweep-log",
        type=Path,
        help="JSONL sweep event log (default: RESULT_DIR/sweep-progress.jsonl)",
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=5.0,
        help="benchmark progress interval; 0 disables periodic updates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_env = {str(k): str(v) for k, v in config.get("base_env", {}).items()}
    base_url = config.get("base_url", "http://127.0.0.1:8000/v1")
    output_tokens = args.output_tokens or int(config.get("output_tokens", 4096))
    concurrencies = [int(value) for value in config["concurrencies"]]
    rows: list[dict[str, Any]] = []
    selected = set(args.config_name or [])
    enabled_configs = [
        item
        for item in config["server_configs"]
        if item.get("enabled", True) and (not selected or item["name"] in selected)
    ]
    total_runs = len(enabled_configs) * len(concurrencies)
    run_index = 0
    completed_durations: list[float] = []
    sweep_log = args.sweep_log or args.result_dir / "sweep-progress.jsonl"
    sweep_log.parent.mkdir(parents=True, exist_ok=True)
    sweep_log.write_text("", encoding="utf-8")

    print(
        f"Sweep plan: {len(enabled_configs)} server configuration(s), "
        f"{len(concurrencies)} concurrency level(s), {total_runs} benchmark run(s).",
        flush=True,
    )
    append_event(
        sweep_log,
        "sweep_started",
        server_configs=len(enabled_configs),
        concurrencies=concurrencies,
        total_runs=total_runs,
        output_tokens=output_tokens,
    )

    for config_index, server_config in enumerate(enabled_configs, start=1):
        name = server_config["name"]
        run_dir = args.result_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(base_env)
        env.update({str(k): str(v) for k, v in server_config.get("env", {}).items()})
        env.setdefault("PYTHONUNBUFFERED", "1")
        server_log_path = run_dir / "server.log"
        print(
            f"\n=== config {config_index}/{len(enabled_configs)}: starting {name} ===",
            flush=True,
        )
        append_event(
            sweep_log,
            "server_config_started",
            server_config=name,
            config_index=config_index,
            config_total=len(enabled_configs),
            server_log=str(server_log_path),
        )
        with server_log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                ["bash", str(root / "serve.sh")],
                cwd=root,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                startup_elapsed = wait_for_server(
                    base_url,
                    process,
                    args.server_start_timeout_s,
                    max(args.progress_interval_s, 10.0),
                )
                append_event(
                    sweep_log,
                    "server_ready",
                    server_config=name,
                    startup_elapsed_s=startup_elapsed,
                )
                for concurrency in concurrencies:
                    run_index += 1
                    run_started = time.monotonic()
                    result_path = run_dir / f"c{concurrency}-o{output_tokens}.json"
                    progress_log_path = run_dir / f"c{concurrency}-o{output_tokens}.progress.jsonl"
                    command = [
                        sys.executable,
                        str(root / "benchmark.py"),
                        "--base-url",
                        base_url,
                        "--model",
                        env.get("SERVED_MODEL_NAME", "qwen3.5-4b"),
                        "--concurrency",
                        str(concurrency),
                        "--num-requests",
                        str(concurrency),
                        "--output-tokens",
                        str(output_tokens),
                        "--result",
                        str(result_path),
                        "--progress-interval-s",
                        str(args.progress_interval_s),
                        "--progress-log",
                        str(progress_log_path),
                    ]
                    print(
                        f"--- run {run_index}/{total_runs}: {name}, "
                        f"concurrency={concurrency}, output={output_tokens} ---",
                        flush=True,
                    )
                    append_event(
                        sweep_log,
                        "benchmark_started",
                        run_index=run_index,
                        total_runs=total_runs,
                        server_config=name,
                        concurrency=concurrency,
                        output_tokens=output_tokens,
                        result_file=str(result_path),
                        progress_log=str(progress_log_path),
                    )
                    completed = subprocess.run(command, cwd=root, env=env, check=False)
                    run_elapsed = time.monotonic() - run_started
                    completed_durations.append(run_elapsed)
                    remaining_runs = total_runs - run_index
                    eta_s = statistics.fmean(completed_durations) * remaining_runs
                    print(
                        f"Run {run_index}/{total_runs} finished in "
                        f"{format_duration(run_elapsed)}; approximate sweep ETA "
                        f"{format_duration(eta_s)}.",
                        flush=True,
                    )
                    if not result_path.exists():
                        rows.append(
                            {
                                "server_config": name,
                                "concurrency": concurrency,
                                "output_tokens_target": output_tokens,
                                "failed_requests": concurrency,
                                "result_file": str(result_path),
                                "progress_log": str(progress_log_path),
                            }
                        )
                        write_csv(rows, args.result_dir / "summary.csv")
                        append_event(
                            sweep_log,
                            "benchmark_finished",
                            run_index=run_index,
                            total_runs=total_runs,
                            server_config=name,
                            concurrency=concurrency,
                            returncode=completed.returncode,
                            elapsed_s=run_elapsed,
                            approximate_sweep_eta_s=eta_s,
                            result_available=False,
                            progress_log=str(progress_log_path),
                        )
                        if completed.returncode:
                            print(f"benchmark failed with code {completed.returncode}", flush=True)
                        continue
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    summary = result["summary"]
                    gpu = summary.get("gpu", {})
                    rows.append(
                        {
                            "server_config": name,
                            "concurrency": concurrency,
                            "output_tokens_target": output_tokens,
                            "output_throughput_tok_s": summary.get("output_throughput_tok_s"),
                            "total_throughput_tok_s": summary.get("total_throughput_tok_s"),
                            "tpot_ms_p50": summary.get("tpot_ms_p50"),
                            "ttft_ms_p50": summary.get("ttft_ms_p50"),
                            "e2e_s_p50": summary.get("e2e_s_p50"),
                            "gpu_utilization_pct_mean": gpu.get("utilization_pct_mean"),
                            "gpu_memory_used_mib_max": gpu.get("memory_used_mib_max"),
                            "failed_requests": summary.get("failed_requests"),
                            "exact_target_length": summary.get("exact_target_length"),
                            "result_file": str(result_path),
                            "progress_log": str(progress_log_path),
                        }
                    )
                    write_csv(rows, args.result_dir / "summary.csv")
                    final_signals: list[str] = []
                    if progress_log_path.exists():
                        progress_lines = progress_log_path.read_text(encoding="utf-8").splitlines()
                        if progress_lines:
                            final_signals = json.loads(progress_lines[-1]).get("signals", [])
                    append_event(
                        sweep_log,
                        "benchmark_finished",
                        run_index=run_index,
                        total_runs=total_runs,
                        server_config=name,
                        concurrency=concurrency,
                        returncode=completed.returncode,
                        elapsed_s=run_elapsed,
                        approximate_sweep_eta_s=eta_s,
                        result_available=True,
                        output_throughput_tok_s=summary.get("output_throughput_tok_s"),
                        gpu_utilization_pct_mean=gpu.get("utilization_pct_mean"),
                        gpu_memory_used_mib_max=gpu.get("memory_used_mib_max"),
                        failed_requests=summary.get("failed_requests"),
                        exact_target_length=summary.get("exact_target_length"),
                        signals=final_signals,
                        progress_log=str(progress_log_path),
                    )
            finally:
                stop_process_group(process)
        print(f"=== stopped {name}; log: {server_log_path} ===", flush=True)
        append_event(sweep_log, "server_config_stopped", server_config=name)

    write_csv(rows, args.result_dir / "summary.csv")
    ranked = sorted(
        (row for row in rows if isinstance(row.get("output_throughput_tok_s"), (int, float))),
        key=lambda row: row["output_throughput_tok_s"],
        reverse=True,
    )
    print("\nTop runs by aggregate output throughput:")
    for row in ranked[:10]:
        print(
            f"{row['server_config']:24s} c={row['concurrency']:>2} "
            f"{row['output_throughput_tok_s']:.1f} tok/s"
        )
    append_event(
        sweep_log,
        "sweep_finished",
        completed_runs=len(rows),
        successful_ranked_runs=len(ranked),
        summary_csv=str(args.result_dir / "summary.csv"),
    )
    print(f"Sweep event log: {sweep_log}", flush=True)


if __name__ == "__main__":
    main()
