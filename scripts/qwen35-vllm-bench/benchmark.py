#!/usr/bin/env python3
"""Streaming OpenAI-compatible benchmark for very long Qwen reasoning runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_PROMPT = """Solve the following problem carefully. Explore alternatives, check your
work, and use the available reasoning budget before giving the final answer.

Problem: Find all positive integers n for which n^2 + 20 is divisible by n + 2.
"""


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    started_s: float
    first_token_s: float | None
    finished_s: float
    prompt_tokens: int
    output_tokens: int
    finish_reason: str | None
    error: str | None = None

    @property
    def e2e_s(self) -> float:
        return self.finished_s - self.started_s

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.started_s

    @property
    def tpot_ms(self) -> float | None:
        if self.first_token_s is None or self.output_tokens <= 1:
            return None
        return 1000 * (self.finished_s - self.first_token_s) / (self.output_tokens - 1)


@dataclass
class ProgressState:
    total_requests: int
    target_output_tokens: int
    started_at: float = field(default_factory=time.perf_counter)
    started_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    token_units: int = 0
    finalized_output_tokens: int = 0
    completed_results: list[RequestResult] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress(snapshot: dict[str, Any]) -> None:
    requests = snapshot["requests"]
    tokens = snapshot["tokens"]
    performance = snapshot["performance"]
    label = snapshot["event"]
    message = (
        f"[{label}] requests {requests['completed']}/{requests['total']} "
        f"(active {requests['active']}, failed {requests['failed']}) | "
        f"output ~{tokens['observed']:,}/{tokens['target']:,} "
        f"({tokens['progress_pct']:5.1f}%) | "
        f"~{performance['output_throughput_tok_s']:,.1f} tok/s | "
        f"elapsed {format_duration(snapshot['elapsed_s'])} | "
        f"ETA {format_duration(snapshot['eta_s'])}"
    )
    if sys.stderr.isatty() and label != "final":
        print(f"\r{message}", end="", file=sys.stderr, flush=True)
    else:
        prefix = "\r" if sys.stderr.isatty() else ""
        print(f"{prefix}{message}", file=sys.stderr, flush=True)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def progress_snapshot(
    state: ProgressState,
    gpu_samples: list[dict[str, float]],
    *,
    final: bool,
    previous_elapsed_s: float | None = None,
    previous_tokens: int | None = None,
) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - state.started_at, 1e-9)
    target = state.total_requests * state.target_output_tokens
    observed = max(state.token_units, 0)
    throughput = observed / elapsed
    interval_throughput = None
    if previous_elapsed_s is not None and previous_tokens is not None:
        interval_elapsed = elapsed - previous_elapsed_s
        if interval_elapsed > 0:
            interval_throughput = max(0, observed - previous_tokens) / interval_elapsed
    eta = (target - observed) / throughput if throughput > 0 and observed < target else None
    if observed >= target:
        eta = 0.0

    successful = [result for result in state.completed_results if result.ok]
    ttfts_ms = [1000 * value for result in successful if (value := result.ttft_s) is not None]
    tpots_ms = [value for result in successful if (value := result.tpot_ms) is not None]
    e2es_s = [result.e2e_s for result in successful]
    latency = {
        "ttft_ms_p50": percentile(ttfts_ms, 50),
        "ttft_ms_p95": percentile(ttfts_ms, 95),
        "tpot_ms_p50": percentile(tpots_ms, 50),
        "tpot_ms_p95": percentile(tpots_ms, 95),
        "e2e_s_p50": percentile(e2es_s, 50),
        "e2e_s_p95": percentile(e2es_s, 95),
    }

    gpu: dict[str, float | int] = {"sample_count": len(gpu_samples)}
    if gpu_samples:
        latest = gpu_samples[-1]
        gpu.update(
            {
                "utilization_pct_latest": latest["utilization_pct"],
                "utilization_pct_mean": statistics.fmean(
                    sample["utilization_pct"] for sample in gpu_samples
                ),
                "memory_used_mib_latest": latest["memory_used_mib"],
                "memory_used_mib_max": max(sample["memory_used_mib"] for sample in gpu_samples),
                "memory_total_mib": latest["memory_total_mib"],
                "power_w_latest": latest["power_w"],
                "power_w_mean": statistics.fmean(sample["power_w"] for sample in gpu_samples),
            }
        )

    signals: list[str] = []
    active = state.started_requests - state.completed_requests
    if state.failed_requests:
        signals.append("request_failures")
    if len(gpu_samples) >= 3:
        mean_util = float(gpu["utilization_pct_mean"])
        if mean_util >= 90:
            signals.append("gpu_compute_saturated")
        elif mean_util < 60 and active > 0:
            signals.append("gpu_underutilized")
        memory_fraction = float(gpu["memory_used_mib_max"]) / float(gpu["memory_total_mib"])
        if memory_fraction >= 0.95:
            signals.append("gpu_memory_near_capacity")
    if (
        interval_throughput is not None
        and elapsed >= 30
        and throughput > 0
        and interval_throughput < throughput * 0.8
        and active > 0
    ):
        signals.append("recent_throughput_below_run_average")
    if (
        latency["tpot_ms_p50"] is not None
        and latency["tpot_ms_p95"] is not None
        and latency["tpot_ms_p95"] > latency["tpot_ms_p50"] * 1.5
    ):
        signals.append("high_tpot_tail")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "final" if final else "progress",
        "elapsed_s": elapsed,
        "eta_s": eta,
        "requests": {
            "total": state.total_requests,
            "started": state.started_requests,
            "active": active,
            "completed": state.completed_requests,
            "successful": len(successful),
            "failed": state.failed_requests,
        },
        "tokens": {
            "target": target,
            "observed": observed,
            "finalized_exact": state.finalized_output_tokens,
            "progress_pct": min(100.0, 100.0 * observed / target) if target else 100.0,
        },
        "performance": {
            "output_throughput_tok_s": throughput,
            "interval_output_throughput_tok_s": interval_throughput,
            **latency,
        },
        "gpu": gpu,
        "signals": signals,
    }


async def report_progress(
    state: ProgressState,
    interval_s: float,
    gpu_samples: list[dict[str, float]],
    progress_log: Path | None,
) -> None:
    handle = None
    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        handle = progress_log.open("w", encoding="utf-8", buffering=1)
    previous_elapsed: float | None = None
    previous_tokens: int | None = None
    try:
        while not state.done.is_set():
            if interval_s <= 0:
                await state.done.wait()
                break
            try:
                await asyncio.wait_for(state.done.wait(), timeout=interval_s)
            except TimeoutError:
                snapshot = progress_snapshot(
                    state,
                    gpu_samples,
                    final=False,
                    previous_elapsed_s=previous_elapsed,
                    previous_tokens=previous_tokens,
                )
                render_progress(snapshot)
                if handle is not None:
                    handle.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
                    handle.flush()
                previous_elapsed = snapshot["elapsed_s"]
                previous_tokens = snapshot["tokens"]["observed"]
        snapshot = progress_snapshot(
            state,
            gpu_samples,
            final=True,
            previous_elapsed_s=previous_elapsed,
            previous_tokens=previous_tokens,
        )
        render_progress(snapshot)
        if handle is not None:
            handle.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
            handle.flush()
    finally:
        if handle is not None:
            handle.close()


async def sample_gpu(
    stop: asyncio.Event, every_s: float, samples: list[dict[str, float]]
) -> list[dict[str, float]]:
    if every_s <= 0:
        return samples
    origin = time.perf_counter()
    query = "utilization.gpu,memory.used,memory.total,power.draw"
    while not stop.is_set():
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                first_gpu = stdout.decode().strip().splitlines()[0]
                util, used, total, power = [float(x.strip()) for x in first_gpu.split(",")]
                samples.append(
                    {
                        "elapsed_s": time.perf_counter() - origin,
                        "utilization_pct": util,
                        "memory_used_mib": used,
                        "memory_total_mib": total,
                        "power_w": power,
                    }
                )
        except (FileNotFoundError, IndexError, ValueError):
            return samples
        try:
            await asyncio.wait_for(stop.wait(), timeout=every_s)
        except TimeoutError:
            pass
    return samples


def request_payload(args: argparse.Namespace, request_id: int, max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "seed": args.seed + request_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "top_k": args.top_k,
        "min_p": args.min_p,
        "ignore_eos": args.ignore_eos,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    return payload


async def one_request(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    request_id: int,
    max_tokens: int,
    progress: ProgressState | None = None,
) -> RequestResult:
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    start = time.perf_counter()
    first_token: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    stream_chunks = 0
    if progress is not None:
        progress.started_requests += 1
    try:
        async with session.post(url, json=request_payload(args, request_id, max_tokens)) as response:
            if response.status != 200:
                body = (await response.text())[:2000]
                raise RuntimeError(f"HTTP {response.status}: {body}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if text:
                    stream_chunks += 1
                    if progress is not None:
                        # With vLLM's default stream interval this is normally one
                        # token per non-empty delta. Final usage replaces this
                        # approximation when the request completes.
                        progress.token_units += 1
                    if first_token is None:
                        first_token = time.perf_counter()
        end = time.perf_counter()
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        if progress is not None:
            progress.completed_requests += 1
            progress.finalized_output_tokens += output_tokens
            if output_tokens > 0:
                progress.token_units += output_tokens - stream_chunks
        result = RequestResult(
            request_id=request_id,
            ok=True,
            started_s=start,
            first_token_s=first_token,
            finished_s=end,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )
        if progress is not None:
            progress.completed_results.append(result)
        return result
    except Exception as exc:  # keep the other benchmark requests running
        if progress is not None:
            progress.completed_requests += 1
            progress.failed_requests += 1
        result = RequestResult(
            request_id=request_id,
            ok=False,
            started_s=start,
            first_token_s=first_token,
            finished_s=time.perf_counter(),
            prompt_tokens=0,
            output_tokens=0,
            finish_reason=finish_reason,
            error=f"{type(exc).__name__}: {exc}",
        )
        if progress is not None:
            progress.completed_results.append(result)
        return result


async def run_group(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    count: int,
    max_tokens: int,
    id_offset: int,
    progress: ProgressState | None = None,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(args.concurrency)

    async def limited(i: int) -> RequestResult:
        async with semaphore:
            return await one_request(session, args, i, max_tokens, progress)

    return await asyncio.gather(*(limited(id_offset + i) for i in range(count)))


def summarize(
    results: list[RequestResult], gpu_samples: list[dict[str, float]], args: argparse.Namespace
) -> dict[str, Any]:
    successful = [r for r in results if r.ok]
    if not successful:
        return {"successful_requests": 0, "failed_requests": len(results)}
    wall_s = max(r.finished_s for r in successful) - min(r.started_s for r in successful)
    output_tokens = sum(r.output_tokens for r in successful)
    prompt_tokens = sum(r.prompt_tokens for r in successful)
    ttfts = [r.ttft_s for r in successful if r.ttft_s is not None]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms is not None]
    e2es = [r.e2e_s for r in successful]

    gpu_summary: dict[str, float] = {}
    if gpu_samples:
        for key in ("utilization_pct", "memory_used_mib", "power_w"):
            values = [sample[key] for sample in gpu_samples]
            gpu_summary[f"{key}_mean"] = statistics.fmean(values)
            gpu_summary[f"{key}_max"] = max(values)

    return {
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "wall_time_s": wall_s,
        "request_throughput_rps": len(successful) / wall_s,
        "output_throughput_tok_s": output_tokens / wall_s,
        "total_throughput_tok_s": (prompt_tokens + output_tokens) / wall_s,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tokens_per_successful_request_mean": output_tokens / len(successful),
        "ttft_ms_p50": 1000 * percentile(ttfts, 50) if ttfts else None,
        "ttft_ms_p95": 1000 * percentile(ttfts, 95) if ttfts else None,
        "tpot_ms_p50": percentile(tpots, 50),
        "tpot_ms_p95": percentile(tpots, 95),
        "e2e_s_p50": percentile(e2es, 50),
        "e2e_s_p95": percentile(e2es, 95),
        "exact_target_length": all(r.output_tokens == args.output_tokens for r in successful),
        "gpu": gpu_summary,
    }


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=None, connect=args.connect_timeout_s, sock_read=None)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency + 4, 16))
    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
        if args.warmup_requests:
            print(
                f"Warming up with {args.warmup_requests} request(s) × "
                f"{args.warmup_tokens} tokens...",
                file=sys.stderr,
                flush=True,
            )
            warmups = await run_group(session, args, args.warmup_requests, args.warmup_tokens, -10000)
            failures = [r.error for r in warmups if not r.ok]
            if failures:
                raise RuntimeError(f"Warmup failed: {failures[0]}")
            print("Warmup complete. Starting measured run.", file=sys.stderr, flush=True)

        stop_gpu = asyncio.Event()
        gpu_samples: list[dict[str, float]] = []
        gpu_task = asyncio.create_task(sample_gpu(stop_gpu, args.sample_gpu_every_s, gpu_samples))
        progress = ProgressState(args.num_requests, args.output_tokens)
        progress_task = asyncio.create_task(
            report_progress(
                progress,
                args.progress_interval_s,
                gpu_samples,
                args.progress_log,
            )
        )
        try:
            results = await run_group(
                session, args, args.num_requests, args.output_tokens, 0, progress
            )
        finally:
            progress.done.set()
            await progress_task
            stop_gpu.set()
            gpu_samples = await gpu_task

    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "num_requests": args.num_requests,
            "output_tokens": args.output_tokens,
            "ignore_eos": args.ignore_eos,
            "enable_thinking": args.enable_thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "progress_log": str(args.progress_log) if args.progress_log else None,
        },
        "summary": summarize(results, gpu_samples, args),
        "requests": [
            {
                **asdict(r),
                "e2e_s": r.e2e_s,
                "ttft_s": r.ttft_s,
                "tpot_ms": r.tpot_ms,
            }
            for r in results
        ],
        "gpu_samples": gpu_samples,
    }
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "qwen3.5-4b"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--output-tokens", type=int, default=65536)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--sample-gpu-every-s", type=float, default=2.0)
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=5.0,
        help="seconds between progress updates; 0 disables them",
    )
    parser.add_argument("--connect-timeout-s", type=float, default=60.0)
    parser.add_argument("--result", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument(
        "--progress-log",
        type=Path,
        help="JSONL progress log (default: RESULT stem plus .progress.jsonl)",
    )
    parser.add_argument(
        "--disable-progress-log",
        action="store_true",
        help="do not write the partial JSONL performance log",
    )
    args = parser.parse_args()
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    if args.num_requests is None:
        args.num_requests = args.concurrency
    if args.concurrency < 1 or args.num_requests < 1 or args.output_tokens < 1:
        parser.error("concurrency, num-requests, and output-tokens must be positive")
    if args.progress_interval_s < 0:
        parser.error("progress-interval-s must be non-negative")
    if args.disable_progress_log:
        args.progress_log = None
    elif args.progress_log is None:
        args.progress_log = args.result.with_name(f"{args.result.stem}.progress.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if args.progress_log:
        print(f"Writing partial performance snapshots to {args.progress_log}", file=sys.stderr)
    document = asyncio.run(async_main(args))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document["summary"], indent=2))
    print(f"Saved detailed result to {args.result}")
    if document["summary"].get("failed_requests"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
