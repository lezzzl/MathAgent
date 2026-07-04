"""Общая реализация запуска математических бенчмарков"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mathagent.agent.graph import ModelConfig, create_solver_graph

DEFAULT_PROMPT = ROOT / "conf/base/prompts/solver-v0.yml"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Хранит параметры, которые различаются у бенчмарков."""

    name: str
    dataset_name: str
    split: str
    task_id_field: str
    output_directory: str
    metadata_fields: tuple[str, ...] = ()


def parse_benchmark_args(
    description: str,
    default_model: str,
    *,
    include_output: bool = True,
) -> argparse.Namespace:
    """Считывает общие параметры модели и запуска из командной строки."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=os.getenv("MODEL", default_model))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "ollama"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--role", default="solver")
    parser.add_argument("--limit", type=int)
    if include_output:
        parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_record(
    config: BenchmarkConfig,
    task_id: str,
    solution: str | None,
    ground_truth: str,
    model_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Формирует одну запись итогового JSONL-файла."""
    return {
        "benchmark_name": config.name,
        "task_id": task_id,
        "solution": solution,
        "ground_truth": ground_truth,
        "model_name": model_name,
        "metadata": metadata,
    }


def resolve_output_path(config: BenchmarkConfig, output: Path | None) -> Path:
    """Выбирает переданный путь или генерирует имя JSONL по времени запуска."""
    if output is not None:
        return output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "results" / config.output_directory / f"{timestamp}.jsonl"


def run_benchmark(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Решает задачи выбранного бенчмарка и записывает результаты в JSONL."""
    from datasets import load_dataset

    dataset = load_dataset(config.dataset_name, split=config.split)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_path = resolve_output_path(config, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = create_solver_graph(
        ModelConfig(
            name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        ),
        prompt_path=args.prompt,
        role_name=args.role,
    )

    errors = 0
    progress_width = len(str(len(dataset)))
    with output_path.open("w", encoding="utf-8") as output:
        for position, item in enumerate(dataset, start=1):
            task_id = str(item[config.task_id_field])
            started = time.perf_counter()
            solution: str | None = None
            usage: dict[str, Any] = {}
            error: str | None = None
            prompt_version: str | None = None
            print(
                f"[{position:0{progress_width}d}/{len(dataset):0{progress_width}d}] "
                f"{config.name} task {task_id}"
            )

            try:
                state = graph.invoke({"problem": item["problem"]})
                solution = state["solution"]
                usage = state["usage"]
                prompt_version = state["prompt_version"]
            except Exception as exc:
                errors += 1
                error = f"{type(exc).__name__}: {exc}"

            metadata = {
                "dataset": config.dataset_name,
                **{field: item.get(field) for field in config.metadata_fields},
                "prompt_version": prompt_version,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "usage": usage,
                "error": error,
            }
            record = build_record(
                config,
                task_id,
                solution,
                str(item["solution"]),
                args.model,
                metadata,
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    print(f"Saved {len(dataset)} records to {output_path}")
    return 1 if errors else 0
