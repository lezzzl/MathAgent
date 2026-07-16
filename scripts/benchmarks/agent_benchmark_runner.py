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
sys.path.insert(0, str(ROOT))

import langgraph_math_solver
from langgraph_math_solver import build_solver_graph

from dotenv import load_dotenv
load_dotenv()

DEFAULT_PROMPT = ROOT / "conf/base/prompts/agent-step-v1.yml"

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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--role", default="solver")
    parser.add_argument("--limit", type=int)

    parser.add_argument("--k-branches", type=int, default=3, help="Количество генерируемых ветвей в режиме recovery")
    parser.add_argument("--score-threshold", type=float, default=0.8, help="Порог оценки для прохождения шага")
    parser.add_argument("--branch-mode", default="single", choices=["single", "multi"], help="Режим генерации шага по умолчанию")
    parser.add_argument("--token-budget", type=int, default=50000, help="Лимит токенов на одну задачу")
    parser.add_argument("--max-recoveries", type=int, default=3, help="Максимальное число recovery-попыток на задачу")

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
    return ROOT / "results" / config.output_directory / f"agent_{timestamp}.jsonl"


def run_benchmark(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Решает задачи выбранного бенчмарка через LangGraph агент и записывает результаты в JSONL."""
    from datasets import load_dataset

    langgraph_math_solver.MODEL_NAME = args.model

    dataset = load_dataset(config.dataset_name, split=config.split)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_path = resolve_output_path(config, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.prompt:
        langgraph_math_solver.load_prompts_from_yaml(args.prompt)
    graph = build_solver_graph()

    errors = 0
    progress_width = len(str(len(dataset)))
    with output_path.open("w", encoding="utf-8") as output:
        for position, item in enumerate(dataset, start=1):
            task_id = str(item[config.task_id_field])
            started = time.perf_counter()
            
            solution: str | None = None
            tokens_used: int = 0
            error: str | None = None
            gave_up: bool = False
            gave_up_reason: str | None = None
            is_valid: bool = False
            verifier_rationale: str | None = None
            steps_taken: list[str] = []

            print(
                f"[{position:0{progress_width}d}/{len(dataset):0{progress_width}d}] "
                f"{config.name} task {task_id}"
            )

            initial_state = {
                "problem": item["problem"],
                "steps": [],
                "candidate_steps": [],
                "candidate_scores": [],
                "k_branches": args.k_branches,
                "score_threshold": args.score_threshold,
                "branch_mode": args.branch_mode,
                "base_temperature": args.temperature,
                "tokens_used": 0,
                "token_budget": args.token_budget,
                "in_recovery": False,
                "recovery_count": 0,
                "max_recoveries": args.max_recoveries,
                "total_recovery_events": 0,
                "final_answer": None,
                "is_valid": False,
                "verifier_rationale": "",
                "gave_up": False,
                "gave_up_reason": "",
            }

            try:
                state = graph.invoke(initial_state)
                solution = state.get("final_answer")
                tokens_used = state.get("tokens_used", 0)
                is_valid = state.get("is_valid", False)
                verifier_rationale = state.get("verifier_rationale")
                gave_up = state.get("gave_up", False)
                gave_up_reason = state.get("gave_up_reason")
                steps_taken = state.get("steps", [])
            except Exception as exc:
                errors += 1
                error = f"{type(exc).__name__}: {exc}"

            metadata = {
                "dataset": config.dataset_name,
                **{field: item.get(field) for field in config.metadata_fields},
                "prompt_version": "langgraph_math_solver_v2",
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "error": error,
                "agent_metrics": {
                    "tokens_used": tokens_used,
                    "is_valid": is_valid,
                    "verifier_rationale": verifier_rationale,
                    "gave_up": gave_up,
                    "gave_up_reason": gave_up_reason,
                    "steps_count": len(steps_taken),
                }
            }
            
            record = build_record(
                config,
                task_id,
                solution,
                str(item.get("answer", "")),
                args.model,
                metadata,
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    print(f"Saved {len(dataset)} records to {output_path}")
    return 1 if errors else 0