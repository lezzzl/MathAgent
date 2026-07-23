"""Конвертер результатов в командный формат results/runs/<run_id>/ (как на GitHub).

Берёт мой layout  results/<model>/<bench>/<timestamp>.jsonl  и приводит к формату
ветки pavel/code_agent:
    results/runs/<run_id>/<bench>.jsonl   — по одному файлу на бенчмарк
    results/runs/<run_id>/manifest.json   — сводка запуска

Ключевые преобразования записи:
  * solution  -> только финальный ответ (то, что после </think>)
  * reasoning -> размышления (то, что до </think>); при обрыве думанья solution=""
  * ground_truth -> чистый эталон (берём из моего поля gold)
  * добавляется run_id; sampler-параметры, которых я не логировал, ставятся в null

Использование:
    python scripts/to_run_format.py results/qwen35-9b --run-id baseline-qwen35-9b-all-v1
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# датасет/сплит для manifest (в записях сплит не сохранялся) — из scripts/benchmarks/run_*.py
BENCH_META = {
    "AIME24":    ("HuggingFaceH4/aime_2024", "train"),
    "AIME25":    ("Sunny8781/AIME2025_w_solution", "test"),
    "AIME26":    ("MathArena/aime_2026", "train"),
    "HMMT25":    ("MathArena/hmmt_feb_2025", "train"),
    "IMOAnswer": ("OpenEvals/IMO-AnswerBench", "train"),
    "MATH500":   ("HuggingFaceH4/MATH-500", "test"),
}

# sampler-поля из референсного формата, которые я не логировал (пишем null для совпадения схемы)
NULL_SAMPLER = {
    "top_p": None, "top_k": None, "min_p": None,
    "presence_penalty": None, "repetition_penalty": None, "seed": None,
}


def split_reasoning(solution: str) -> tuple[str, str]:
    r"""Вернуть (reasoning, final). Делим по </think>: до — размышления, после — ответ.

    Если </think> нет — думанье оборвано на max_tokens: reasoning = весь текст, final = "".
    """
    tag = "</think>"
    idx = solution.find(tag)
    if idx == -1:
        return solution.strip(), ""
    return solution[:idx].strip(), solution[idx + len(tag):].strip()


def parse_run_ts(name: str) -> datetime | None:
    """20260722T085441Z -> aware datetime (UTC). None, если не распарсилось."""
    m = re.match(r"(\d{8}T\d{6}Z)", name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def convert_file(src: Path, run_id: str, out: Path) -> dict[str, Any]:
    """Переписать один <bench>.jsonl в новый формат, вернуть summary для manifest."""
    total = ok = failed = 0
    tin = tout = ttot = 0
    bench_name = ""
    dataset = ""
    prompt_version = None
    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            bench_name = r.get("benchmark_name", "")
            meta = dict(r.get("metadata") or {})
            dataset = meta.get("dataset", "")
            prompt_version = meta.get("prompt_version")
            usage = dict(meta.get("usage") or {})
            usage.setdefault("finish_reason", None)
            tin += usage.get("input_tokens") or 0
            tout += usage.get("output_tokens") or 0
            ttot += usage.get("total_tokens") or 0
            if meta.get("error"):
                failed += 1
            else:
                ok += 1

            reasoning, final = split_reasoning(str(r.get("solution") or ""))
            new_meta = {
                "dataset": dataset,
                "prompt_version": prompt_version,
                "temperature": meta.get("temperature"),
                **NULL_SAMPLER,
                "thinking": True,
                "max_tokens": meta.get("max_tokens"),
                "latency_seconds": meta.get("latency_seconds"),
                "usage": usage,
                "error": meta.get("error"),
            }
            record = {
                "run_id": run_id,
                "benchmark_name": bench_name,
                "model_name": r.get("model_name"),
                "task_id": r.get("task_id"),
                "solution": final,
                "reasoning": reasoning,
                "ground_truth": r.get("gold"),  # чистый эталон
                "metadata": new_meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    ds_ref, split = BENCH_META.get(bench_name, (dataset, ""))
    created = parse_run_ts(src.name)
    updated = datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc)
    wall = (updated - created).total_seconds() if created else 0.0
    return {
        "benchmark_name": bench_name,
        "dataset": ds_ref or dataset,
        "split": split,
        "output": str(out),
        "created_at": created.isoformat() if created else None,
        "updated_at": updated.isoformat(),
        "prompt_version": prompt_version,
        "summary": {
            "total_tasks": total,
            "successful_tasks": ok,
            "failed_tasks": failed,
            "remaining_tasks": 0,
            "input_tokens": tin,
            "output_tokens": tout,
            "total_tokens": ttot,
            "wall_time_seconds": round(wall, 3),
            "tasks_per_second": round(total / wall, 4) if wall > 0 else 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Конвертер в results/runs/<run_id>/ формат")
    parser.add_argument("model_dir", type=Path, help="напр. results/qwen35-9b")
    parser.add_argument("--run-id", required=True, help="напр. baseline-qwen35-9b-all-v1")
    parser.add_argument("--model", default=None, help="имя модели для manifest (по умолч. из записей)")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/runs")
    args = parser.parse_args()

    src_files = sorted(
        f for f in args.model_dir.rglob("*.jsonl")
        if not f.name.endswith(("_verified.jsonl", ".graded.jsonl"))
    )
    if not src_files:
        print(f"Не найдено исходных .jsonl в {args.model_dir}")
        return 1

    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks: dict[str, Any] = {}
    tot = ok = failed = 0
    tin = tout = ttot = 0
    wall_max = 0.0
    created_min: str | None = None
    updated_max: str | None = None
    model_name = args.model
    prompt_version = None

    for src in src_files:
        out_file = out_dir / (src.parent.name + ".jsonl")  # aime24/<ts>.jsonl -> aime24.jsonl
        info = convert_file(src, args.run_id, out_file)
        s = info["summary"]
        benchmarks[info["benchmark_name"]] = {
            "dataset": info["dataset"],
            "split": info["split"],
            "total_tasks": s["total_tasks"],
            "output": info["output"],
            "status": "completed" if s["failed_tasks"] == 0 else "partial",
            "updated_at": info["updated_at"],
            "summary": s,
        }
        tot += s["total_tasks"]; ok += s["successful_tasks"]; failed += s["failed_tasks"]
        tin += s["input_tokens"]; tout += s["output_tokens"]; ttot += s["total_tokens"]
        wall_max = max(wall_max, s["wall_time_seconds"])
        prompt_version = prompt_version or info["prompt_version"]
        if info["created_at"]:
            created_min = min(created_min, info["created_at"]) if created_min else info["created_at"]
        updated_max = max(updated_max, info["updated_at"]) if updated_max else info["updated_at"]
        if model_name is None:
            # достаём model_name из первой записи готового файла
            with out_file.open(encoding="utf-8") as fh:
                model_name = json.loads(fh.readline()).get("model_name")
        print(f"{info['benchmark_name']:<12} {s['total_tasks']:>4} tasks  ->  {out_file}")

    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "model": model_name,
        "pipeline": "solver",
        "prompt": str(ROOT / "conf/base/prompts/solver-reasoning.yml"),
        "prompt_version": prompt_version,
        "role": "solver",
        "generation": {
            "thinking": True,
            "temperature": 0.6,
            **NULL_SAMPLER,
            "max_tokens": 49152,
        },
        "runtime": {"concurrency": 32, "timeout": 3000.0, "max_retries": 1},
        "serving": {"engine": "vllm", "reasoning_parser": "qwen3"},
        "status": "completed" if failed == 0 else "partial",
        "created_at": created_min,
        "updated_at": updated_max,
        "benchmarks": benchmarks,
        "summary": {
            "total_tasks": tot,
            "successful_tasks": ok,
            "failed_tasks": failed,
            "remaining_tasks": 0,
            "input_tokens": tin,
            "output_tokens": tout,
            "total_tokens": ttot,
            "wall_time_seconds": wall_max,
            "tasks_per_second": round(tot / wall_max, 4) if wall_max > 0 else 0.0,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {manifest_path}")
    print(f"ИТОГО: {tot} задач, ошибок {failed}, out_tokens {tout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
