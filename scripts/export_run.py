"""Готовит папку прогона в общем формате results/runs/<run-id>/ для сравнения.

    python scripts/export_run.py --run-id nikita-qwen35-4b-all-v1 \
        AIME24=results/aime24/agent_..._trajectory.json \
        AIME25=results/aime25/agent_..._trajectory.json

Для каждого бенчмарка нужен файл траектории (--trajectory у раннера); рядом
автоматически ищется <тот же префикс>_verified.jsonl или _imoverified.jsonl.

На выходе — <bench>.jsonl в формате коллег (run_id / benchmark_name /
model_name / task_id / solution / reasoning / ground_truth / metadata) и
manifest.json с теми же полями, что у baseline-* и ilya-* прогонов.

Пошаговый пайплайн отличается от их «solver» тем, что решение собирается из
принятых шагов, а не приходит одним куском, поэтому:
  * solution  — принятые шаги по порядку плюс финальный ответ в \\boxed{};
  * reasoning — сырые генерации по шагам (аналог их <think>-трассы).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from answer_utils import extract_answer

# Как называются файлы в папке прогона (у коллег это aime26.jsonl и т.п.).
FILE_NAMES = {
    "AIME24": "aime24", "AIME25": "aime25", "AIME26": "aime26",
    "HMMT25": "hmmt25", "HMMT26": "hmmt26",
    "HMMT_Feb2025": "hmmt25", "HMMT_Feb2026": "hmmt26",
    "IMOAnswerBench": "imo_answerbench", "IMOAnswer": "imo_answerbench",
    "MATH500": "math500", "GSM8K": "gsm8k",
}


def _txt(rec: Optional[dict], key: str = "content") -> str:
    return ((rec or {}).get(key) or {}).get("text", "") or ""


def find_verified(trajectory: Path) -> Optional[Path]:
    """Ищет рядом файл со сверенными ответами."""
    stem = trajectory.stem.replace("_trajectory", "")
    for suffix in ("_verified.jsonl", "_imoverified.jsonl"):
        candidate = trajectory.with_name(stem + suffix)
        if candidate.exists():
            return candidate
    return None


def build_solution(task: dict) -> str:
    """Собирает читаемое решение из принятых шагов.

    У коллег в `solution` лежит цельный текст решения, из которого сайт
    сравнения достаёт \\boxed{}. Пошаговый пайплайн хранит шаги по отдельности,
    поэтому склеиваем их и в конце обязательно повторяем ответ в \\boxed{}.
    """
    steps = [r for r in task.get("records", []) if r.get("stage") == "commit"]
    steps.sort(key=lambda r: (r.get("depth") if r.get("depth") is not None else 0))
    parts = [f"Step {i}: {_txt(r).strip()}" for i, r in enumerate(steps, 1) if _txt(r).strip()]
    answer = str(task.get("final_answer") or "").strip()
    if answer:
        # Если ответ уже стоит в \boxed внутри последнего шага, второй раз не дублируем.
        tail = parts[-1] if parts else ""
        if not extract_answer(tail):
            parts.append(f"Final answer: \\boxed{{{answer}}}")
    return "\n\n".join(parts)


def build_reasoning(task: dict) -> str:
    """Сырые генерации по шагам — аналог <think>-трассы у коллег."""
    chunks: List[str] = []
    for r in task.get("records", []):
        if r.get("stage") != "generate":
            continue
        body = _txt(r).strip()
        if not body:
            continue
        head = f"[depth {r.get('depth')} branch {r.get('branch')}]"
        chunks.append(f"{head}\n{body}")
    return "\n\n".join(chunks)


def task_usage(task: dict) -> Dict[str, int]:
    tin = tout = ttot = 0
    for r in task.get("records", []):
        tk = r.get("tokens") or {}
        tin += tk.get("input", 0) or 0
        tout += tk.get("output", 0) or 0
        ttot += tk.get("total", 0) or 0
    return {"input_tokens": tin, "output_tokens": tout, "total_tokens": ttot}


def infer_temperature(tasks: List[dict]) -> Optional[float]:
    """Базовая температура генератора.

    В прогонах до добавления поля run.temperature её нет в шапке, но она есть в
    каждой записи вызова; берём минимальную (ветки идут с надбавкой +0.15·i).
    """
    temps = [r.get("temperature") for t in tasks for r in t.get("records", [])
             if r.get("stage") == "generate" and r.get("temperature") is not None]
    return min(temps) if temps else None


def infer_generator_num_predict(tasks: List[dict]) -> Optional[int]:
    """Потолок генерации у генератора — из записей вызовов.

    В шапке прогона его нет, а различать прогоны по нему нужно: лимит роли
    поднимается через ROLE_NUM_PREDICT, не меняя версию промпта, и без этого
    поля два таких прогона выглядят в manifest.json одинаково.
    """
    caps = {r.get("num_predict") for t in tasks for r in t.get("records", [])
            if r.get("stage") == "generate" and r.get("num_predict")}
    return max(caps) if caps else None


def export_benchmark(bench: str, trajectory: Path, run_id: str, out_dir: Path
                     ) -> Dict[str, Any]:
    data = json.loads(trajectory.read_text(encoding="utf-8"))
    run = data.get("run", {}) or {}
    tasks = data.get("tasks", []) or []
    if run.get("temperature") is None:
        run["temperature"] = infer_temperature(tasks)
    if run.get("generator_num_predict") is None:
        run["generator_num_predict"] = infer_generator_num_predict(tasks)

    verified_path = find_verified(trajectory)
    verified: Dict[str, dict] = {}
    if verified_path:
        for line in verified_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            verified[str(rec.get("task_id"))] = rec

    name = FILE_NAMES.get(bench, bench.lower())
    out_path = out_dir / f"{name}.jsonl"
    model = run.get("model", "")
    n_ok = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    wall = 0.0

    with out_path.open("w", encoding="utf-8") as handle:
        for task in sorted(tasks, key=lambda t: str(t.get("task_id"))):
            tid = str(task.get("task_id"))
            vrec = verified.get(tid, {})
            usage = task_usage(task)
            for k in totals:
                totals[k] += usage[k]
            wall += task.get("elapsed") or 0.0
            metrics = task.get("metrics") or {}
            if vrec.get("is_correct"):
                n_ok += 1

            record = {
                "run_id": run_id,
                "benchmark_name": bench,
                "model_name": model,
                "task_id": tid,
                "solution": build_solution(task),
                "reasoning": build_reasoning(task),
                "ground_truth": str(task.get("ground_truth") or vrec.get("ground_truth") or ""),
                "metadata": {
                    "dataset": run.get("dataset", ""),
                    "prompt_version": Path(str(run.get("prompt", ""))).stem,
                    "temperature": run.get("temperature"),
                    "top_p": None, "top_k": None, "min_p": None,
                    "presence_penalty": None, "repetition_penalty": None,
                    "seed": None,
                    "thinking": run.get("thinking") == "on",
                    "max_tokens": run.get("token_budget"),
                    "latency_seconds": round(task.get("elapsed") or 0.0, 3),
                    "usage": {
                        **usage,
                        "input_token_details": {},
                        "output_token_details": {},
                        "finish_reason": None,
                    },
                    "error": (task.get("error") or None),
                    # Специфика пошагового пайплайна: у «solver»-прогонов коллег
                    # этих полей нет, но лишние ключи в metadata уже встречаются
                    # (у IMO там Category/Source), так что формат не ломается.
                    "agent": {
                        "pipeline": run.get("pipeline"),
                        "branch_mode": run.get("branch_mode"),
                        "k_branches": run.get("k_branches"),
                        "use_tools": run.get("use_tools"),
                        "steps_count": metrics.get("steps_count"),
                        "answer_depth": metrics.get("answer_depth"),
                        "tool_calls": metrics.get("tool_calls"),
                        "segmenter_calls": metrics.get("segmenter_calls"),
                        "gave_up": metrics.get("gave_up"),
                        "gave_up_reason": metrics.get("gave_up_reason"),
                        "is_valid": metrics.get("is_valid"),
                    },
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(tasks)
    print(f"  {bench:16s} -> {out_path.name:22s} задач={n:4d} верно={n_ok:4d} "
          f"({100*n_ok/max(n,1):5.1f}%) токенов={totals['total_tokens']:,}")
    return {
        "dataset": run.get("dataset", ""),
        "split": run.get("split", "train"),
        "total_tasks": n,
        "output": str((out_dir / f"{name}.jsonl").as_posix()),
        "status": "completed",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_tasks": n,
            "successful_tasks": sum(1 for t in tasks if str(t.get("final_answer") or "").strip()),
            "failed_tasks": sum(1 for t in tasks if not str(t.get("final_answer") or "").strip()),
            "remaining_tasks": 0,
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "total_tokens": totals["total_tokens"],
            "wall_time_seconds": round(wall, 3),
            "tasks_per_second": round(n / wall, 4) if wall else 0.0,
        },
        "_correct": n_ok,
        "_run": run,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", metavar="BENCH=trajectory.json",
                    help="пары «имя бенчмарка = путь к файлу траектории»")
    ap.add_argument("--run-id", required=True, help="имя папки прогона")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs",
                    help="куда класть папку (по умолчанию results/runs)")
    ap.add_argument("--status", default="completed", choices=["completed", "partial"])
    args = ap.parse_args()

    out_dir = args.out / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Папка прогона: {out_dir}")

    benchmarks: Dict[str, Any] = {}
    run_meta: Dict[str, Any] = {}
    for spec in args.specs:
        if "=" not in spec:
            print(f"Неверный формат: {spec!r}, нужно BENCH=путь")
            return 2
        bench, path = spec.split("=", 1)
        trajectory = Path(path)
        if not trajectory.exists():
            print(f"Не найден файл траектории: {trajectory}")
            return 2
        info = export_benchmark(bench, trajectory, args.run_id, out_dir)
        run_meta = info.pop("_run") or run_meta
        info.pop("_correct", None)
        benchmarks[bench] = info

    agg = {"total_tasks": 0, "successful_tasks": 0, "failed_tasks": 0,
           "remaining_tasks": 0, "input_tokens": 0, "output_tokens": 0,
           "total_tokens": 0, "wall_time_seconds": 0.0}
    for info in benchmarks.values():
        s = info["summary"]
        for k in agg:
            if k in s:
                agg[k] += s[k]
    agg["wall_time_seconds"] = round(agg["wall_time_seconds"], 3)
    agg["tasks_per_second"] = (round(agg["total_tasks"] / agg["wall_time_seconds"], 4)
                               if agg["wall_time_seconds"] else 0.0)

    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "model": run_meta.get("model", ""),
        "pipeline": run_meta.get("pipeline", ""),
        "prompt": run_meta.get("prompt", ""),
        "prompt_version": Path(str(run_meta.get("prompt", ""))).stem,
        "role": "solver",
        "generation": {
            "thinking": run_meta.get("thinking") == "on",
            "temperature": run_meta.get("temperature"),
            "top_p": None, "top_k": None, "min_p": None,
            "presence_penalty": None, "repetition_penalty": None, "seed": None,
            "max_tokens": run_meta.get("token_budget"),
        },
        "runtime": {
            "concurrency": run_meta.get("workers"),
            "timeout": run_meta.get("timeout"),
            "max_retries": 1,
        },
        "serving": {"engine": "vllm", "reasoning_parser": "qwen3"},
        "agent": {
            "branch_mode": run_meta.get("branch_mode"),
            "k_branches": run_meta.get("k_branches"),
            "score_threshold": run_meta.get("score_threshold"),
            "use_tools": run_meta.get("use_tools"),
            "token_budget": run_meta.get("token_budget"),
            # Потолок одной генерации у роли generator. Поднимается через
            # ROLE_NUM_PREDICT без смены версии промпта, поэтому без этого поля
            # два таких прогона неразличимы в манифесте.
            "generator_num_predict": run_meta.get("generator_num_predict"),
        },
        "status": args.status,
        "created_at": run_meta.get("started_utc", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
        "summary": agg,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest.json записан. Итого задач: {agg['total_tasks']}, "
          f"токенов: {agg['total_tokens']:,}")
    print(f"\nЗагрузка на сервер:\n"
          f"  scp -r {out_dir} <сервер>:/mnt/storage-1/MathAgent/results/runs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
