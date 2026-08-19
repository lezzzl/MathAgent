"""Аналитика прогонов react-агента: тулы, ошибки тулов, токены, ablation-сравнение.

Отвечает на вопросы из брифа:
  • какие тулы как часто использовались     → --tools
  • распределение ошибок тулов               → --tools (раздел «ошибки»)
  • статистика по токенам                     → --tokens (нужен runner.log)
  • правда ли с тулами решается ДРУГОЙ набор  → --compare A B (два *_verified.jsonl)
    задач, чем без тулов? (+ leave-one-out)

Источники данных (всё уже пишется прогоном, доп. инструментовка не нужна):
  • <run>/imo_answerbench.jsonl (или любой бенч) — поле reasoning содержит строки
    `Observation (<tool>):\\n<вывод>`, откуда берём и тул, и признак ошибки;
  • <run>/runner.log — строки task_completed с output_tokens/latency;
  • *_verified.jsonl (от verify_answers.py) — поле is_correct для сравнения наборов.

Примеры:
  python scripts/analyze_runs.py results/runs/ilya-react-imo-v7 --tools --tokens
  python scripts/analyze_runs.py --compare with_tools_verified.jsonl no_tools_verified.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Строка наблюдения в transcript: `Observation (<tool>):\n<тело до следующего [role]>`
_OBS = re.compile(r"Observation \(([^)]+)\):\n(.*?)(?=\n\[|\nObservation \(|\Z)", re.S)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Прочитать jsonl в список словарей (пустые строки пропускаем)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def classify_observation(text: str) -> str:
    """Классифицировать вывод тула в категорию (для распределения ошибок).

    Порядок важен: сначала явные сбои, потом «нет результата», иначе ok."""
    low = text.lower()
    if "traceback (most recent call last)" in low or re.search(r"\b\w*error\b", low):
        return "exception"          # питоновское исключение в run_python
    if "превышен таймаут" in low:
        return "timeout"
    if "пустой вывод" in low:
        return "empty_output"       # забыл print(...)
    if "ошибка разбора" in low or "ошибка:" in low or "не смог вычислить" in low:
        return "parse_error"        # sympy_check/solve/closed_form не разобрали ввод
    if (
        "решений не найдено" in low
        or "не нашлось" in low
        or "контрпример не подобран" in low
        or "не тождество" in low
        or "неверно" in low
    ):
        return "no_result"          # тул отработал, но полезного ответа нет/опровержение
    return "ok"


def analyze_tools(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Частота тулов, доля задач с тулом, распределение шагов и ошибок."""
    calls = Counter()               # вызовов на каждый тул
    tasks_with_tool = Counter()     # в скольких задачах тул звался ≥1 раза
    errors = Counter()              # категория вывода → счётчик (по всем вызовам)
    errors_by_tool: dict[str, Counter] = {}
    steps_hist = Counter()          # tool_steps → число задач
    tasks_any_tool = 0

    for row in rows:
        reasoning = row.get("reasoning") or ""
        obs = _OBS.findall(reasoning)
        seen_here: set[str] = set()
        for tool, body in obs:
            calls[tool] += 1
            seen_here.add(tool)
            cat = classify_observation(body)
            errors[cat] += 1
            errors_by_tool.setdefault(tool, Counter())[cat] += 1
        for tool in seen_here:
            tasks_with_tool[tool] += 1
        if seen_here:
            tasks_any_tool += 1
        steps = (row.get("trace") or {}).get("tool_steps", 0)
        steps_hist[steps] += 1

    return {
        "total_tasks": len(rows),
        "tasks_any_tool": tasks_any_tool,
        "calls": calls,
        "tasks_with_tool": tasks_with_tool,
        "errors": errors,
        "errors_by_tool": errors_by_tool,
        "steps_hist": steps_hist,
    }


def analyze_tokens(log_path: Path) -> dict[str, Any]:
    """Статистика по токенам/латентности из task_completed строк runner.log."""
    out_tokens: list[int] = []
    latencies: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"output_tokens=(\d+)", line)
        if not m:
            continue
        out_tokens.append(int(m.group(1)))
        lat = re.search(r"latency=([\d.]+)", line)
        if lat:
            latencies.append(float(lat.group(1)))

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        return s[min(len(s) - 1, int(len(s) * p))]

    return {
        "tasks": len(out_tokens),
        "out_tokens": {
            "sum": sum(out_tokens),
            "median": pct(out_tokens, 0.5),
            "p90": pct(out_tokens, 0.9),
            "max": max(out_tokens) if out_tokens else 0,
        },
        "latency_s": {
            "median": pct(latencies, 0.5),
            "p90": pct(latencies, 0.9),
            "max": max(latencies) if latencies else 0.0,
        },
    }


def compare_runs(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, Any]:
    """Сравнить наборы решённых задач двух прогонов (*_verified.jsonl с is_correct).

    A — обычно «с тулами», B — «без тулов». Показывает, решается ли РАЗНЫЙ набор."""
    def by_id(rows: list[dict[str, Any]]) -> dict[str, bool]:
        return {str(r["task_id"]): bool(r.get("is_correct")) for r in rows}

    a, b = by_id(rows_a), by_id(rows_b)
    common = a.keys() & b.keys()
    only_a = sorted(t for t in common if a[t] and not b[t])   # решил только A (с тулами)
    only_b = sorted(t for t in common if b[t] and not a[t])   # решил только B (без тулов)
    both = sorted(t for t in common if a[t] and b[t])
    neither = sorted(t for t in common if not a[t] and not b[t])
    return {
        "common": len(common),
        "acc_a": sum(a[t] for t in common),
        "acc_b": sum(b[t] for t in common),
        "only_a": only_a,
        "only_b": only_b,
        "both": len(both),
        "neither": len(neither),
    }


def _find(run_dir: Path, suffix: str) -> Path | None:
    """Найти в каталоге прогона файл по суффиксу (первый по алфавиту)."""
    hits = sorted(run_dir.glob(f"*{suffix}"))
    return hits[0] if hits else None


def _print_tools(stats: dict[str, Any]) -> None:
    total = stats["total_tasks"]
    print(f"\n=== ТУЛЫ (задач: {total}) ===")
    any_t = stats["tasks_any_tool"]
    print(f"использовали тул хотя бы раз: {any_t}/{total} = {any_t / total * 100:.0f}%")
    print("\nчастота (вызовов | задач с тулом):")
    for tool, n in stats["calls"].most_common():
        t = stats["tasks_with_tool"][tool]
        print(f"  {tool:<14} {n:>4} вызовов | {t:>3} задач ({t / total * 100:.0f}%)")
    print("\nраспределение tool_steps по задачам:")
    for steps, n in sorted(stats["steps_hist"].items()):
        print(f"  {steps} шагов: {n} задач")
    print("\nисходы вызовов тулов (ошибки/результаты):")
    total_calls = sum(stats["errors"].values()) or 1
    for cat, n in stats["errors"].most_common():
        print(f"  {cat:<12} {n:>4} ({n / total_calls * 100:.0f}%)")
    if any(cat != "ok" for cat in stats["errors"]):
        print("\nошибки по тулам:")
        for tool, cats in stats["errors_by_tool"].items():
            bad = {c: k for c, k in cats.items() if c != "ok"}
            if bad:
                print(f"  {tool}: " + ", ".join(f"{c}={k}" for c, k in bad.items()))


def _print_tokens(stats: dict[str, Any]) -> None:
    print(f"\n=== ТОКЕНЫ (задач в логе: {stats['tasks']}) ===")
    o = stats["out_tokens"]
    print(f"output_tokens: сумма={o['sum']} медиана={o['median']} p90={o['p90']} max={o['max']}")
    l = stats["latency_s"]
    print(f"latency, сек:  медиана={l['median']:.0f} p90={l['p90']:.0f} max={l['max']:.0f}")


def _print_compare(cmp: dict[str, Any], name_a: str, name_b: str) -> None:
    print(f"\n=== СРАВНЕНИЕ НАБОРОВ (общих задач: {cmp['common']}) ===")
    print(f"  {name_a}: {cmp['acc_a']}/{cmp['common']} верных")
    print(f"  {name_b}: {cmp['acc_b']}/{cmp['common']} верных")
    print(f"  решили оба: {cmp['both']} | не решил никто: {cmp['neither']}")
    print(f"  ТОЛЬКО {name_a} (выигрыш от тулов): {len(cmp['only_a'])} → {cmp['only_a']}")
    print(f"  ТОЛЬКО {name_b} (тулы помешали/шум): {len(cmp['only_b'])} → {cmp['only_b']}")
    net = len(cmp["only_a"]) - len(cmp["only_b"])
    print(f"  чистый эффект тулов: {net:+d} задач; наборы {'РАЗЛИЧАЮТСЯ' if cmp['only_a'] or cmp['only_b'] else 'совпадают'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="Каталог прогона results/runs/<id> или путь к .jsonl")
    p.add_argument("--tools", action="store_true", help="Частота тулов и распределение ошибок")
    p.add_argument("--tokens", action="store_true", help="Статистика токенов (нужен runner.log)")
    p.add_argument("--log", help="Путь к runner.log (если не в каталоге прогона)")
    p.add_argument("--compare", nargs=2, metavar=("WITH_TOOLS", "NO_TOOLS"),
                   help="Два *_verified.jsonl: сравнить наборы решённых задач")
    args = p.parse_args()

    if args.compare:
        a = _iter_jsonl(Path(args.compare[0]))
        b = _iter_jsonl(Path(args.compare[1]))
        _print_compare(compare_runs(a, b), "с тулами", "без тулов")
        return 0

    if not args.run:
        p.error("укажите каталог прогона / .jsonl, либо --compare A B")

    run = Path(args.run)
    jsonl = run if run.suffix == ".jsonl" else _find(run, ".jsonl")
    if not jsonl or not jsonl.exists():
        p.error(f"не нашёл .jsonl в {run}")
    rows = _iter_jsonl(jsonl)
    print(f"прогон: {jsonl}")

    if args.tools or not (args.tokens or args.compare):
        _print_tools(analyze_tools(rows))
    if args.tokens:
        log = Path(args.log) if args.log else (run if run.is_dir() else run.parent) / "runner.log"
        if log.exists():
            _print_tokens(analyze_tokens(log))
        else:
            print(f"\n(нет {log} — статистику токенов пропускаю)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
