"""Отчёт по трём полосам вместо одного числа «N из 30».

    python scripts/report_bands.py --baseline results/aime24/agent_A results/aime24/agent_B \
        --runs results/aime24/agent_NEW

Зачем. Разбор всех чистых прогонов показал, что задачи делятся на три группы:
ядро решается в каждом прогоне, безнадёжные — ни в одном, а треть плавает от
прогона к прогону. Единственное число «26/30» смешивает всё это и тонет в
разбросе ±9: у двух конфигураций может быть одинаковый счёт при совершенно
разном поведении.

Скрипт делит задачи на полосы по набору БАЗОВЫХ прогонов и показывает для
каждого проверяемого прогона три отдельные величины:

  * ядро           — сколько из «решаемых всегда» удержано (регрессия?);
  * плавающие      — сколько из «нестабильных» взято (здесь и лежит выигрыш);
  * безнадёжные    — сколько из «не решённых ни разу» задето (прорыв?).

Полосы считаются по базовым прогонам и НЕ включают проверяемый, иначе он влиял
бы на собственную разметку.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def load_verified(prefix: str) -> Dict[str, bool]:
    """Читает <prefix>_verified.jsonl (или _imoverified) -> {task_id: верно?}."""
    base = Path(prefix)
    if base.suffix == ".jsonl":
        candidates = [base]
    else:
        candidates = [base.with_name(base.name + s)
                      for s in ("_verified.jsonl", "_imoverified.jsonl")]
    for path in candidates:
        if path.exists():
            out = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[str(rec.get("task_id"))] = bool(rec.get("is_correct"))
            return out
    raise SystemExit(f"не найден файл сверки для {prefix}")


def bands(baselines: List[Dict[str, bool]]) -> Dict[str, str]:
    """Размечает задачи по базовым прогонам."""
    ids = set(baselines[0])
    for b in baselines[1:]:
        ids &= set(b)
    out = {}
    for t in ids:
        k = sum(b[t] for b in baselines)
        out[t] = "ядро" if k == len(baselines) else ("никогда" if k == 0 else "плавает")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="префиксы базовых прогонов (2+), по ним размечаются полосы")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="префиксы проверяемых прогонов")
    ap.add_argument("--show-tasks", action="store_true",
                    help="печатать id задач, взятых и потерянных относительно полос")
    args = ap.parse_args()

    if len(args.baseline) < 2:
        print("⚠️  Полосы по одному прогону не определить: «плавает» неотличимо "
              "от «ядра». Нужно минимум два базовых прогона.")
        return 2

    base = [load_verified(p) for p in args.baseline]
    band = bands(base)
    sizes = {b: sum(1 for v in band.values() if v == b)
             for b in ("ядро", "плавает", "никогда")}
    print(f"Полосы размечены по {len(base)} базовым прогонам, задач: {len(band)}")
    print(f"  ядро {sizes['ядро']}   плавает {sizes['плавает']}   "
          f"никогда {sizes['никогда']}\n")

    print(f"{'прогон':<34} {'ядро':>12} {'плавающие':>14} {'безнадёжные':>14} {'всего':>8}")
    print("-" * 86)
    for prefix in args.runs:
        got = load_verified(prefix)
        hit = {b: sum(1 for t, v in band.items() if v == b and got.get(t)) for b in sizes}
        total = sum(1 for t in band if got.get(t))
        name = Path(prefix).name[:33]
        print(f"{name:<34} "
              f"{hit['ядро']:>5}/{sizes['ядро']:<6} "
              f"{hit['плавает']:>6}/{sizes['плавает']:<7} "
              f"{hit['никогда']:>7}/{sizes['никогда']:<6} "
              f"{total:>4}/{len(band)}")
        if args.show_tasks:
            lost = sorted((t for t, v in band.items() if v == "ядро" and not got.get(t)),
                          key=lambda x: int(x) if x.isdigit() else 0)
            won = sorted((t for t, v in band.items() if v == "никогда" and got.get(t)),
                         key=lambda x: int(x) if x.isdigit() else 0)
            if lost:
                print(f"    ⚠️  регрессия в ядре: {lost}")
            if won:
                print(f"    ★ взято из безнадёжных: {won}")

    print("\nЧитать так: падение в «ядре» — регрессия, её видно сразу и она важнее "
          "прироста.\nПрирост в «плавающих» — то, ради чего вводится голосование."
          "\nЛюбое ненулевое число в «безнадёжных» — единственный настоящий прорыв.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
