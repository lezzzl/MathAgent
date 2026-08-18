"""Готовит файл для ручной разметки шагов: выборка, ослепление, форма для меток.

    python scripts/make_labeling.py
    python scripts/make_labeling.py --per-cell 25 --depths 0 1 --out labeling/

Зачем. Мы знаем по счёту, что отключение оценщика на глубине 0 ничего не стоит,
а на глубине 1 стоит задач (EXPERIMENTS.md §2б), и по вердиктам — что на глубине
0 он неотличим от монетки (§4а). Оба вывода косвенные: метка там — исход всей
задачи. Ручная разметка отвечает прямо: сколько ХОРОШИХ шагов оценщик зарубил и
сколько ПЛОХИХ пропустил, отдельно на каждой глубине.

Выборка стратифицированная: глубина x вердикт, поровну в каждой ячейке. Иначе
редкие ячейки (принятые шаги на глубине 0) утонут и сравнивать будет нечего.

ОСЛЕПЛЕНИЕ. В файл для разметки не попадают ни балл оценщика, ни его
обоснование; порядок перемешан. Ключ пишется отдельным файлом, который до конца
разметки открывать нельзя.

Чего скрыть НЕЛЬЗЯ: глубину. На глубине 0 принятых шагов нет, на глубине 1 —
ровно один, и разметчик видит это по определению — префикс ему нужен, чтобы
судить о шаге. Так что ослеплён только вердикт; это главное, но не всё.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def text_of(value: Any) -> str:
    """Поля траектории хранятся как {'text': ..., 'clipped': ...} либо строкой."""
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return str(value or "")


def collect(trajectory: Path) -> List[Dict[str, Any]]:
    """Все вердикты оценщика вместе с условием задачи и принятым префиксом."""
    data = json.loads(trajectory.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []
    for task in data.get("tasks", []):
        records = task.get("records") or []
        # Принятые шаги — это стадия commit. Префикс для вердикта на глубине d —
        # всё, что закоммичено на глубинах меньше d.
        committed = {int(r.get("depth") or 0): text_of(r.get("content"))
                     for r in records if r.get("stage") == "commit"}
        for rec in records:
            if rec.get("stage") != "evaluate_result":
                continue
            step = text_of(rec.get("step_text")).strip()
            if len(step) < 40:
                continue          # пустышки и обрывки размечать нечего
            depth = int(rec.get("depth") or 0)
            items.append({
                "task_id": str(task.get("task_id")),
                "problem": (task.get("problem") or "").strip(),
                "prefix": [committed[d] for d in sorted(committed) if d < depth],
                "step": step,
                "depth": depth,
                "accepted": float(rec.get("score") or 0.0) >= 0.5,
                "rationale": text_of(rec.get("content")).strip(),
            })
    return items


def sample(items: List[Dict[str, Any]], depths: List[int], per_cell: int,
           seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for depth in depths:
        for accepted in (False, True):
            cell = [i for i in items if i["depth"] == depth and i["accepted"] == accepted]
            rng.shuffle(cell)
            take = cell[:per_cell]
            verdict = "принят" if accepted else "отвергнут"
            print(f"  глубина {depth}, {verdict:<9}: доступно {len(cell):>3}, "
                  f"берём {len(take)}")
            out.extend(take)
    rng.shuffle(out)      # порядок не должен выдавать ячейку
    return out


HEADER = """# Разметка шагов: сколько хороших зарубил оценщик и сколько плохих пропустил

Заполните строку `МЕТКА:` в каждом блоке одним словом:

* `good`    — шаг математически верен И продвигает решение;
* `bad`     — есть ошибка ЛИБО шаг не продвигает (топтание, повтор, пустая фраза);
* `unclear` — не удаётся решить, не разобрав задачу целиком; или верно, но
              непонятно, продвигает ли.

`unclear` ставьте без стеснения. Эта метка нужна, чтобы сомнительное не
расползлось по двум другим наугад — при подсчёте она выносится отдельно.

Вердикт оценщика скрыт намеренно: он в отдельном файле-ключе, и до конца
разметки его лучше не открывать.

Судите шаг ровно по тем же двум вопросам, что и оценщик: **верно ли** и
**продвигает ли** — относительно условия и уже принятых шагов, а не относительно
полного решения задачи. Шаг не обязан доводить до ответа.

Всего блоков: {n}. Ориентировочно 2–3 минуты на блок.

---
"""

BLOCK = """
## Шаг {idx:03d}

**Задача {task}.**

{problem}

**Принятые ранее шаги:** {prefix}

**Кандидат:**

{step}

МЕТКА:

---
"""


def render(items: List[Dict[str, Any]]) -> str:
    parts = [HEADER.format(n=len(items))]
    for idx, item in enumerate(items, 1):
        if item["prefix"]:
            prefix = "\n\n" + "\n\n".join(
                f"{n}. {s}" for n, s in enumerate(item["prefix"], 1))
        else:
            prefix = "*нет, это первый шаг*"
        parts.append(BLOCK.format(idx=idx, task=item["task_id"],
                                  problem=item["problem"], prefix=prefix,
                                  step=item["step"]))
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", type=Path,
                    default=ROOT / "results/aime26/agent_20260806T011449Z_trajectory.json")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "labeling")
    args = ap.parse_args()

    items = collect(args.trajectory)
    print(f"Вердиктов в траектории: {len(items)}")
    picked = sample(items, args.depths, args.per_cell, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    form = args.out / "steps_to_label.md"
    key = args.out / "steps_key.json"
    form.write_text(render(picked), encoding="utf-8")
    key.write_text(json.dumps(
        [{"idx": i, "task_id": it["task_id"], "depth": it["depth"],
          "accepted": it["accepted"], "rationale": it["rationale"],
          "step": it["step"]} for i, it in enumerate(picked, 1)],
        ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nФорма для разметки: {form}  ({len(picked)} блоков)")
    print(f"Ключ (не открывать до конца): {key}")
    print(f"\nКогда заполните — посчитать: python scripts/score_labeling.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
