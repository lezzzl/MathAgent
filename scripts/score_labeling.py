"""Считает метрики оценщика по заполненной вручную разметке.

    python scripts/score_labeling.py
    python scripts/score_labeling.py --form labeling/steps_to_label.md --key labeling/steps_key.json

Соединяет метки человека (good / bad / unclear) с вердиктами оценщика из ключа и
считает отдельно по каждой глубине:

  precision   — из принятых оценщиком шагов сколько действительно хороши;
  recall      — из хороших шагов сколько он принял;
  specificity — из плохих шагов сколько он отверг;
  точный тест Фишера — связан ли вердикт с меткой ВНУТРИ глубины.

Главное здесь — не абсолютные числа, а сравнение глубин. Замер по счёту
(EXPERIMENTS.md §2б) говорит, что отключение оценщика на глубине 0 не стоит
задач, а на глубине 1 стоит; разбор вердиктов (§4а) — что на глубине 0 он
неотличим от монетки. Разметка проверяет это напрямую: на глубине 0 связь
вердикта с меткой должна быть слабой, на глубине 1 — заметной.

`unclear` в основные таблицы не входит: его доля печатается отдельно. Если она
велика (больше четверти), метрики шаткие и критерий надо уточнять.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

VALID = {"good", "bad", "unclear"}
BLOCK_RE = re.compile(r"^##[^\S\n]*Шаг[^\S\n]+(\d+)[^\S\n]*$", re.MULTILINE)
# [^\S\n], а не \s: \s включает перевод строки, и у незаполненной метки regex
# перепрыгивал на следующую строку, принимая за метку разделитель «---».
LABEL_RE = re.compile(r"^МЕТКА:[^\S\n]*(.*)$", re.MULTILINE)


def read_labels(form: Path) -> Dict[int, str]:
    """Достаёт метки из заполненной формы. Пустые пропускает, мусор — с ошибкой."""
    text = form.read_text(encoding="utf-8")
    starts = [(int(m.group(1)), m.start()) for m in BLOCK_RE.finditer(text)]
    if not starts:
        raise SystemExit(f"в {form} не найдено ни одного блока «## Шаг NNN»")
    out: Dict[int, str] = {}
    bad: List[str] = []
    for i, (idx, pos) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        found = LABEL_RE.search(text, pos, end)
        value = (found.group(1) if found else "").strip().lower()
        if not value:
            continue
        if value not in VALID:
            bad.append(f"шаг {idx}: '{value}'")
            continue
        out[idx] = value
    if bad:
        raise SystemExit("непонятные метки (допустимы good / bad / unclear):\n  "
                         + "\n  ".join(bad))
    return out


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Двусторонний точный тест Фишера для таблицы 2x2 [[a,b],[c,d]].

    Своя реализация, чтобы не тянуть scipy: суммируем вероятности всех таблиц с
    теми же краевыми суммами, не более вероятных, чем наблюдённая.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    def prob(x: int) -> float:
        return (comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1))
    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed * (1 + 1e-9)))


def report(depth_label: str, rows: List[dict]) -> None:
    tp = sum(1 for r in rows if r["accepted"] and r["label"] == "good")
    fp = sum(1 for r in rows if r["accepted"] and r["label"] == "bad")
    fn = sum(1 for r in rows if not r["accepted"] and r["label"] == "good")
    tn = sum(1 for r in rows if not r["accepted"] and r["label"] == "bad")
    total = tp + fp + fn + tn
    if total < 8:
        print(f"  {depth_label}: размечено слишком мало ({total})")
        return
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    p = fisher_exact(tp, fp, fn, tn)
    print(f"  {depth_label}")
    print(f"    таблица: принят+хорош {tp}, принят+плох {fp}, "
          f"отвергнут+хорош {fn}, отвергнут+плох {tn}")
    print(f"    precision {prec:.0%}  recall {rec:.0%}  specificity {spec:.0%}"
          f"   Фишер p = {p:.3f}")
    if fn:
        print(f"    ЗАРУБЛЕНО ХОРОШИХ: {fn} из {fn + tp} "
              f"({fn / (fn + tp):.0%} всех хороших шагов)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--form", type=Path, default=ROOT / "labeling/steps_to_label.md")
    ap.add_argument("--key", type=Path, default=ROOT / "labeling/steps_key.json")
    args = ap.parse_args()

    labels = read_labels(args.form)
    key = {item["idx"]: item for item in json.loads(args.key.read_text(encoding="utf-8"))}

    rows = [{**key[idx], "label": label} for idx, label in labels.items() if idx in key]
    print(f"Размечено блоков: {len(labels)} из {len(key)}")
    unclear = [r for r in rows if r["label"] == "unclear"]
    print(f"Из них unclear: {len(unclear)} ({len(unclear) / max(len(rows), 1):.0%})"
          + ("  ⚠️ много — критерий стоит уточнить" if len(rows) and len(unclear) > 0.25 * len(rows) else ""))
    if len(labels) < len(key):
        print(f"Не заполнено: {len(key) - len(labels)} — метрики считаются по заполненным")

    graded = [r for r in rows if r["label"] in ("good", "bad")]
    print(f"\nДоля хороших шагов среди размеченных: "
          f"{sum(1 for r in graded if r['label'] == 'good') / max(len(graded), 1):.0%}")

    print("\nПО ГЛУБИНАМ")
    depths = sorted({r["depth"] for r in graded})
    for d in depths:
        report(f"глубина {d}", [r for r in graded if r["depth"] == d])

    if len(depths) >= 2:
        print("\nСРАВНЕНИЕ ГЛУБИН — доля хороших шагов среди ОТВЕРГНУТЫХ")
        print("  (именно её предсказывает §4а: на глубине 0 оценщик рубит хорошее "
              "наравне с плохим)")
        for d in depths:
            sub = [r for r in graded if r["depth"] == d and not r["accepted"]]
            if len(sub) < 5:
                continue
            good = sum(1 for r in sub if r["label"] == "good")
            print(f"    глубина {d}: {good}/{len(sub)} = {good / len(sub):.0%} "
                  f"зарубленных были хорошими")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
