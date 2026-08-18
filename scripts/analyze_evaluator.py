"""Насколько вердикт оценщика связан с исходом задачи — по уже готовым прогонам.

    python scripts/analyze_evaluator.py
    python scripts/analyze_evaluator.py --llm sglang-s42b sglang-s43b --random evalrandom-s42b

Зачем. Мы знаем, что оценщик окупается в сумме (без него 60/90 против 69/90), и
что отключение его на глубинах 0-1 не стоит точности при экономии 47%. Оба факта
измерены по ИТОГУ задачи. Здесь мы спускаемся на уровень отдельного вердикта.

ЧЕСТНО ПРО МЕТОД. Разметки шагов у нас нет: никто не говорил про конкретный шаг
«он верный». Поэтому меткой служит исход всей задачи — грубое приближение, и оно
смещено: в трудной задаче даже правильные шаги окажутся в классе «неверно».
Само по себе число precision в такой схеме не значит почти ничего.

Что делает эту схему рабочей — вторая рука. У нас есть прогоны, где вердикт
бросался монетой (--evaluator-mode random) и по построению НЕ зависит от
качества шага. Считая ту же метрику на них, мы получаем её нулевой уровень со
всеми смещениями сразу. Разница между руками — то, что вердикт действительно
несёт.

Ключевая величина — lift:

    lift = P(задача решена | шаг принят) - P(задача решена | шаг отвергнут)

У случайного оценщика он показывает, сколько «пользы» наскребает сама схема
измерения. У живого — сколько её на самом деле. Разбивка по глубине отвечает на
вопрос эксперимента 2 напрямую: если на глубинах 0-1 живой оценщик не отличается
от монетки, то понятно, почему его отключение там ничего не стоило.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
PREFIX = "nikita-qwen4b-v11-aime26-"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


class Verdict(NamedTuple):
    """Один вердикт оценщика вместе с исходом задачи, в которой он прозвучал."""
    run: str
    task: str
    depth: int
    accepted: bool
    in_recovery: bool
    task_correct: bool


def load(run: str) -> List[Verdict]:
    path = RUNS / f"{PREFIX}{run}" / "aime26_verified.jsonl" if not run.startswith("/") else Path(run)
    if not path.exists():
        raise SystemExit(f"нет файла {path}")
    out: List[Verdict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ok = bool(rec.get("is_correct"))
        metrics = (rec.get("metadata") or {}).get("agent_metrics") or {}
        for round_ in metrics.get("eval_history") or []:
            for score in round_.get("scores") or []:
                out.append(Verdict(run, str(rec.get("task_id")), int(round_.get("depth") or 0),
                                   float(score) >= 0.5, bool(round_.get("in_recovery")), ok))
    return out


def rates(vs: List[Verdict]) -> Dict[str, Any]:
    acc = [v for v in vs if v.accepted]
    rej = [v for v in vs if not v.accepted]
    p_acc = sum(v.task_correct for v in acc) / len(acc) if acc else float("nan")
    p_rej = sum(v.task_correct for v in rej) / len(rej) if rej else float("nan")
    return {"n": len(vs), "accept_rate": len(acc) / len(vs) if vs else float("nan"),
            "p_correct_if_accepted": p_acc, "p_correct_if_rejected": p_rej,
            "lift": p_acc - p_rej}


def proxy_prf(vs: List[Verdict]) -> Dict[str, float]:
    """precision/recall в схеме «метка = исход задачи». Смотреть только в
    сравнении с той же величиной у случайного оценщика."""
    tp = sum(1 for v in vs if v.accepted and v.task_correct)
    fp = sum(1 for v in vs if v.accepted and not v.task_correct)
    fn = sum(1 for v in vs if not v.accepted and v.task_correct)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def bootstrap_lift(vs: List[Verdict], n_iter: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Интервал для lift бутстрэпом ПО ЗАДАЧАМ, а не по вердиктам.

    Вердикты внутри одной задачи делят одну метку и потому не независимы:
    бутстрэп по вердиктам дал бы интервал уже настоящего в разы.
    """
    by_task: Dict[tuple, List[Verdict]] = {}
    for v in vs:
        by_task.setdefault((v.run, v.task), []).append(v)
    tasks = list(by_task.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(n_iter):
        sample: List[Verdict] = []
        for _ in range(len(tasks)):
            sample.extend(rng.choice(tasks))
        lift = rates(sample)["lift"]
        if lift == lift:  # не NaN
            vals.append(lift)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def table(title: str, arms: Dict[str, List[Verdict]], depths: List[Any]) -> None:
    print(f"\n{title}")
    print(f"  {'рука':<12}{'глубина':>9}{'вердиктов':>11}{'принято':>10}"
          f"{'P(верно|принят)':>17}{'P(верно|отвергнут)':>20}{'lift':>9}")
    for name, vs in arms.items():
        for d in depths:
            sub = vs if d == "все" else [v for v in vs if v.depth == d]
            if len(sub) < 25:
                continue
            r = rates(sub)
            print(f"  {name:<12}{str(d):>9}{r['n']:>11}{r['accept_rate']:>9.0%}"
                  f"{r['p_correct_if_accepted']:>17.0%}{r['p_correct_if_rejected']:>20.0%}"
                  f"{r['lift']:>+9.0%}")


def bootstrap_delta(a: List[Verdict], b: List[Verdict],
                    n_iter: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Разница lift между руками и её интервал. Бутстрэп по задачам в каждой
    руке независимо — руки это разные прогоны, общих задач у них нет."""
    def by_task(vs):
        d: Dict[tuple, List[Verdict]] = {}
        for v in vs:
            d.setdefault((v.run, v.task), []).append(v)
        return list(d.values())
    ta, tb = by_task(a), by_task(b)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_iter):
        sa = [v for _ in range(len(ta)) for v in rng.choice(ta)]
        sb = [v for _ in range(len(tb)) for v in rng.choice(tb)]
        la, lb = rates(sa)["lift"], rates(sb)["lift"]
        if la == la and lb == lb:
            vals.append(la - lb)
    vals.sort()
    point = rates(a)["lift"] - rates(b)["lift"]
    if not vals:
        return point, float("nan"), float("nan")
    return point, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", nargs="+",
                    default=["sglang-s42b", "sglang-s43b", "sglang-s44", "s45"],
                    help="прогоны с обычным оценщиком")
    ap.add_argument("--random", nargs="+",
                    default=["evalrandom-s42b", "evalrandom-s43b", "evalrandom-s44"],
                    help="прогоны со случайным оценщиком (нулевой уровень)")
    ap.add_argument("--depth-split", type=int, default=2,
                    help="граница, по которой делим мелкую и глубокую зону "
                         "(та же, что у --evaluator-min-depth)")
    args = ap.parse_args()

    llm = [v for r in args.llm for v in load(r)]
    rnd = [v for r in args.random for v in load(r)]
    arms = {"оценщик": llm, "монетка": rnd}

    print(f"Прогоны с оценщиком: {', '.join(args.llm)} — {len(llm)} вердиктов")
    print(f"Прогоны с монеткой:  {', '.join(args.random)} — {len(rnd)} вердиктов")
    print("\nМетка = исход всей задачи. Абсолютные числа смещены; смысл несёт "
          "\nтолько разница между руками — у монетки вердикт по построению не "
          "\nзависит от качества шага.")

    table("ПО ГЛУБИНЕ", arms, ["все", 0, 1, 2, 3])

    print("\nГЛАВНОЕ: НАСКОЛЬКО ВЕРДИКТ ЛУЧШЕ МОНЕТКИ (разница lift и её интервал)")
    print(f"  {'глубина':<10}{'вердиктов llm/rnd':>20}{'lift оценщика':>15}"
          f"{'lift монетки':>14}{'разница':>10}{'95% интервал':>20}")
    for d in ["все", 0, 1, 2, 3]:
        sub_l = llm if d == "все" else [v for v in llm if v.depth == d]
        sub_r = rnd if d == "все" else [v for v in rnd if v.depth == d]
        if len(sub_l) < 25 or len(sub_r) < 25:
            continue
        point, lo, hi = bootstrap_delta(sub_l, sub_r)
        print(f"  {str(d):<10}{f'{len(sub_l)}/{len(sub_r)}':>20}"
              f"{rates(sub_l)['lift']:>+15.0%}{rates(sub_r)['lift']:>+14.0%}"
              f"{point:>+10.0%}{f'[{lo:+.0%}, {hi:+.0%}]':>20}")

    d = args.depth_split
    print(f"\nЗОНЫ (граница {d} — та же, что у --evaluator-min-depth {d})")
    print(f"  {'рука':<12}{'зона':<14}{'вердиктов':>11}{'lift':>9}{'95% интервал':>22}")
    for name, vs in arms.items():
        for label, sub in (("мелкая 0..%d" % (d - 1), [v for v in vs if v.depth < d]),
                           ("глубокая %d+" % d, [v for v in vs if v.depth >= d])):
            if len(sub) < 25:
                continue
            lift = rates(sub)["lift"]
            lo, hi = bootstrap_lift(sub)
            print(f"  {name:<12}{label:<14}{len(sub):>11}{lift:>+9.0%}"
                  f"{f'[{lo:+.0%}, {hi:+.0%}]':>22}")

    print("\nPRECISION / RECALL в той же схеме (метка = исход задачи)")
    print(f"  {'рука':<12}{'зона':<14}{'precision':>11}{'recall':>9}{'F1':>8}")
    for name, vs in arms.items():
        for label, sub in (("все", vs), ("мелкая 0..%d" % (d - 1), [v for v in vs if v.depth < d]),
                           ("глубокая %d+" % d, [v for v in vs if v.depth >= d])):
            if len(sub) < 25:
                continue
            p = proxy_prf(sub)
            print(f"  {name:<12}{label:<14}{p['precision']:>11.0%}{p['recall']:>9.0%}{p['f1']:>8.0%}")

    print("\nЧитать так: precision у монетки — это доля решённых задач среди всех "
          "\nвердиктов, то есть чистая база. Насколько живой оценщик её "
          "\nпревышает, столько информации в вердикте и есть.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
