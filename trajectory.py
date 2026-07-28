"""Запись траекторий решения — что именно сгенерировала модель на каждом шаге
и каждой ветке, как это оценил оценщик и что сказал верификатор.

Зачем
-----
JSONL с результатами хранит только агрегаты (score'ы, счётчики), а полный текст
генераций живёт лишь в stdout-логе, который при --workers N перемешан между
задачами и не парсится надёжно. Из-за этого нельзя ответить на вопрос «почему
это нововведение сработало именно так»: не видно ни промпта, ни сырого ответа,
ни причины отказа оценщика.

Модуль пишет структурированную траекторию (JSON), которую затем рендерит
scripts/make_viewer.py в самодостаточный HTML.

Дизайн
------
* Потокобезопасно: раннер решает задачи в ThreadPoolExecutor, поэтому «текущая
  задача» живёт в threading.local(), а общий список — под локом.
* Пайплайн-агностично: и пошаговый qwen4b, и оригинальный 9B пишут одни и те же
  записи. Специфика роли уезжает в extra.
* finish_reason и truncated пишутся ВСЕГДА. Раньше обрыв ответа был виден только
  когда content пуст целиком; частичный обрыв (модель успела что-то сказать и
  упёрлась в лимит) не фиксировался нигде — и вопрос «есть ли обрывы» был
  принципиально не отвечаем по логам.
* Текст режется по TRAJECTORY_MAX_CHARS, иначе HTML распухает на гигабайты;
  факт усечения помечается, чтобы усечение просмотрщика не путали с обрывом
  генерации.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Сколько символов текста хранить на одну запись. Генератор с размышлениями
# выдаёт десятки килобайт; полный дамп 30 задач превращается в сотни мегабайт.
# 20000 резало типичную генерацию (~20-25k символов) ровно там, где интересно —
# в хвосте видно, оборвался ответ или дошёл до <step>. Снижайте через env, если
# HTML станет неповоротливым (30 задач при 40000 — порядка 10 МБ).
TRAJECTORY_MAX_CHARS = int(os.getenv("TRAJECTORY_MAX_CHARS", "40000"))

# Цена за 1M токенов — для локального vLLM обычно 0, но поле нужно, чтобы
# сравнивать прогоны на платных API в тех же единицах.
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "0"))


def _clip(text: Optional[str]) -> Dict[str, Any]:
    """Режет текст до лимита, честно помечая факт усечения просмотрщиком."""
    text = text or ""
    if len(text) <= TRAJECTORY_MAX_CHARS:
        return {"text": text, "clipped": False, "full_chars": len(text)}
    return {
        "text": text[:TRAJECTORY_MAX_CHARS],
        "clipped": True,
        "full_chars": len(text),
    }


@dataclass
class _Task:
    task_id: str
    problem: str
    started: float = field(default_factory=time.perf_counter)
    records: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class TrajectoryRecorder:
    """Собирает записи по задачам. Один экземпляр на процесс (см. RECORDER)."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._done: List[Dict[str, Any]] = []
        self.enabled = False
        self.run_meta: Dict[str, Any] = {}

    # -- жизненный цикл задачи -------------------------------------------------
    def start_task(self, task_id: str, problem: str, **meta) -> None:
        if not self.enabled:
            return
        self._local.task = _Task(task_id=str(task_id), problem=problem, meta=meta)

    def _current(self) -> Optional[_Task]:
        if not self.enabled:
            return None
        return getattr(self._local, "task", None)

    def record(
        self,
        *,
        stage: str,
        depth: Optional[int] = None,
        branch: Optional[int] = None,
        system: Optional[str] = None,
        user: Optional[str] = None,
        content: Optional[str] = None,
        reasoning: Optional[str] = None,
        finish_reason: Optional[str] = None,
        tokens: Optional[Dict[str, int]] = None,
        elapsed: Optional[float] = None,
        error: Optional[str] = None,
        **extra,
    ) -> None:
        """Одна запись = один вызов модели (или одно решение узла)."""
        task = self._current()
        if task is None:
            return

        tokens = tokens or {}
        tin = int(tokens.get("input") or 0)
        tout = int(tokens.get("output") or 0)
        total = int(tokens.get("total") or (tin + tout))
        cost = (tin / 1e6) * PRICE_IN_PER_M + (tout / 1e6) * PRICE_OUT_PER_M

        # Обрыв: сервер уперся в лимит токенов. Отдельно помечаем случай, когда
        # при этом не осталось ни текста, ни размышлений — тогда шаг потерян.
        truncated = finish_reason == "length"
        record = {
            "index": len(task.records),
            "stage": stage,
            "depth": depth,
            "branch": branch,
            "system": _clip(system),
            "user": _clip(user),
            "content": _clip(content),
            "reasoning": _clip(reasoning),
            "finish_reason": finish_reason,
            "truncated": truncated,
            "truncated_without_content": truncated and not (content or "").strip(),
            "tokens": {"input": tin, "output": tout, "total": total},
            "cost": cost,
            "elapsed": elapsed,
            "error": error,
        }
        record.update(extra)
        task.records.append(record)

    def finish_task(self, **summary) -> None:
        task = self._current()
        if task is None:
            return
        payload = {
            "task_id": task.task_id,
            "problem": task.problem,
            "elapsed": time.perf_counter() - task.started,
            "meta": task.meta,
            "records": task.records,
            **summary,
        }
        with self._lock:
            self._done.append(payload)
        self._local.task = None

    # -- вывод ----------------------------------------------------------------
    def dump(self, path: "Path | str") -> Optional[Path]:
        if not self.enabled:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tasks = sorted(self._done, key=lambda t: str(t.get("task_id")))
            payload = {"run": self.run_meta, "tasks": tasks}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return path


# Единый экземпляр: пайплайны импортируют его напрямую, раннер включает флагом.
RECORDER = TrajectoryRecorder()
