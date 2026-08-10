"""Бюджет GPU: аренда слотов через flock.

Слот — это lock-файл в gpu-slots/. Сколько файлов, столько карт разрешено занять
одновременно; другого места, где настраивается бюджет, нет.

Два свойства делают схему устойчивой:

* захват «всё или ничего» под общим allocator.lock — иначе два запроса на две
  карты могли бы взять по одной и ждать друг друга вечно;
* flock снимается ядром при смерти процесса — поэтому упавший или убитый запуск
  не оставляет занятый слот, и чистильщик несуществующих блокировок не нужен.

Блокировка живёт в описании открытого файла, а не в дескрипторе, поэтому её
можно передать потомку: диспетчер захватывает слоты, отдаёт дескрипторы запуску
через pass_fds и закрывает свои копии. Аренда действует ровно столько, сколько
живёт процесс запуска.
"""

from __future__ import annotations

import fcntl
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.jobs.paths import slots_dir as default_slots_dir

SLOT_PATTERN = re.compile(r"^gpu(?P<index>\d+)\.lock$")
ALLOCATOR_LOCK_NAME = "allocator.lock"


class GpuLeaseError(RuntimeError):
    """Проблема конфигурации слотов, а не занятость карт."""


@dataclass(frozen=True)
class Slot:
    index: int
    path: Path


def discover_slots(slots_dir: Path | None = None) -> list[Slot]:
    """Перечисляет слоты по именам lock-файлов gpu<N>.lock."""
    directory = slots_dir or default_slots_dir()
    if not directory.is_dir():
        raise GpuLeaseError(
            f"GPU slot directory {directory} does not exist — see jobs/RUNNER.md"
        )

    slots = [
        Slot(index=int(match.group("index")), path=entry)
        for entry in sorted(directory.iterdir())
        if (match := SLOT_PATTERN.fullmatch(entry.name)) is not None
    ]
    if not slots:
        raise GpuLeaseError(
            f"no gpu<N>.lock files in {directory} — the GPU budget would be zero"
        )
    return sorted(slots, key=lambda slot: slot.index)


def slot_holders(slots_dir: Path | None = None) -> dict[int, str | None]:
    """Показывает занятые слоты и run_id их владельцев.

    Занятость определяется попыткой неблокирующего flock: если она не удалась,
    слот занят живым процессом. Владелец читается из содержимого файла и носит
    справочный характер — он для логов и комментария в PR.
    """
    holders: dict[int, str | None] = {}
    for slot in discover_slots(slots_dir):
        descriptor = os.open(slot.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                holders[slot.index] = slot.path.read_text(encoding="utf-8").strip() or None
            else:
                # Слот свободен: сразу отпускаем, мы только смотрим.
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                holders[slot.index] = None
        finally:
            os.close(descriptor)
    return holders


class GpuLease:
    """Аренда `count` карт. Захват неблокирующий: очередью управляет диспетчер."""

    def __init__(
        self,
        count: int,
        *,
        run_id: str,
        slots_dir: Path | None = None,
    ) -> None:
        self.count = count
        self.run_id = run_id
        self._slots_dir = slots_dir or default_slots_dir()
        self._descriptors: list[int] = []
        self.indices: list[int] = []

    # --- захват и освобождение -------------------------------------------------

    def acquire(self) -> bool:
        """Пытается занять ровно count слотов. Возвращает False, если их не хватает."""
        if self.count == 0:
            # Запуски без GPU не участвуют в очереди вовсе.
            return True
        if self._descriptors:
            raise GpuLeaseError("lease is already held")

        slots = discover_slots(self._slots_dir)
        if self.count > len(slots):
            raise GpuLeaseError(
                f"requested {self.count} GPU(s) but only {len(slots)} slot(s) exist"
            )

        # Общий замок сериализует сам подбор слотов: без него два запроса на две
        # карты могли бы взять по одной и заблокировать друг друга.
        allocator_path = self._slots_dir / ALLOCATOR_LOCK_NAME
        allocator = os.open(allocator_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(allocator, fcntl.LOCK_EX)
            acquired = self._try_slots(slots)
            if not acquired:
                self.release()
            return acquired
        finally:
            fcntl.flock(allocator, fcntl.LOCK_UN)
            os.close(allocator)

    def _try_slots(self, slots: list[Slot]) -> bool:
        """Берёт свободные слоты до нужного количества, ничего не ожидая."""
        for slot in slots:
            if len(self._descriptors) == self.count:
                break

            descriptor = os.open(slot.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(descriptor)
                continue

            # Дескриптор должен пережить exec потомка, иначе аренда оборвётся.
            os.set_inheritable(descriptor, True)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"{self.run_id}\n".encode())
            os.fsync(descriptor)
            self._descriptors.append(descriptor)
            self.indices.append(slot.index)

        return len(self._descriptors) == self.count

    def release(self) -> None:
        """Освобождает слоты. Повторный вызов безопасен."""
        for descriptor in self._descriptors:
            try:
                os.ftruncate(descriptor, 0)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                os.close(descriptor)
        self._descriptors.clear()
        self.indices.clear()

    # --- передача потомку ------------------------------------------------------

    @property
    def descriptors(self) -> tuple[int, ...]:
        """Дескрипторы для pass_fds: потомок держит аренду, пока жив."""
        return tuple(self._descriptors)

    def detach(self) -> None:
        """Закрывает свои копии дескрипторов, не снимая блокировку.

        Вызывается родителем после того, как потомок унаследовал дескрипторы:
        блокировка остаётся, пока открыт хотя бы один из них.
        """
        for descriptor in self._descriptors:
            os.close(descriptor)
        self._descriptors.clear()

    @property
    def cuda_visible_devices(self) -> str:
        """Значение CUDA_VISIBLE_DEVICES для арендованных карт."""
        return ",".join(str(index) for index in self.indices)

    # --- контекстный менеджер --------------------------------------------------

    def __enter__(self) -> GpuLease:  # noqa: PYI034 — typing.Self требует Python 3.11
        if not self.acquire():
            raise GpuLeaseError(f"cannot acquire {self.count} GPU(s) right now")
        return self

    def __exit__(self, *_exception: object) -> None:
        self.release()


def adopt_descriptors(descriptors: list[int], indices: list[int], run_id: str) -> GpuLease:
    """Восстанавливает объект аренды в потомке из унаследованных дескрипторов."""
    lease = GpuLease(len(descriptors), run_id=run_id)
    lease._descriptors = list(descriptors)
    lease.indices = list(indices)
    return lease
