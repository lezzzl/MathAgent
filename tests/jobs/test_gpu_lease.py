from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.jobs.gpu_lease import GpuLease, GpuLeaseError, discover_slots, slot_holders

ROOT = Path(__file__).resolve().parents[2]


def test_discovers_slots_from_lock_files(slots: Path) -> None:
    assert [slot.index for slot in discover_slots(slots)] == [0, 1]


def test_empty_directory_is_a_configuration_error(tmp_path: Path) -> None:
    """Ноль слотов — не «все заняты», а неверная установка: бюджет был бы нулевым."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(GpuLeaseError, match="no gpu<N>.lock files"):
        discover_slots(empty)


def test_single_gpu_leases_exhaust_the_budget(slots: Path) -> None:
    first = GpuLease(1, run_id="first", slots_dir=slots)
    second = GpuLease(1, run_id="second", slots_dir=slots)
    third = GpuLease(1, run_id="third", slots_dir=slots)

    assert first.acquire() is True
    assert second.acquire() is True
    assert third.acquire() is False
    assert third.indices == []

    assert slot_holders(slots) == {0: "first", 1: "second"}

    first.release()
    assert third.acquire() is True
    assert third.indices == [0]


def test_two_gpu_lease_is_all_or_nothing(slots: Path) -> None:
    """Частичный захват привёл бы к взаимной блокировке двух двухкарточных запусков."""
    single = GpuLease(1, run_id="single", slots_dir=slots)
    assert single.acquire() is True

    both = GpuLease(2, run_id="both", slots_dir=slots)
    assert both.acquire() is False
    # Свободный слот остался свободным, а не оказался занят наполовину.
    assert slot_holders(slots) == {0: "single", 1: None}

    single.release()
    assert both.acquire() is True
    assert both.indices == [0, 1]
    assert both.cuda_visible_devices == "0,1"


def test_zero_gpu_lease_never_queues(slots: Path) -> None:
    """Запуски без GPU не должны ждать освобождения карт."""
    busy = GpuLease(2, run_id="busy", slots_dir=slots)
    assert busy.acquire() is True

    cpu_only = GpuLease(0, run_id="cpu", slots_dir=slots)
    assert cpu_only.acquire() is True
    assert cpu_only.cuda_visible_devices == ""


def test_requesting_more_than_the_budget_is_an_error(slots: Path) -> None:
    with pytest.raises(GpuLeaseError, match="only 2 slot"):
        GpuLease(3, run_id="greedy", slots_dir=slots).acquire()


def test_lease_is_released_when_the_holder_is_killed(slots: Path) -> None:
    """Убитый запуск не должен оставлять карту занятой навсегда."""
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys, time;"
                f"sys.path.insert(0, {str(ROOT)!r});"
                "from scripts.jobs.gpu_lease import GpuLease;"
                "from pathlib import Path;"
                f"lease = GpuLease(1, run_id='holder', slots_dir=Path({str(slots)!r}));"
                "print(lease.acquire(), flush=True);"
                "time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "True"
        assert slot_holders(slots)[0] == "holder"

        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=10)
    finally:
        if holder.poll() is None:
            holder.kill()

    # Ядро снимает flock вместе с процессом — отдельный чистильщик не нужен.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and slot_holders(slots)[0] is not None:
        time.sleep(0.05)
    assert slot_holders(slots)[0] is None

    reused = GpuLease(1, run_id="reused", slots_dir=slots)
    assert reused.acquire() is True
