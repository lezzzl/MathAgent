from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from dashboard.runtime import (
    DATAFRAME_SERIALIZATION_LOCK,
    use_system_arrow_memory_pool,
)


def test_dashboard_uses_system_arrow_allocator() -> None:
    assert use_system_arrow_memory_pool() == "system"
    assert pa.default_memory_pool().backend_name == "system"


def test_dataframe_serialization_lock_is_shared_across_threads() -> None:
    def lock_identity(_: int) -> int:
        with DATAFRAME_SERIALIZATION_LOCK:
            return id(DATAFRAME_SERIALIZATION_LOCK)

    with ThreadPoolExecutor(max_workers=4) as executor:
        identities = list(executor.map(lock_identity, range(20)))

    assert set(identities) == {id(DATAFRAME_SERIALIZATION_LOCK)}
