"""Process-wide runtime guards for Streamlit's native dataframe serializer."""

from __future__ import annotations

from threading import RLock


# Streamlit runs every browser tab in a separate thread. PyArrow's mimalloc pool
# can crash while initializing on those threads on macOS, so all Arrow-backed
# dataframe rendering is serialized as an additional safety boundary.
DATAFRAME_SERIALIZATION_LOCK = RLock()


def use_system_arrow_memory_pool() -> str:
    """Select Arrow's thread-safe system allocator and return its backend name."""

    import pyarrow as pa

    pool = pa.system_memory_pool()
    pa.set_memory_pool(pool)
    return pa.default_memory_pool().backend_name
