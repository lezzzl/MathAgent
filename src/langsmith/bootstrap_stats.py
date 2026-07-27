"""Paired bootstrap inference and report-row construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    paired_tasks: int
    missing_tasks: int
    unresolved_tasks: int
    run_1_accuracy: float | None
    run_2_accuracy: float | None
    diff: float | None
    p_value: float | None
    holm_p_value: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap_p_value(
    differences: np.ndarray, *, resamples: int = 10_000, seed: int = 42
) -> float:
    """Return a two-sided centered-null paired-bootstrap p-value."""

    if resamples < 1:
        raise ValueError("resamples must be positive")
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("differences must not be empty")
    observed = abs(float(values.mean()))
    if observed == 0:
        return 1.0
    centered = values - values.mean()
    rng = np.random.default_rng(seed)
    extreme = 0
    # Chunking avoids allocating resamples * task_count for large benchmarks.
    for start in range(0, resamples, 1_000):
        count = min(1_000, resamples - start)
        indexes = rng.integers(0, values.size, size=(count, values.size))
        means = centered[indexes].mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(means) >= observed))
    return (extreme + 1.0) / (resamples + 1.0)


def holm_adjust(values: Sequence[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(values)
    valid = sorted(
        ((index, float(value)) for index, value in enumerate(values) if value is not None),
        key=lambda item: item[1],
    )
    running = 0.0
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - rank) * value))
        adjusted[index] = running
    return adjusted


def compare_score_maps(
    left: Mapping[tuple[str, str], bool | None],
    right: Mapping[tuple[str, str], bool | None],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Compare all benchmark scopes represented by either experiment."""

    benchmarks = sorted({key[0] for key in set(left) | set(right)})
    results: list[BenchmarkResult] = []
    for offset, benchmark in enumerate(benchmarks):
        left_keys = {key for key in left if key[0] == benchmark}
        right_keys = {key for key in right if key[0] == benchmark}
        shared = sorted(left_keys & right_keys)
        eligible = [
            key for key in shared if left[key] is not None and right[key] is not None
        ]
        unresolved = len(shared) - len(eligible)
        missing = len(left_keys ^ right_keys)
        if not eligible:
            results.append(
                BenchmarkResult(
                    benchmark, 0, missing, unresolved, None, None, None, None
                )
            )
            continue
        left_values = np.asarray([bool(left[key]) for key in eligible], dtype=float)
        right_values = np.asarray([bool(right[key]) for key in eligible], dtype=float)
        differences = right_values - left_values
        results.append(
            BenchmarkResult(
                benchmark=benchmark,
                paired_tasks=len(eligible),
                missing_tasks=missing,
                unresolved_tasks=unresolved,
                run_1_accuracy=float(left_values.mean()),
                run_2_accuracy=float(right_values.mean()),
                diff=float(differences.mean()),
                p_value=paired_bootstrap_p_value(
                    differences, resamples=resamples, seed=seed + offset
                ),
            )
        )
    adjusted = holm_adjust([result.p_value for result in results])
    return [
        replace(result, holm_p_value=value)
        for result, value in zip(results, adjusted, strict=True)
    ]
