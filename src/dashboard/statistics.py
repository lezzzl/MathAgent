"""Paired statistical comparisons for benchmark correctness outcomes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Hashable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest


ScoreKey = tuple[str, str]


@dataclass(frozen=True)
class PairwiseResult:
    """Effect size, uncertainty, and exact paired test for two runs."""

    left_run: str
    right_run: str
    scope: str
    weighting: str
    paired_tasks: int
    unresolved_tasks: int
    left_accuracy: float | None
    right_accuracy: float | None
    accuracy_delta: float | None
    ci_low: float | None
    ci_high: float | None
    left_wins: int
    right_wins: int
    ties: int
    p_value: float | None
    adjusted_p_value: float | None = None


def _bootstrap_interval(
    differences_by_group: Mapping[Hashable, np.ndarray],
    *,
    weighting: str,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap paired differences, stratified by benchmark."""

    rng = np.random.default_rng(seed)
    group_arrays = list(differences_by_group.values())
    if all(np.all(values == values[0]) for values in group_arrays):
        value = (
            float(np.concatenate(group_arrays).mean())
            if weighting == "task"
            else float(np.mean([values.mean() for values in group_arrays]))
        )
        return value, value

    bootstrap = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        resampled = [
            values[rng.integers(0, len(values), size=len(values))]
            for values in group_arrays
        ]
        if weighting == "task":
            bootstrap[index] = np.concatenate(resampled).mean()
        else:
            bootstrap[index] = np.mean([values.mean() for values in resampled])
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return float(low), float(high)


def compare_paired_scores(
    left_run: str,
    right_run: str,
    left: Mapping[ScoreKey, bool | None],
    right: Mapping[ScoreKey, bool | None],
    *,
    scope: str,
    weighting: str = "task",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> PairwiseResult:
    """Compare shared tasks using paired bootstrap and an exact discordance test."""

    if weighting not in {"task", "benchmark"}:
        raise ValueError("weighting must be 'task' or 'benchmark'")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    shared = sorted(set(left) & set(right))
    eligible = [key for key in shared if left[key] is not None and right[key] is not None]
    unresolved = len(shared) - len(eligible)
    if not eligible:
        return PairwiseResult(
            left_run,
            right_run,
            scope,
            weighting,
            0,
            unresolved,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            0,
            None,
        )

    left_values = np.asarray([bool(left[key]) for key in eligible], dtype=float)
    right_values = np.asarray([bool(right[key]) for key in eligible], dtype=float)
    differences = left_values - right_values
    groups: dict[str, np.ndarray] = {}
    for benchmark in sorted({key[0] for key in eligible}):
        indexes = [i for i, key in enumerate(eligible) if key[0] == benchmark]
        groups[benchmark] = differences[indexes]

    if weighting == "task":
        left_accuracy = float(left_values.mean())
        right_accuracy = float(right_values.mean())
    else:
        left_accuracy = float(
            np.mean(
                [
                    left_values[
                        [i for i, key in enumerate(eligible) if key[0] == benchmark]
                    ].mean()
                    for benchmark in groups
                ]
            )
        )
        right_accuracy = float(
            np.mean(
                [
                    right_values[
                        [i for i, key in enumerate(eligible) if key[0] == benchmark]
                    ].mean()
                    for benchmark in groups
                ]
            )
        )
    delta = left_accuracy - right_accuracy
    ci_low, ci_high = _bootstrap_interval(
        groups,
        weighting=weighting,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )

    left_wins = int(np.sum(differences == 1))
    right_wins = int(np.sum(differences == -1))
    ties = int(np.sum(differences == 0))
    discordant = left_wins + right_wins
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(left_wins, discordant, 0.5).pvalue)
    )
    return PairwiseResult(
        left_run,
        right_run,
        scope,
        weighting,
        len(eligible),
        unresolved,
        left_accuracy,
        right_accuracy,
        delta,
        ci_low,
        ci_high,
        left_wins,
        right_wins,
        ties,
        p_value,
    )


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Adjust a family of p-values with Holm's step-down procedure."""

    adjusted: list[float | None] = [None] * len(p_values)
    valid = [(index, float(value)) for index, value in enumerate(p_values) if value is not None]
    valid.sort(key=lambda item: item[1])
    running_max = 0.0
    total = len(valid)
    for rank, (original_index, p_value) in enumerate(valid):
        candidate = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def adjust_results(results: Sequence[PairwiseResult]) -> list[PairwiseResult]:
    """Return results with Holm-adjusted p-values for this displayed family."""

    adjusted = holm_adjust([result.p_value for result in results])
    return [
        replace(result, adjusted_p_value=value)
        for result, value in zip(results, adjusted, strict=True)
    ]
