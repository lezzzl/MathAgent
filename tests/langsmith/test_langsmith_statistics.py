from __future__ import annotations

import numpy as np
import pytest

from bootstrap_stats import compare_score_maps, holm_adjust, paired_bootstrap_p_value


def test_centered_bootstrap_is_reproducible_and_smoothed() -> None:
    differences = np.asarray([1.0] * 8 + [0.0] * 2)
    first = paired_bootstrap_p_value(differences, resamples=500, seed=9)
    second = paired_bootstrap_p_value(differences, resamples=500, seed=9)
    assert first == second
    assert 1 / 501 <= first <= 1


def test_comparison_uses_run_2_minus_run_1_and_tracks_exclusions() -> None:
    left = {
        ("A", "1"): False,
        ("A", "2"): True,
        ("A", "3"): None,
        ("A", "left-only"): True,
    }
    right = {
        ("A", "1"): True,
        ("A", "2"): True,
        ("A", "3"): False,
        ("A", "right-only"): False,
    }
    row = compare_score_maps(left, right, resamples=100, seed=4)[0]
    assert row.paired_tasks == 2
    assert row.unresolved_tasks == 1
    assert row.missing_tasks == 2
    assert row.run_1_accuracy == pytest.approx(0.5)
    assert row.run_2_accuracy == pytest.approx(1)
    assert row.diff == pytest.approx(0.5)


def test_no_eligible_pair_is_preserved() -> None:
    row = compare_score_maps(
        {("A", "1"): None}, {("A", "1"): True}, resamples=10
    )[0]
    assert row.paired_tasks == 0
    assert row.p_value is None
    assert row.holm_p_value is None


def test_holm_adjustment() -> None:
    assert holm_adjust([0.01, 0.04, 0.03, None]) == pytest.approx(
        [0.03, 0.06, 0.06, None], nan_ok=True
    )
