from __future__ import annotations

import math

import numpy as np
import pytest

from proactive.conformal.aps import aps_scores, evaluate_sets, finite_sample_quantile, fit_aps, prediction_sets


def test_aps_score_and_finite_sample_rank_are_plan_exact() -> None:
    probabilities = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.4, 0.35, 0.25]]
    scores = aps_scores(probabilities, [0, 1, 2])
    assert np.allclose(scores, [0.6, 0.7, 1.0])
    assert finite_sample_quantile(scores, alpha=0.1) == 1.0
    fitted = fit_aps(probabilities, [0, 1, 2], alpha=0.1)
    assert fitted.calibration_count == 3
    assert math.isclose(fitted.target_coverage, 0.9)


def test_prediction_sets_and_metrics_fail_closed() -> None:
    sets = prediction_sets([[0.6, 0.3, 0.1], [0.45, 0.35, 0.2]], 0.9)
    assert sets == [[0, 1], [0, 1]]
    metrics = evaluate_sets(sets, [0, 2])
    assert metrics["coverage"] == 0.5
    with pytest.raises(ValueError, match="sum to one"):
        prediction_sets([[0.8, 0.8]], 0.9)
    with pytest.raises(ValueError, match="non-empty"):
        finite_sample_quantile([], 0.1)
