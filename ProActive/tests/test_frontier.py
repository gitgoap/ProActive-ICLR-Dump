from __future__ import annotations

import pytest

from proactive.eval.frontier import matched_budget, pareto_flags


def test_pareto_flags_use_lower_cost_higher_quality() -> None:
    points = [
        {"mean_acquisition_cost": 1.0, "source_bit_macro_f1": 0.7},
        {"mean_acquisition_cost": 2.0, "source_bit_macro_f1": 0.8},
        {"mean_acquisition_cost": 2.0, "source_bit_macro_f1": 0.6},
    ]
    assert pareto_flags(points) == [True, True, False]


def test_matched_budget_rejects_overspend() -> None:
    matched_budget([{"acquisition_cost": 2}], 2)
    with pytest.raises(ValueError, match="matched"):
        matched_budget([{"acquisition_cost": 3}], 2)
