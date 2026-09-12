from proactive.eval.statistics import (
    average_set_size,
    grouped_bootstrap,
    holm_bonferroni,
    paired_grouped_bootstrap_difference,
)


def _row(group: str, size: int):
    return {
        "metadata": {"group_id": group},
        "prediction_set": list(range(size)),
        "targets": {"source_bits": [1, 0, 0], "six_way": 0},
        "bit_probabilities": [0.9, 0.1, 0.1],
        "six_way_probabilities": [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
        "acquisition_cost": 1,
    }


def test_grouped_bootstrap_resamples_base_groups() -> None:
    rows = [_row("g1", 1), _row("g1", 1), _row("g2", 3)]
    result = grouped_bootstrap(rows, metric=average_set_size, resamples=1000, confidence=0.95, seed=42)
    assert result["group_count"] == 2
    assert result["estimate"] == 5 / 3
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


def test_paired_bootstrap_requires_identical_groups() -> None:
    left = [_row("g1", 1), _row("g2", 1)]
    right = [_row("g1", 2), _row("g2", 2)]
    result = paired_grouped_bootstrap_difference(left, right, metric=average_set_size, resamples=1000, confidence=0.95, seed=42)
    assert result["estimate"] == -1.0
    assert 0.0 < result["p_value_two_sided"] < 0.01


def test_holm_bonferroni_is_monotone_and_preserves_input_order() -> None:
    corrected = holm_bonferroni([0.04, 0.001, 0.02], alpha=0.05)
    assert corrected[0]["p_value_holm"] == 0.04
    assert corrected[1]["p_value_holm"] == 0.003
    assert corrected[2]["p_value_holm"] == 0.04
    assert all(item["reject_familywise"] for item in corrected)
