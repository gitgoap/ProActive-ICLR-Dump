from __future__ import annotations

import pytest

from proactive.eval.diagnostic_metrics import diagnostic_metrics


def test_diagnostic_metrics_perfect_predictions() -> None:
    bits = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]
    bit_probability = [[0.01 if value == 0 else 0.99 for value in row] for row in bits]
    six_probability = [[0.99 if column == row else 0.002 for column in range(6)] for row in range(6)]
    # Normalize the deliberately sharp rows exactly.
    six_probability = [[value / sum(row) for value in row] for row in six_probability]
    metrics = diagnostic_metrics(bit_probability, six_probability, bits, list(range(6)))
    assert metrics["source_bit_macro_f1"] == 1.0
    assert metrics["six_way_macro_f1"] == 1.0


def test_auroc_fails_closed_on_one_class_bit() -> None:
    with pytest.raises(ValueError, match="one target class"):
        diagnostic_metrics(
            [[0.1, 0.1, 0.1], [0.2, 0.9, 0.9]],
            [[0.9, 0.02, 0.02, 0.02, 0.02, 0.02], [0.02, 0.9, 0.02, 0.02, 0.02, 0.02]],
            [[0, 0, 0], [0, 1, 1]],
            [0, 1],
        )
