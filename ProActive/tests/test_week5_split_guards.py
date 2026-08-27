from __future__ import annotations

import numpy as np

from proactive.eval.shortcut_controls import IdentityVocabulary, control_features


def test_identity_controls_are_explicit_and_unknown_values_do_not_expand_schema() -> None:
    train_metadata = {"dataset": ["pope", "vsr"], "model_id": ["qwen", "gemma"]}
    vocabulary = IdentityVocabulary.fit(train_metadata)
    train = control_features(
        clean_features=np.ones((2, 4), dtype=np.float32),
        metadata=train_metadata,
        vocabulary=vocabulary,
    )
    test = control_features(
        clean_features=np.ones((1, 4), dtype=np.float32),
        metadata={"dataset": ["unseen"], "model_id": ["unseen"]},
        vocabulary=vocabulary,
    )
    assert train.shape == (2, 8)
    assert test.shape == (1, 8)
    assert test[0, 4:].sum() == 0


def test_probe_evidence_control_requires_mask() -> None:
    vocabulary = IdentityVocabulary.fit({"dataset": ["pope"], "model_id": ["qwen"]})
    try:
        control_features(
            clean_features=np.ones((1, 4)),
            metadata={"dataset": ["pope"], "model_id": ["qwen"]},
            vocabulary=vocabulary,
            probe_numeric=np.zeros((1, 7, 10)),
        )
    except ValueError as exc:
        assert "together" in str(exc)
    else:
        raise AssertionError("Missing acquired mask was silently accepted")
