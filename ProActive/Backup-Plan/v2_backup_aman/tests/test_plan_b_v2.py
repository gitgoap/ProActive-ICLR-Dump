from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[1]
V1_ROOT = HERE.parent / "v1_backup_sourish_sir_plan"
for location in (V1_ROOT, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from plan_b.core import Candidate, Observation  # noqa: E402
from plan_b_v2.features import (  # noqa: E402
    CONFIDENCE_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    cached_confidence_features,
    source_confidence,
    structural_features,
)
from plan_b_v2.pipeline import _leaderboard_key, _select_safe_correction, load_v2_config  # noqa: E402


def _row() -> dict:
    probe = {
        "norm_answer": "no",
        "raw_answer": "no",
        "valid": True,
        "applicable": True,
        "conf_shift": 0.1,
        "margin_shift": 0.05,
        "entropy_shift": -0.1,
    }
    return {
        "instance_id": "example",
        "clean": {
            "norm_answer": "yes",
            "raw_answer": "yes",
            "valid": True,
            "answer_prob": 0.6,
            "token_margin_mean": 0.2,
            "token_entropy_mean": 0.8,
        },
        "probes": {name: dict(probe) for name in ("grounding", "blur", "crop", "brightness", "noise", "blank")},
    }


def test_feature_contract_has_no_gold_dataset_or_model_identity() -> None:
    parameters = inspect.signature(structural_features).parameters
    assert "gold" not in parameters
    forbidden = {"dataset", "model", "gold", "label", "correct"}
    assert not any(any(token in name for token in forbidden) for name in STRUCTURAL_FEATURE_NAMES)


def test_structural_features_encode_independent_grounding_support() -> None:
    observations = [
        Observation("clean", "yes", "yes", True, True),
        Observation("grounding", "no", "no", True, True),
        Observation("blur", "no", "no", True, True),
    ]
    candidates = [
        Candidate("yes", ("clean",), "yes", 1, 0),
        Candidate("no", ("grounding", "blur"), "no", 2, 1),
    ]
    values = structural_features(candidates[1], candidates, observations, 2)
    mapped = dict(zip(STRUCTURAL_FEATURE_NAMES, values))
    assert mapped["candidate_has_grounding_support"] == 1.0
    assert mapped["candidate_independent_grounding_support_fraction"] == 1.0
    assert mapped["candidate_vote_gap"] > 0
    assert len(values) == len(STRUCTURAL_FEATURE_NAMES)


def test_cached_confidence_is_reconstructed_and_explicitly_isolated() -> None:
    row = _row()
    assert source_confidence(row, "grounding") == (0.7, 0.25, 0.7000000000000001)
    observations = [
        Observation("clean", "yes", "yes", True, True),
        Observation("grounding", "no", "no", True, True),
    ]
    candidates = [
        Candidate("yes", ("clean",), "yes", 1, 0),
        Candidate("no", ("grounding",), "no", 1, 1),
    ]
    values = cached_confidence_features(row, candidates[1], candidates, observations, 1)
    mapped = dict(zip(CONFIDENCE_FEATURE_NAMES, values))
    assert mapped["candidate_minus_clean_probability"] > 0
    assert len(values) == len(CONFIDENCE_FEATURE_NAMES)


def test_safe_selector_abstains_to_original_below_threshold() -> None:
    item = {
        "observations": [Observation("clean", "yes", "yes", True, True)],
        "alternatives": [Candidate("no", ("grounding",), "no", 1, 1)],
        "repair_probabilities": [0.8],
        "damage_probabilities": [0.4],
    }
    selection, _, _, _, abstained = _select_safe_correction(
        item, repair_minimum=0.7, damage_penalty=2.0, threshold=0.1, tolerance=1e-12
    )
    assert selection.source == "clean"
    assert abstained is True


def test_safe_selector_switches_only_when_both_safety_conditions_pass() -> None:
    item = {
        "observations": [Observation("clean", "yes", "yes", True, True)],
        "alternatives": [Candidate("no", ("grounding", "blur"), "no", 2, 1)],
        "repair_probabilities": [0.95],
        "damage_probabilities": [0.01],
    }
    selection, repair, damage, score, abstained = _select_safe_correction(
        item, repair_minimum=0.85, damage_penalty=4.0, threshold=0.8, tolerance=1e-12
    )
    assert selection.norm_answer == "no"
    assert repair == 0.95 and damage == 0.01 and abs(score - 0.91) < 1e-12
    assert abstained is False


def test_validation_ranking_prefers_gain_then_safety() -> None:
    base = {
        "six_cell_net_repair_gain": 0.02,
        "repairs": 3,
        "damage": 1,
        "repair_probability_minimum": 0.7,
        "safe_score_threshold": 0.3,
        "damage_penalty": 2.0,
        "feature_set": "structural_v2",
        "model_name": "a",
    }
    safer = dict(base, damage=0, model_name="b")
    assert _leaderboard_key(safer) < _leaderboard_key(base)


def test_config_freezes_posthoc_and_no_test_tuning_contract() -> None:
    config = load_v2_config(HERE / "config" / "plan_b_v2_config.json")
    constraints = config["scientific_constraints"]
    assert constraints["post_hoc_after_v1_validation"] is True
    assert constraints["test_or_shift_tuning"] is False
    assert constraints["cached_confidence_is_unverified_proxy"] is True
    assert config["gates"]["maximum_validation_break_rate"] == 0.01
