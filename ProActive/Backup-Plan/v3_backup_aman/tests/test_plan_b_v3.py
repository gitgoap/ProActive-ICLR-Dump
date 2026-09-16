from __future__ import annotations

import inspect
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = V3_ROOT.parent
REPO_ROOT = BACKUP_ROOT.parent
for location in (BACKUP_ROOT / "v1_backup_sourish_sir_plan", BACKUP_ROOT / "v2_backup_aman", V3_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from plan_b.core import Candidate, Observation  # noqa: E402
from plan_b_v3.pipeline import (  # noqa: E402
    CandidateExample,
    _build_examples,
    _confirmation_gate,
    _ensemble_predictions,
    _leaderboard_key,
    assign_grouped_folds,
    load_v3_config,
    mine_hard_negatives,
)


def _config() -> dict:
    return load_v3_config(V3_ROOT / "config" / "plan_b_v3_config.json", REPO_ROOT)


def _payload(answer: str, *, confidence: float = 0.1) -> dict:
    return {
        "raw_answer": answer,
        "norm_answer": answer,
        "valid": True,
        "applicable": True,
        "conf_shift": confidence,
        "margin_shift": confidence / 2,
        "entropy_shift": -confidence,
    }


def _row(instance: str = "example", verifier_answer: str = "yes") -> dict:
    return {
        "instance_id": instance,
        "group_id": instance,
        "dataset": "pope",
        "split": "val",
        "model_id": "model",
        "answer_type": "binary",
        "image_path": f"/images/{instance}.jpg",
        "gold_answer": "yes",
        "clean": {
            "raw_answer": "no",
            "norm_answer": "no",
            "valid": True,
            "correct": 0,
            "answer_prob": 0.55,
            "token_margin_mean": 0.1,
            "token_entropy_mean": 0.9,
        },
        "probes": {
            "grounding": _payload("yes", confidence=0.2),
            "blur": _payload("yes", confidence=0.15),
            "crop": _payload(verifier_answer, confidence=0.1),
            "brightness": _payload("no", confidence=0.05),
            "noise": _payload("no", confidence=0.03),
        },
    }


def test_config_freezes_posthoc_no_leakage_protocol() -> None:
    config = _config()
    assert config["implementation_authorization"] == "OWNER_APPROVED_2026-09-16"
    assert config["oof"]["group_key"] == "image_key"
    assert config["verification_probe_candidates"] == ["crop", "brightness", "noise"]
    assert config["scientific_constraints"]["hard_negative_mining_uses_train_oof_only"] is True
    assert config["scientific_constraints"]["test_or_shift_tuning"] is False
    assert float(config["gates"]["maximum_validation_break_rate"]) <= 0.01


def test_grouped_folds_are_deterministic_and_keep_images_together() -> None:
    rows = [_row("a"), _row("b"), _row("c"), _row("d"), _row("e"), _row("f")]
    rows.append(dict(_row("a-second"), image_path=rows[0]["image_path"]))
    first = assign_grouped_folds(rows, folds=5, seed=42)
    second = assign_grouped_folds(list(reversed(rows)), folds=5, seed=42)
    assert first == second
    assert len(first) == 6
    assert set(first.values()) == {0, 1, 2, 3, 4}


def test_hard_negative_uses_oof_model_proposals_only() -> None:
    candidate = Candidate("yes", ("grounding", "blur"), "yes", 2, 1)
    example = CandidateExample(
        row=_row(),
        observations=[Observation("clean", "no", "no", True, True)],
        candidates=[candidate],
        candidate=candidate,
        matched_cost=2,
        features=[0.0],
        original_correct=1,
        candidate_correct=0,
        repair_label=0,
        damage_label=1,
        base_weight=1.0,
        fold=0,
    )
    oof = {
        "xgboost_default": {"repair": [0.9], "damage": [0.1]},
        "lightgbm_default": {"repair": [0.8], "damage": [0.2]},
        "mlp_64_32_regularized": {"repair": [0.1], "damage": [0.9]},
    }
    flags, rows = mine_hard_negatives([example], oof, _config())
    assert flags == [True]
    assert rows[0]["models_proposing"] == 2


def test_active_verifier_switches_only_on_agreement_and_counts_cost() -> None:
    config = _config()
    agreeing = _row(verifier_answer="yes")
    examples = _build_examples([agreeing], config, None)
    scores = {
        name: {"repair": [0.99] * len(examples), "damage": [0.01] * len(examples)}
        for name in ("xgboost_default", "lightgbm_default", "mlp_64_32_regularized")
    }
    switched = _ensemble_predictions(
        [agreeing], examples, scores, config,
        verifier_probe="crop", repair_minimum=0.9, damage_maximum=0.1, safe_minimum=0.5,
    )[0]
    assert switched["selected_norm_answer"] == "yes"
    assert switched["repair"] == 1
    assert switched["verifier_acquired"] == 1
    assert switched["verifier_agreed"] == 1
    assert switched["matched_cost"] == 2
    assert switched["operational_cost"] == 3

    disagreeing = _row(instance="other", verifier_answer="no")
    other_examples = _build_examples([disagreeing], config, None)
    other_scores = {
        name: {"repair": [0.99] * len(other_examples), "damage": [0.01] * len(other_examples)}
        for name in scores
    }
    kept = _ensemble_predictions(
        [disagreeing], other_examples, other_scores, config,
        verifier_probe="crop", repair_minimum=0.9, damage_maximum=0.1, safe_minimum=0.5,
    )[0]
    assert kept["selected_norm_answer"] == "no"
    assert kept["correction_abstained"] == 1
    assert kept["verifier_acquired"] == 1 and kept["verifier_agreed"] == 0


def test_unanimity_is_encoded_in_selector_api_without_identity_features() -> None:
    source = inspect.getsource(_ensemble_predictions)
    assert "_unanimous_proposal" in source
    assert "model_id" not in inspect.signature(_ensemble_predictions).parameters
    assert "dataset" not in inspect.signature(_ensemble_predictions).parameters


def test_validation_ranking_prefers_gain_then_repairs_then_less_damage() -> None:
    base = {
        "six_cell_macro_gain": 0.02,
        "repairs": 3,
        "damage": 1,
        "repair_probability_minimum": 0.7,
        "damage_probability_maximum": 0.25,
        "safe_score_minimum": 0.5,
    }
    safer = dict(base, damage=0)
    assert _leaderboard_key(safer) < _leaderboard_key(base)


def test_confirmation_gate_is_fixed_and_fail_closed() -> None:
    config = _config()
    passing = {"v3_selector": {"break_rate": 0.005, "repairs": 2}}
    assert _confirmation_gate(passing, 0.011, config) is True
    assert _confirmation_gate(passing, 0.009, config) is False
    assert _confirmation_gate({"v3_selector": {"break_rate": None, "repairs": 2}}, 0.02, config) is False
