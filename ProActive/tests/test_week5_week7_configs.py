from __future__ import annotations

import yaml


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_week5_architecture_and_gate_are_predeclared() -> None:
    config = _load("configs/experiments/diag_bakeoff.yaml")
    assert config["metadata"]["approval_status"] in {"PENDING", "APPROVED"}
    assert config["seeds"] == [42, 43, 44]
    assert config["encoders"] == ["clean_mlp", "gru", "masked_slot_mlp", "deep_sets"]
    assert config["initial_budgets"] == [1, 2, 4]
    assert config["temporary_aps"]["budgets"] == [1, 2, 3, 4, 7]
    assert config["architecture"]["set_transformer"] == "disabled_until_gate"
    assert config["selection_gate"] == {
        "macro_f1_tolerance": 0.01,
        "invariant_tolerance": 0.000001,
        "substantial_drift_ratio": 0.10,
        "set_transformer_gap_trigger": 0.015,
    }
    assert config["raps_gate"] == {
        "average_set_size_trigger": 3.2,
        "singleton_rate_trigger": 0.35,
        "coverage_near_target_tolerance": 0.03,
        "role_if_triggered": "appendix_efficiency_ablation",
    }
    assert config["shortcut_gate"] == {
        "minimum_main_minus_identity_macro_f1": 0.02,
    }
    assert config["features"]["include_dataset_id"] is False
    assert config["features"]["include_model_id"] is False


def test_week6_objective_and_nonleaking_uncertainty_baseline_are_predeclared() -> None:
    config = _load("configs/experiments/policy_train.yaml")
    assert config["seeds"] == [42, 43, 44]
    assert config["eta_loss"] == 0.25
    assert config["policy"]["ranking_margin"] == 0.05
    assert config["policy"]["mse_weight"] == 0.25
    assert config["policy"]["action_ce_weight"] == 0.5
    assert config["baselines"]["uncertainty_greedy"] == "supervised_train_only_entropy_reduction_v1"
    assert config["active_signal_gate"] == {
        "minimum_relative_full_teacher_gain_over_clean": 0.10,
    }
    assert config["splits"]["forbidden_during_week6"] == ["cal", "test"]


def test_week7_calibration_and_permutation_protocol_are_predeclared() -> None:
    config = _load("configs/experiments/calibrate_aps.yaml")
    assert config["budgets"] == [1, 2, 3, 4, 7]
    assert config["coverages"] == [0.9, 0.95]
    assert config["splits"] == {"calibration": "cal", "locked_test": "test"}
    assert config["permutation"]["minimum_states"] == 2000
    assert config["permutation"]["maximum_permutations_per_state"] == 10
