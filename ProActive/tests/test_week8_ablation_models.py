import torch
import copy
import sys

from proactive.networks.diagnostic import (
    IndependentSourceDiagnosticModel,
    build_diagnostic_model,
)
from proactive.train.state_data import PROBE_NUMERIC_FEATURES, PROBE_ORDER
from proactive.train.ablations import (
    apply_state_record_ablation,
    apply_teacher_record_ablation,
    validate_week8_ablation_authorization,
)
from scripts.build_voi_targets import _jobs_for_records
from scripts.validate_ablation_rebuild_equivalence import validate_equivalence
from scripts.aggregate_week8_ablations import parse_args as parse_ablation_aggregate_args
from tests._week5_fixtures import make_state, make_teacher


def _input(batch: int = 2):
    return {
        "clean_features": torch.zeros(batch, 4),
        "probe_numeric": torch.zeros(
            batch, len(PROBE_ORDER), len(PROBE_NUMERIC_FEATURES)
        ),
        "acquired_mask": torch.zeros(batch, len(PROBE_ORDER), dtype=torch.bool),
        "sequence_indices": torch.full(
            (batch, len(PROBE_ORDER)), -1, dtype=torch.long
        ),
        "remaining_budget": torch.tensor([1, 3][:batch], dtype=torch.long),
        "max_budget": torch.full((batch,), 7, dtype=torch.long),
        "action_mask": torch.ones(batch, len(PROBE_ORDER) + 1, dtype=torch.bool),
    }


def test_zero_budget_embedding_is_a_real_supported_ablation() -> None:
    model = build_diagnostic_model(
        "deep_sets",
        {"state_dim": 16, "budget_embedding_dim": 0, "max_budget": 7},
    )
    output = model(_input())
    assert output.bit_logits.shape == (2, 3)
    assert model.encoder.budget_embedding.embedding is None


def test_independent_source_ablation_uses_distinct_encoders() -> None:
    model = build_diagnostic_model(
        "deep_sets",
        {
            "state_dim": 16,
            "budget_embedding_dim": 0,
            "max_budget": 7,
            "multi_task_mode": "independent_source_encoders",
        },
    )
    assert isinstance(model, IndependentSourceDiagnosticModel)
    assert len({id(encoder) for encoder in model.source_encoders}) == 3
    output = model(_input())
    assert output.bit_logits.shape == (2, 3)
    assert output.six_way_logits.shape == (2, 6)
    assert output.signature.shape == (2, 3)


def test_unknown_multitask_mode_fails_closed() -> None:
    try:
        build_diagnostic_model("deep_sets", {"multi_task_mode": "not-real"})
    except ValueError as exc:
        assert "multi_task_mode" in str(exc)
    else:
        raise AssertionError("Unknown multi-task mode did not fail closed")


def test_feature_ablation_is_consistent_across_state_and_future_teacher_evidence() -> None:
    state = make_state(acquired=("blank",))
    teacher = make_teacher(state)
    original_state = copy.deepcopy(state)
    original_teacher = copy.deepcopy(teacher)
    transformed = apply_state_record_ablation(state, "no_semantic_match")
    transformed_teacher = apply_teacher_record_ablation(teacher, "no_semantic_match")
    assert transformed is not None
    assert transformed["state_id"].endswith("|ablation")
    assert transformed["learner_input"]["acquired_observations"][0]["semantic_match"] == 0
    assert all(value["semantic_match"] == 0 for value in transformed_teacher["probes"].values())
    assert state == original_state
    assert teacher == original_teacher


def test_feature_ablation_suffix_is_idempotent() -> None:
    state = apply_state_record_ablation(make_state(acquired=()), "no_answer_flip")
    assert state is not None
    repeated = apply_state_record_ablation(state, "no_answer_flip")
    assert repeated is not None
    assert repeated["state_id"].count("|ablation") == 1


def test_no_relation_ablation_filters_acquired_states_and_disables_future_relation() -> None:
    acquired = make_state(acquired=("relation",), relation_applicable=True)
    assert apply_state_record_ablation(acquired, "no_relation_probe") is None
    empty = make_state(acquired=(), relation_applicable=True)
    transformed = apply_state_record_ablation(empty, "no_relation_probe")
    assert transformed is not None
    assert transformed["learner_input"]["action_mask"]["relation"] == 0


def test_voi_jobs_use_same_ablation_state_ids_as_tensor_manifest() -> None:
    state = make_state(acquired=("blank",), relation_applicable=True)
    teacher = {
        (state["metadata"]["model_id"], state["metadata"]["instance_id"]): make_teacher(state)
    }
    _, jobs = _jobs_for_records(
        [state],
        teacher=teacher,
        budgets=[2],
        ablation_id="no_confidence_shift",
    )
    assert len(jobs) == 1
    assert jobs[0]["state"]["state_id"] == f"{state['state_id']}|ablation"
    assert jobs[0]["state"]["learner_input"]["acquired_observations"][0]["conf_shift"] == 0


def test_no_relation_voi_jobs_never_offer_relation() -> None:
    state = make_state(acquired=(), relation_applicable=True)
    teacher = {
        (state["metadata"]["model_id"], state["metadata"]["instance_id"]): make_teacher(state)
    }
    _, jobs = _jobs_for_records(
        [state], teacher=teacher, budgets=[1], ablation_id="no_relation_probe"
    )
    assert len(jobs) == 1
    assert "relation" not in jobs[0]["action_indices"]


def test_week8_ablation_authorization_is_exact_and_fail_closed() -> None:
    authorization = {
        "approved": True,
        "approved_on": "2026-09-11",
        "max_physical_gpus": 2,
        "seed": 42,
        "evidence_split": "val",
        "approved_combined_gpu_hours": 8.0,
        "training_timeout_minutes": 45,
        "frontier_timeout_minutes": 20,
        "heldout_shift_tuning_allowed": False,
    }
    config = {
        "ablation_execution": {
            "split": "val",
            "locked_test_reopen_allowed": False,
            "compute_authorization": authorization,
        }
    }
    assert validate_week8_ablation_authorization(config) == authorization
    drifted = copy.deepcopy(config)
    drifted["ablation_execution"]["compute_authorization"]["evidence_split"] = "test"
    try:
        validate_week8_ablation_authorization(drifted)
    except ValueError as exc:
        assert "evidence_split" in str(exc)
    else:
        raise AssertionError("Unsafe ablation split drift was accepted")


def test_rebuilt_ablation_requires_exact_history_and_scientific_aps() -> None:
    def row(epoch: int, source_f1: float) -> dict:
        return {
            "epoch": epoch,
            "train_loss": {"total": 1.0 - source_f1},
            "validation": {
                "source_bit_macro_f1": source_f1,
                "six_way_macro_f1": 0.5,
            },
        }

    archived = [
        row(0, 0.5),
        row(1, 0.6),
        row(2, 0.59),
        row(3, 0.58),
        row(4, 0.57),
        row(5, 0.56),
        row(6, 0.55),
        row(7, 0.7),
    ]
    rebuilt = archived[:7]
    reference_aps = {
        "checkpoint_path": "old.pt",
        "checkpoint_sha256": "a" * 64,
        "report_sha256": "b" * 64,
        "thresholds": {"1": {"0.9": {"threshold": 0.75}}},
    }
    rebuilt_aps = {
        **reference_aps,
        "checkpoint_path": "new.pt",
        "checkpoint_sha256": "c" * 64,
        "report_sha256": "d" * 64,
    }
    result = validate_equivalence(
        rebuilt_history=rebuilt,
        archived_history=archived,
        reference_aps=reference_aps,
        rebuilt_aps=rebuilt_aps,
        patience=5,
    )
    assert result["stopping_epoch"] == 6
    assert result["epoch_count"] == 7

    changed = copy.deepcopy(rebuilt_aps)
    changed["thresholds"]["1"]["0.9"]["threshold"] = 0.76
    try:
        validate_equivalence(
            rebuilt_history=rebuilt,
            archived_history=archived,
            reference_aps=reference_aps,
            rebuilt_aps=changed,
            patience=5,
        )
    except ValueError as exc:
        assert "APS thresholds" in str(exc)
    else:
        raise AssertionError("Scientifically different rebuilt APS was accepted")


def test_ablation_aggregator_accumulates_repeated_evidence_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_week8_ablations.py",
            "--evidence",
            "no_stop_action=stop.json",
            "--evidence",
            "no_voi_training=voi.json",
        ],
    )
    args = parse_ablation_aggregate_args()
    assert args.evidence == [
        "no_stop_action=stop.json",
        "no_voi_training=voi.json",
    ]
