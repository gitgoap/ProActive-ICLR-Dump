"""Unit, adversarial, and end-to-end CPU tests for the Week 4 substrate."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from proactive.audits.human_audit import LABELS, select_audit_keys
from proactive.audits.week4_validation import collect_artifact_rows, validate_week4_artifacts
from proactive.features.evidence_state import ProbeAction, ProbeObservation
from proactive.teacher.label_computation import DEFAULT_THRESHOLDS, compute_teacher_labels
from proactive.teacher.offline import (
    CORE_PROBES,
    build_label_record,
    build_state_records,
    sample_partial_subsets,
    stable_shard_id,
    thresholds_from_mapping,
    validate_resume_teacher_records,
)
from proactive.utils.io import write_jsonl


REVISION = "a" * 40
MODEL_ID = "test/model"


def _observation(name: str) -> ProbeObservation:
    semantic_match = 0.0 if name == "blank" else 1.0
    return ProbeObservation(
        probe_id=ProbeAction(name),
        raw_answer="yes",
        norm_answer="yes",
        flip=False,
        conf_shift=0.0,
        entropy_shift=0.0,
        margin_shift=0.0,
        exact_match=1.0,
        semantic_match=semantic_match,
        applicable=True,
        severity=8.0 if name == "blur" else None,
        latency_ms=1.0,
        prompt_hash="1" * 64,
        image_transform_hash="2" * 64,
        generation_config_hash="3" * 64,
        valid=True,
    )


def _teacher_record(
    instance_id: str = "pope_image_question", relation_applicable: bool = False
) -> dict:
    observations = {ProbeAction(name): _observation(name) for name in CORE_PROBES}
    if relation_applicable:
        observations[ProbeAction.RELATION] = _observation("relation")
    labels = compute_teacher_labels(
        observations,
        clean_answer_prob=0.8,
        clean_correct=True,
        relation_applicable=relation_applicable,
        swap_invariance=False if relation_applicable else None,
        thresholds=DEFAULT_THRESHOLDS,
    )
    return {
        "record_type": "teacher_cache",
        "instance_id": instance_id,
        "group_id": "sha256:0123456789abcdef",
        "dataset": "pope",
        "split": "train",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "image_path": "data/image.jpg",
        "prompt_text": "Is there a cat?",
        "gold_answer": "yes",
        "relation_applicable": relation_applicable,
        "valid": True,
        "clean": {
            "raw_answer": "yes",
            "norm_answer": "yes",
            "correct": 1,
            "answer_logprob": -0.2,
            "answer_prob": 0.8,
            "token_entropy_mean": 0.1,
            "token_margin_mean": 0.7,
            "answer_len_tokens": 1,
            "latency_ms": 1.0,
            "score_method": "generation_logits",
            "valid": True,
        },
        "probes": {
            action.value: observation.to_dict()
            for action, observation in observations.items()
        },
        "teacher_signature": {
            "V": labels.teacher_signature.V,
            "L": labels.teacher_signature.L,
            "A": labels.teacher_signature.A,
        },
        "teacher_bits": {
            "visual": int(labels.source_bits.visual),
            "language": int(labels.source_bits.language),
            "alignment": int(labels.source_bits.alignment),
        },
        "teacher_label6": labels.six_way_state.value,
        "benchmark_family": "random",
        "swap_invariance": False if relation_applicable else None,
    }


def test_stable_sharding_is_order_independent_and_total() -> None:
    assignments = [stable_shard_id(f"id-{index}", MODEL_ID, 4) for index in range(100)]
    reversed_assignments = {
        instance_id: stable_shard_id(instance_id, MODEL_ID, 4)
        for instance_id in reversed([f"id-{index}" for index in range(100)])
    }
    assert assignments == [reversed_assignments[f"id-{index}"] for index in range(100)]
    assert set(assignments) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="positive"):
        stable_shard_id("id", MODEL_ID, 0)


def test_resume_validation_rejects_duplicate_and_hash_drift() -> None:
    teacher = _teacher_record()
    shard_id = stable_shard_id(teacher["instance_id"], MODEL_ID, 4)
    teacher.update(
        {
            "source_manifest_sha256": "a" * 64,
            "frozen_probe_config_sha256": "b" * 64,
            "shard_id": shard_id,
            "num_shards": 4,
        }
    )
    completed = validate_resume_teacher_records(
        [teacher],
        model_id=MODEL_ID,
        selected_ids={teacher["instance_id"]},
        manifest_sha256="a" * 64,
        frozen_config_sha256="b" * 64,
        shard_id=shard_id,
        num_shards=4,
    )
    assert completed == {(MODEL_ID, teacher["instance_id"])}
    with pytest.raises(ValueError, match="Duplicate"):
        validate_resume_teacher_records(
            [teacher, copy.deepcopy(teacher)],
            MODEL_ID,
            {teacher["instance_id"]},
            "a" * 64,
            "b" * 64,
            shard_id,
            4,
        )
    drifted = copy.deepcopy(teacher)
    drifted["source_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Manifest hash drift"):
        validate_resume_teacher_records(
            [drifted],
            MODEL_ID,
            {teacher["instance_id"]},
            "a" * 64,
            "b" * 64,
            shard_id,
            4,
        )


def test_week4_config_contract_parses_frozen_thresholds() -> None:
    with open("configs/experiments/teacher_core.yaml", "r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    with open(experiment["frozen_probe_config"], "r", encoding="utf-8") as handle:
        frozen = yaml.safe_load(handle)
    thresholds = thresholds_from_mapping(frozen["label_thresholds"])
    assert experiment["state_sampling"]["random_subsets"] == 16
    assert experiment["human_audit"]["total_examples"] == 180
    assert len(experiment["core_model_keys"]) == 2
    assert experiment["metadata"]["approval_status"] == "APPROVED"
    assert experiment["compute_authorization"]["staged_max_examples"] == 100
    assert experiment["compute_authorization"]["full_core_approved"] is True
    assert thresholds.blank_conf_ratio_threshold == 0.8


def test_human_audit_sampling_is_deterministic_balanced_and_covered() -> None:
    datasets = ("hallusionbench", "pope", "vizwiz", "vsr")
    models = ("qwen", "gemma", "internvl")
    teachers = {}
    labels = {}
    for index in range(360):
        model_id = models[index % len(models)]
        key = (model_id, f"instance-{index:04d}")
        dataset = datasets[index % len(datasets)]
        label_name = LABELS[index % len(LABELS)]
        teachers[key] = {"instance_id": key[1], "model_id": key[0]}
        labels[key] = {
            "instance_id": key[1],
            "model_id": key[0],
            "dataset": dataset,
            "split": "train",
            "teacher_label6": label_name,
            "teacher_signature": {"V": 0.15, "L": 0.8, "A": 0.15},
        }
    kwargs = {
        "teachers": teachers,
        "labels": labels,
        "total": 180,
        "natural_count": 60,
        "target_per_label": 30,
        "seed": 42,
        "thresholds": {
            "visual_conf_threshold": 0.15,
            "blank_conf_ratio_threshold": 0.8,
            "grounding_conf_threshold": 0.15,
        },
        "allowed_splits": {"train", "val"},
    }
    selected, provenance = select_audit_keys(**kwargs)
    assert (selected, provenance) == select_audit_keys(**kwargs)
    assert len(selected) == len(set(selected)) == 180
    assert sum(value == "natural" for value in provenance.values()) == 60
    assert {labels[key]["teacher_label6"] for key in selected} == set(LABELS)
    assert {labels[key]["dataset"] for key in selected} == set(datasets)
    assert {labels[key]["model_id"] for key in selected} == set(models)


def test_labels_are_recomputed_deterministically_and_fail_closed() -> None:
    teacher = _teacher_record()
    first = build_label_record(teacher, DEFAULT_THRESHOLDS, "a" * 64)
    second = build_label_record(copy.deepcopy(teacher), DEFAULT_THRESHOLDS, "a" * 64)
    assert first == second
    assert first["teacher_label6"] == "no-failure"

    drifted = copy.deepcopy(teacher)
    drifted["teacher_bits"]["visual"] = 1
    with pytest.raises(ValueError, match="disagree"):
        build_label_record(drifted, DEFAULT_THRESHOLDS, "a" * 64)

    invalid = copy.deepcopy(teacher)
    invalid["probes"]["blur"]["valid"] = False
    with pytest.raises(ValueError, match="invalid"):
        build_label_record(invalid, DEFAULT_THRESHOLDS, "a" * 64)


def test_partial_state_sampling_contains_all_mandatory_sources() -> None:
    teacher = _teacher_record()
    subsets = sample_partial_subsets(teacher, seed=42, random_subset_count=16)
    source_tags = {tag for _, sources in subsets for tag in sources}
    assert "empty" in source_tags
    for probe in CORE_PROBES:
        assert f"singleton:{probe}" in source_tags
    assert len({tag for tag in source_tags if tag.startswith("random:")}) == 16
    assert all(set(subset).issubset(set(CORE_PROBES)) for subset, _ in subsets)
    assert subsets == sample_partial_subsets(teacher, seed=42, random_subset_count=16)


def test_relation_applicable_record_requires_exactly_seven_legal_probes() -> None:
    teacher = _teacher_record(dataset_instance_id := "vsr_image_question", relation_applicable=True)
    label = build_label_record(teacher, DEFAULT_THRESHOLDS, "a" * 64)
    assert len(teacher["probes"]) == 7
    assert label["instance_id"] == dataset_instance_id

    missing = copy.deepcopy(teacher)
    del missing["probes"]["relation"]
    with pytest.raises(ValueError, match="Probe set mismatch"):
        build_label_record(missing, DEFAULT_THRESHOLDS, "a" * 64)


def test_partial_states_never_serialize_unacquired_evidence() -> None:
    teacher = _teacher_record()
    label = build_label_record(teacher, DEFAULT_THRESHOLDS, "a" * 64)
    states = build_state_records(
        teacher,
        label,
        source_teacher_file_sha256="a" * 64,
        source_label_file_sha256="b" * 64,
    )
    assert states
    forbidden = {"dataset", "model_id", "gold_answer", "correct", "teacher_label6"}
    for state in states:
        learner = state["learner_input"]
        assert not (forbidden & set(learner))
        acquired = learner["acquired_probe_names"]
        visible = [row["probe_id"] for row in learner["acquired_observations"]]
        assert visible == acquired
        assert len(visible) == len(set(visible))
        assert all(name in acquired for name in visible)
        assert "raw_answer" not in str(learner)
        assert "norm_answer" not in str(learner)


def test_week4_end_to_end_validator_and_split_hash_adversary() -> None:
    teacher = _teacher_record()
    label = build_label_record(teacher, DEFAULT_THRESHOLDS, "a" * 64)
    states = build_state_records(
        teacher,
        label,
        source_teacher_file_sha256="a" * 64,
        source_label_file_sha256="b" * 64,
    )
    manifest = [
        {
            "instance_id": teacher["instance_id"],
            "group_id": teacher["group_id"],
            "dataset": teacher["dataset"],
            "split": teacher["split"],
        }
    ]
    report = validate_week4_artifacts(
        manifest_records=manifest,
        teacher_rows=[(teacher, Path("teacher.jsonl"), "a" * 64)],
        label_rows=[(label, Path("labels.jsonl"), "b" * 64)],
        state_rows=[(state, Path("states.jsonl"), "c" * 64) for state in states],
        required_model_ids={MODEL_ID},
        thresholds=DEFAULT_THRESHOLDS,
        max_sixway_fraction=1.0,
        min_bit_count_per_dataset_model=0,
        require_complete=True,
    )
    assert report["is_valid"], report["errors"]
    assert report["unique_probe_records"] == 6

    leaked_teacher = copy.deepcopy(teacher)
    leaked_teacher["split"] = "test"
    bad = validate_week4_artifacts(
        manifest_records=manifest,
        teacher_rows=[(leaked_teacher, Path("teacher.jsonl"), "a" * 64)],
        label_rows=[(label, Path("labels.jsonl"), "b" * 64)],
        state_rows=[(state, Path("states.jsonl"), "c" * 64) for state in states],
        required_model_ids={MODEL_ID},
        thresholds=DEFAULT_THRESHOLDS,
        max_sixway_fraction=1.0,
        min_bit_count_per_dataset_model=0,
        require_complete=True,
    )
    assert not bad["is_valid"]
    assert any("split mismatch" in error for error in bad["errors"])


def test_duplicate_teacher_keys_are_rejected() -> None:
    teacher = _teacher_record()
    label = build_label_record(teacher, DEFAULT_THRESHOLDS, "a" * 64)
    states = build_state_records(
        teacher,
        label,
        source_teacher_file_sha256="a" * 64,
        source_label_file_sha256="b" * 64,
    )
    report = validate_week4_artifacts(
        manifest_records=[
            {
                "instance_id": teacher["instance_id"],
                "group_id": teacher["group_id"],
                "dataset": teacher["dataset"],
                "split": teacher["split"],
            }
        ],
        teacher_rows=[
            (teacher, Path("one.jsonl"), "a" * 64),
            (copy.deepcopy(teacher), Path("two.jsonl"), "d" * 64),
        ],
        label_rows=[(label, Path("labels.jsonl"), "b" * 64)],
        state_rows=[(state, Path("states.jsonl"), "c" * 64) for state in states],
        required_model_ids={MODEL_ID},
        thresholds=DEFAULT_THRESHOLDS,
        max_sixway_fraction=1.0,
        min_bit_count_per_dataset_model=0,
        require_complete=True,
    )
    assert not report["is_valid"]
    assert any("Duplicate model-instance teacher key" in error for error in report["errors"])


def test_teacher_collection_excludes_failure_ledgers(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher_model_shard00.jsonl"
    failure_path = tmp_path / "teacher_model_shard00.failures.jsonl"
    write_jsonl([_teacher_record()], teacher_path)
    write_jsonl([{"record_type": "teacher_failure"}], failure_path)
    rows, manifest = collect_artifact_rows(tmp_path, "teacher_")
    assert len(rows) == 1
    assert [Path(item["path"]).name for item in manifest] == [teacher_path.name]
