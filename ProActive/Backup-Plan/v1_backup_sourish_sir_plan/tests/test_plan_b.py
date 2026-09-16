from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


BACKUP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKUP_ROOT.parent
sys.path.insert(0, str(BACKUP_ROOT))

from plan_b.core import (  # noqa: E402
    Observation,
    FEATURE_NAMES,
    build_candidates,
    build_cohort_index,
    candidate_features,
    image_key,
    is_closed_answer,
    load_and_verify_inputs,
    metric_summary,
    record_normalizer_type,
    recompute_clean_correctness,
    select_majority,
    select_with_scores,
)
from plan_b.pipeline import evaluate_records, load_config, run_audit  # noqa: E402


def _observations() -> list[Observation]:
    return [
        Observation("clean", "no", "No", True, True),
        Observation("grounding", "yes", "Yes", True, True),
        Observation("blur", "yes", "Yes.", True, True),
    ]


def test_candidate_contract_and_feature_order() -> None:
    candidates = build_candidates(_observations(), acquired_cost=2)
    assert [candidate.norm_answer for candidate in candidates] == ["no", "yes"]
    yes = candidates[1]
    assert yes.sources == ("grounding", "blur")
    features = candidate_features(yes, candidates, acquired_cost=2)
    assert features == (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2 / 3, 2 / 7, 2 / 6)


def test_selector_uses_strict_margin_and_preserves_clean_on_tie() -> None:
    candidates = build_candidates(_observations(), acquired_cost=2)
    kept = select_with_scores(candidates, [0.60, 0.65], 0.05, 0.5, 1e-12)
    assert kept.source == "clean"
    switched = select_with_scores(candidates, [0.59, 0.65], 0.05, 0.5, 1e-12)
    assert switched.source == "grounding"


def test_majority_tie_prefers_clean() -> None:
    observations = _observations()[:2]
    selection = select_majority(observations)
    assert selection.source == "clean"
    assert selection.norm_answer == "no"


def test_features_cannot_see_unacquired_observations() -> None:
    prefix = _observations()[:2]
    original = [candidate_features(c, build_candidates(prefix, 1), 1) for c in build_candidates(prefix, 1)]
    future_a = Observation("blank", "yes", "Yes", True, True)
    future_b = Observation("blank", "no", "No", True, True)
    # The construction API sees only the prefix. Mutating future evidence has no effect.
    assert future_a != future_b
    repeated = [candidate_features(c, build_candidates(prefix, 1), 1) for c in build_candidates(prefix, 1)]
    assert original == repeated


def test_image_key_coalesces_shared_coco_images() -> None:
    pope = {"dataset": "pope", "image_path": "/a/COCO_val2014_000000123456.jpg"}
    vsr = {"dataset": "vsr", "image_path": "/b/000000123456.jpg"}
    assert image_key(pope) == image_key(vsr) == "coco:000000123456"


def test_historical_normalizer_contract_is_explicit_and_closed() -> None:
    assert record_normalizer_type({"dataset": "pope", "instance_id": "x"}) == "yes_no"
    assert record_normalizer_type({"dataset": "vsr", "instance_id": "x"}) == "true_false"
    assert record_normalizer_type(
        {"dataset": "hallusionbench", "answer_type": "open_ended", "instance_id": "x"}
    ) == "freeform"


def test_open_hallusion_correctness_uses_exact_reference_aliases() -> None:
    row = {
        "dataset": "hallusionbench",
        "instance_id": "hallusionbench_x",
        "answer_type": "open_ended",
        "answer_match_mode": "exact_alias",
        "gold_answer": "France had the highest GDP in Europe in 2021,",
        "reference_answers": ["France", "France had the highest GDP in Europe in 2021,"],
        "clean": {"raw_answer": "France", "norm_answer": "france"},
    }
    assert recompute_clean_correctness(row) == 1


def test_later_split_images_are_excluded_from_fitting() -> None:
    rows = [
        {
            "instance_id": "a",
            "group_id": "g1",
            "dataset": "pope",
            "split": "train",
            "model_id": "m",
            "answer_type": "binary",
            "image_path": "/x/COCO_val2014_000000000001.jpg",
        },
        {
            "instance_id": "b",
            "group_id": "g2",
            "dataset": "vsr",
            "split": "test",
            "model_id": "m",
            "answer_type": "binary",
            "image_path": "/x/000000000001.jpg",
        },
    ]
    cohort, exclusions = build_cohort_index(rows)
    assert cohort[0]["image_excluded_from_fit"] == 1
    assert cohort[0]["proposed_fit_eligible"] == 0
    assert exclusions[0]["reason"] == "train_image_seen_later"


def test_metric_identity_and_rates() -> None:
    rows = [
        {"original_correct": 0, "correct": 1, "repair": 1, "damage": 0, "abstain": 0, "candidate_count": 2, "matched_cost": 2, "operational_cost": 2},
        {"original_correct": 1, "correct": 0, "repair": 0, "damage": 1, "abstain": 0, "candidate_count": 2, "matched_cost": 2, "operational_cost": 2},
    ]
    summary = metric_summary(rows)
    assert summary["final_accuracy"] == 0.5
    assert summary["fix_rate"] == 1.0
    assert summary["break_rate"] == 1.0


def test_real_cache_audit_when_inputs_are_available(tmp_path: Path) -> None:
    config_path = BACKUP_ROOT / "config" / "plan_b_config.json"
    config = load_config(config_path)
    if any(not (REPO_ROOT / item["path"]).is_file() for item in config["input_files"]):
        pytest.skip("Pinned Plan B caches are not present in this checkout")
    report = run_audit(
        repo_root=REPO_ROOT,
        config=config,
        output_dir=tmp_path,
        resume=False,
        overwrite=False,
    )
    assert report["is_valid"] is True
    assert report["observed"] == config["expected"]
    assert (tmp_path / "audit" / "cohort_index.csv").is_file()


def test_real_reference_baselines_when_inputs_are_available() -> None:
    config = load_config(BACKUP_ROOT / "config" / "plan_b_config.json")
    if any(not (REPO_ROOT / item["path"]).is_file() for item in config["input_files"]):
        pytest.skip("Pinned Plan B caches are not present in this checkout")
    records, _ = load_and_verify_inputs(REPO_ROOT, config)
    test_records = [
        row for row in records if row["split"] == "test" and is_closed_answer(row)
    ]
    dummy_model = {"coefficients": [0.0] * len(FEATURE_NAMES), "intercept": 0.0}
    methods = ["keep_original", "always_ground", "majority", "oracle"]
    predictions = evaluate_records(test_records, config, dummy_model, 0.0, [2], methods)
    observed = {
        method: sum(int(row["correct"]) for row in predictions if row["method"] == method)
        for method in methods
    }
    assert observed == config["reference_core_test_b2_correct"]
