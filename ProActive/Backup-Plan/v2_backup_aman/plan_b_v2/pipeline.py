"""Audited Plan B V2 train/validation and separately locked evaluation pipeline."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from plan_b.core import (
    ABSTAIN,
    PlanBError,
    Selection,
    acquired_observations,
    atomic_write_csv,
    atomic_write_json,
    build_candidates,
    build_cohort_index,
    canonical_json_sha256,
    correctness,
    image_key,
    load_and_verify_inputs,
    load_json,
    metric_summary,
    observation_from_row,
    operational_cost,
    record_gold_norm,
    recompute_clean_correctness,
    select_ground,
    select_keep,
    select_majority,
    select_oracle,
    select_supported_ground,
    sha256_file,
    verify_self_hash,
    with_self_hash,
)
from plan_b.pipeline import _six_cell_macro, _slice_metrics, run_audit

from .features import feature_names, feature_vector
from .models import (
    ModelSpec,
    dependency_report,
    fit_estimator,
    load_estimator,
    model_specs,
    positive_probability,
    save_estimator,
)


PREDICTION_FIELDS = (
    "split", "dataset", "model_id", "instance_id", "group_id", "image_key",
    "budget", "method", "selected_source", "selected_norm_answer", "selected_raw_answer",
    "correct", "original_correct", "repair", "damage", "abstain", "correction_abstained",
    "matched_cost", "operational_cost", "candidate_count", "alternative_count",
    "selected_repair_probability", "selected_damage_probability", "selected_safe_score",
)

LEADERBOARD_FIELDS = (
    "rank", "feature_set", "model_name", "repair_probability_minimum", "damage_penalty",
    "safe_score_threshold", "six_cell_macro_accuracy", "six_cell_net_repair_gain",
    "pooled_original_accuracy", "pooled_final_accuracy", "repairs", "damage", "fix_rate",
    "break_rate", "correction_abstentions", "eligible_under_break_constraint", "gate_passed",
)


def load_v2_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != "plan_b_v2_config_v1":
        raise PlanBError("Unsupported Plan B V2 config schema")
    if config.get("acquisition_order") != ["grounding", "blur", "crop", "brightness", "noise", "blank"]:
        raise PlanBError("Plan B V2 acquisition order drift")
    constraints = config.get("scientific_constraints", {})
    required_true = (
        "post_hoc_after_v1_validation", "validation_only_model_and_threshold_selection",
        "locked_test_requires_validation_gate", "cached_confidence_is_unverified_proxy",
        "no_new_mllm_calls", "no_dataset_or_model_identity_features",
    )
    if any(constraints.get(key) is not True for key in required_true):
        raise PlanBError("V2 scientific constraints are incomplete")
    if constraints.get("calibration_split_access") is not False or constraints.get("test_or_shift_tuning") is not False:
        raise PlanBError("V2 must forbid calibration/test/shift tuning")
    if not config.get("feature_sets") or not config.get("model_ablations"):
        raise PlanBError("V2 feature/model ablation grid is empty")
    for name in config["feature_sets"]:
        feature_names(str(name))
    model_specs(config)
    search = config.get("safe_correction_search", {})
    if search.get("abstention_action") != "KEEP_ORIGINAL":
        raise PlanBError("V2 safety contract requires correction abstention to keep the original answer")
    for key in ("repair_probability_minimums", "damage_penalties", "safe_score_thresholds"):
        values = search.get(key)
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in values
        ):
            raise PlanBError(f"Invalid V2 search grid: {key}")
    if any(not 0.0 <= float(value) <= 1.0 for value in search["repair_probability_minimums"]):
        raise PlanBError("Repair-probability thresholds must lie in [0, 1]")
    if any(float(value) < 0.0 for value in search["damage_penalties"]):
        raise PlanBError("Damage penalties must be non-negative")
    maximum_break = float(config.get("gates", {}).get("maximum_validation_break_rate", -1.0))
    if not 0.0 <= maximum_break <= 0.01:
        raise PlanBError("V2 validation BreakRate ceiling may not exceed 1%")
    return config


def load_v1_config(repo_root: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = (repo_root / str(config["v1_backbone_config"])).resolve()
    if not path.is_file():
        raise PlanBError(f"Missing V1 backbone config: {path}")
    actual = sha256_file(path)
    if actual != config["v1_backbone_config_sha256"]:
        raise PlanBError(f"V1 backbone config hash mismatch: {actual}")
    payload = load_json(path)
    if payload.get("schema_version") != "plan_b_config_v1":
        raise PlanBError("Unsupported V1 backbone schema")
    return path, payload


def _stage_guard(path: Path, *, resume: bool, overwrite: bool) -> dict[str, Any] | None:
    if not path.exists() or overwrite:
        return None
    if not resume:
        raise PlanBError(f"Output exists; use --resume or --overwrite: {path}")
    payload = load_json(path)
    verify_self_hash(payload)
    return payload


def _artifact_inventory(output_dir: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    output = []
    for path in paths:
        if not path.is_file():
            raise PlanBError(f"Expected output artifact is missing: {path}")
        output.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return output


def _cohort_lookup(cohort: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["instance_id"]), str(row["model_id"])): row for row in cohort}


def _fit_records(
    records: Sequence[Mapping[str, Any]], cohort: Sequence[Mapping[str, Any]], split: str
) -> list[Mapping[str, Any]]:
    lookup = _cohort_lookup(cohort)
    result = []
    for row in records:
        entry = lookup[(str(row["instance_id"]), str(row["model_id"]))]
        if entry["split"] == split and int(entry["proposed_fit_eligible"]) == 1:
            result.append(row)
    return result


def _alternative_candidates(row: Mapping[str, Any], budget: int, config: Mapping[str, Any]):
    observations, cost = acquired_observations(row, budget, config["acquisition_order"])
    candidates = build_candidates(observations, cost)
    clean = observation_from_row(row, "clean")
    alternatives = [candidate for candidate in candidates if not clean.usable or candidate.norm_answer != clean.norm_answer]
    return observations, cost, candidates, alternatives


def _training_matrix(
    records: Sequence[Mapping[str, Any]], feature_set: str, config: Mapping[str, Any]
) -> tuple[list[list[float]], list[int], list[int], list[float], dict[str, Any]]:
    group_sizes = Counter(str(row["group_id"]) for row in records)
    features: list[list[float]] = []
    repair_labels: list[int] = []
    damage_labels: list[int] = []
    raw_weights: list[float] = []
    no_alternative_prefixes = 0
    for row in records:
        gold = record_gold_norm(row)
        original_correct = recompute_clean_correctness(row)
        for budget_value in config["training_budgets"]:
            budget = int(budget_value)
            observations, cost, candidates, alternatives = _alternative_candidates(row, budget, config)
            if not alternatives:
                no_alternative_prefixes += 1
                continue
            denominator = group_sizes[str(row["group_id"])] * len(config["training_budgets"]) * len(alternatives)
            if denominator <= 0:
                raise PlanBError("Invalid V2 sample-weight denominator")
            for candidate in alternatives:
                values = feature_vector(feature_set, row, candidate, candidates, observations, cost)
                candidate_correct = int(candidate.norm_answer == gold)
                features.append(list(values))
                repair_labels.append(int(not original_correct and candidate_correct))
                damage_labels.append(int(original_correct and not candidate_correct))
                raw_weights.append(1.0 / denominator)
    if not features or set(repair_labels) != {0, 1} or set(damage_labels) != {0, 1}:
        raise PlanBError(f"V2 {feature_set} repair/damage targets do not contain both classes")
    mean_weight = sum(raw_weights) / len(raw_weights)
    weights = [value / mean_weight for value in raw_weights]
    return features, repair_labels, damage_labels, weights, {
        "feature_set": feature_set,
        "feature_count": len(feature_names(feature_set)),
        "eligible_model_examples": len(records),
        "alternative_candidate_rows": len(features),
        "repair_positive_rows": sum(repair_labels),
        "damage_positive_rows": sum(damage_labels),
        "no_alternative_prefixes": no_alternative_prefixes,
        "mean_normalized_sample_weight": sum(weights) / len(weights),
    }


def _score_records(
    records: Sequence[Mapping[str, Any]],
    feature_set: str,
    repair_model: Any,
    damage_model: Any,
    budget: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scored = []
    for row in records:
        observations, cost, candidates, alternatives = _alternative_candidates(row, budget, config)
        features = [feature_vector(feature_set, row, candidate, candidates, observations, cost) for candidate in alternatives]
        repair = positive_probability(repair_model, features)
        damage = positive_probability(damage_model, features)
        scored.append(
            {
                "row": row,
                "observations": observations,
                "cost": cost,
                "candidates": candidates,
                "alternatives": alternatives,
                "repair_probabilities": repair,
                "damage_probabilities": damage,
            }
        )
    return scored


def _select_safe_correction(
    item: Mapping[str, Any], *, repair_minimum: float, damage_penalty: float, threshold: float, tolerance: float
) -> tuple[Selection, float | None, float | None, float | None, bool]:
    alternatives = item["alternatives"]
    repair = item["repair_probabilities"]
    damage = item["damage_probabilities"]
    if len(alternatives) != len(repair) or len(alternatives) != len(damage):
        raise PlanBError("V2 candidate/probability length mismatch")
    eligible = []
    for index, candidate in enumerate(alternatives):
        repair_probability = float(repair[index])
        damage_probability = float(damage[index])
        safe_score = repair_probability - damage_penalty * damage_probability
        if repair_probability + tolerance >= repair_minimum and safe_score + tolerance >= threshold:
            eligible.append((index, safe_score, repair_probability, damage_probability, candidate))
    if not eligible:
        return select_keep(item["observations"]), None, None, None, True
    eligible.sort(key=lambda value: (-value[1], -value[2], value[3], value[4].earliest_rank, value[4].norm_answer))
    _, safe_score, repair_probability, damage_probability, candidate = eligible[0]
    selection = Selection(candidate.sources[0], candidate.norm_answer, candidate.raw_answer)
    return selection, repair_probability, damage_probability, safe_score, False


def _prediction_row(
    record: Mapping[str, Any], observations: Sequence[Any], candidates: Sequence[Any], alternatives: Sequence[Any],
    cost: int, budget: int, method: str, selection: Selection, *, correction_abstained: bool,
    repair_probability: float | None = None, damage_probability: float | None = None,
    safe_score: float | None = None,
) -> dict[str, Any]:
    gold = record_gold_norm(record)
    original_correct = recompute_clean_correctness(record)
    final_correct = correctness(selection, gold)
    return {
        "split": record["split"],
        "dataset": record["dataset"],
        "model_id": record["model_id"],
        "instance_id": record["instance_id"],
        "group_id": record["group_id"],
        "image_key": image_key(record),
        "budget": budget,
        "method": method,
        "selected_source": selection.source,
        "selected_norm_answer": selection.norm_answer,
        "selected_raw_answer": selection.raw_answer,
        "correct": final_correct,
        "original_correct": original_correct,
        "repair": int(not original_correct and final_correct),
        "damage": int(original_correct and not final_correct),
        "abstain": int(selection.norm_answer == ABSTAIN),
        "correction_abstained": int(correction_abstained),
        "matched_cost": cost,
        "operational_cost": operational_cost(method, observations, cost),
        "candidate_count": len(candidates),
        "alternative_count": len(alternatives),
        "selected_repair_probability": repair_probability,
        "selected_damage_probability": damage_probability,
        "selected_safe_score": safe_score,
    }


def _baseline_predictions(
    records: Sequence[Mapping[str, Any]], budgets: Sequence[int], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    methods = {
        "keep_original": select_keep,
        "always_ground": select_ground,
        "majority": select_majority,
        "supported_ground": select_supported_ground,
    }
    for record in records:
        gold = record_gold_norm(record)
        for budget in budgets:
            observations, cost, candidates, alternatives = _alternative_candidates(record, int(budget), config)
            for name, function in methods.items():
                selection = function(observations)
                output.append(
                    _prediction_row(
                        record, observations, candidates, alternatives, cost, int(budget), name, selection,
                        correction_abstained=name == "keep_original",
                    )
                )
            oracle = select_oracle(observations, gold)
            output.append(
                _prediction_row(
                    record, observations, candidates, alternatives, cost, int(budget), "oracle", oracle,
                    correction_abstained=False,
                )
            )
    return output


def _selector_predictions(
    scored: Sequence[Mapping[str, Any]], *, budget: int, method: str, repair_minimum: float,
    damage_penalty: float, threshold: float, tolerance: float,
) -> list[dict[str, Any]]:
    output = []
    for item in scored:
        selection, repair, damage, safe_score, abstained = _select_safe_correction(
            item,
            repair_minimum=repair_minimum,
            damage_penalty=damage_penalty,
            threshold=threshold,
            tolerance=tolerance,
        )
        output.append(
            _prediction_row(
                item["row"], item["observations"], item["candidates"], item["alternatives"],
                item["cost"], budget, method, selection, correction_abstained=abstained,
                repair_probability=repair, damage_probability=damage, safe_score=safe_score,
            )
        )
    return output


def _pooled(predictions: Sequence[Mapping[str, Any]], method: str, budget: int) -> dict[str, Any]:
    rows = [row for row in predictions if row["method"] == method and int(row["budget"]) == budget]
    return metric_summary(rows)


def _leaderboard_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["six_cell_net_repair_gain"]),
        -int(row["repairs"]),
        int(row["damage"]),
        -float(row["repair_probability_minimum"]),
        -float(row["safe_score_threshold"]),
        -float(row["damage_penalty"]),
        str(row["feature_set"]),
        str(row["model_name"]),
    )


def _bootstrap_delta(
    predictions: Sequence[Mapping[str, Any]], split: str, method: str, budget: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanBError("V2 bootstrap requires NumPy") from exc
    relevant = [row for row in predictions if row["split"] == split and int(row["budget"]) == budget]
    selected = {(str(row["instance_id"]), str(row["model_id"])): row for row in relevant if row["method"] == method}
    kept = {(str(row["instance_id"]), str(row["model_id"])): row for row in relevant if row["method"] == "keep_original"}
    if selected.keys() != kept.keys() or not selected:
        raise PlanBError(f"Unpaired V2 bootstrap rows for {split}")
    clusters: dict[str, list[float]] = defaultdict(list)
    for key, row in selected.items():
        clusters[str(row["image_key"])].append(float(row["correct"]) - float(kept[key]["correct"]))
    names = sorted(clusters)
    values = [sum(clusters[name]) for name in names]
    counts = [len(clusters[name]) for name in names]
    observed = sum(values) / sum(counts)
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    draws = []
    for _ in range(int(config["statistics"]["bootstrap_resamples"])):
        indexes = rng.integers(0, len(names), size=len(names))
        numerator = sum(values[int(index)] for index in indexes)
        denominator = sum(counts[int(index)] for index in indexes)
        draws.append(numerator / denominator)
    alpha = (1.0 - float(config["statistics"]["confidence_level"])) / 2.0
    return {
        "split": split,
        "method": method,
        "comparator": "keep_original",
        "budget": budget,
        "model_example_pairs": len(selected),
        "image_clusters": len(names),
        "pooled_accuracy_delta": observed,
        "cluster_bootstrap_ci": [float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))],
    }


def run_prepare(
    *, repo_root: Path, config: Mapping[str, Any], output_dir: Path, resume: bool, overwrite: bool
) -> dict[str, Any]:
    _, v1_config = load_v1_config(repo_root, config)
    dependencies = dependency_report(model_specs(config))
    if dependencies["missing"]:
        raise PlanBError("Missing V2 dependencies: " + ", ".join(dependencies["missing"]))
    audit = run_audit(
        repo_root=repo_root, config=v1_config, output_dir=output_dir,
        resume=resume, overwrite=overwrite,
    )
    freeze_path = output_dir / "training" / "freeze_manifest.json"
    existing = _stage_guard(freeze_path, resume=resume, overwrite=overwrite)
    config_hash = canonical_json_sha256(config)
    if existing is not None:
        if existing.get("config_sha256") != config_hash or existing.get("v1_audit_report_sha256") != audit["report_sha256"]:
            raise PlanBError("Unsafe V2 preparation resume refused: provenance drift")
        for artifact in existing.get("training_artifacts", []):
            path = output_dir / artifact["path"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise PlanBError(f"Unsafe V2 resume refused: output drift for {artifact['path']}")
        return existing

    records, inventory = load_and_verify_inputs(repo_root, v1_config)
    cohort, _ = build_cohort_index(records)
    train_records = _fit_records(records, cohort, "train")
    validation_records = _fit_records(records, cohort, "val")
    training_dir = output_dir / "training"
    model_dir = training_dir / "models"
    artifacts: list[Path] = []
    training_summaries = []
    fitted: dict[tuple[str, str], tuple[Any, Any, Path, Path]] = {}
    for feature_set in config["feature_sets"]:
        features, repair_labels, damage_labels, weights, summary = _training_matrix(
            train_records, str(feature_set), config
        )
        training_summaries.append(summary)
        for spec in model_specs(config):
            repair_model = fit_estimator(spec, features, repair_labels, weights, seed=int(config["seed"]))
            damage_model = fit_estimator(spec, features, damage_labels, weights, seed=int(config["seed"]))
            stem = f"{feature_set}__{spec.name}"
            repair_path = model_dir / f"{stem}__repair.joblib"
            damage_path = model_dir / f"{stem}__damage.joblib"
            save_estimator(repair_model, repair_path)
            save_estimator(damage_model, damage_path)
            artifacts.extend([repair_path, damage_path])
            fitted[(str(feature_set), spec.name)] = (repair_model, damage_model, repair_path, damage_path)

    primary_budget = int(config["primary_budget"])
    baseline_predictions = _baseline_predictions(validation_records, [primary_budget], config)
    keep_macro = _six_cell_macro(baseline_predictions, "keep_original", primary_budget)
    search = config["safe_correction_search"]
    tolerance = float(search["tie_tolerance"])
    maximum_break = float(config["gates"]["maximum_validation_break_rate"])
    minimum_gain = float(config["gates"]["minimum_validation_six_cell_net_repair_gain"])
    minimum_repairs = int(config["gates"]["minimum_validation_repairs"])
    leaderboard = []
    prediction_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (feature_set, model_name), (repair_model, damage_model, _, _) in fitted.items():
        scored = _score_records(
            validation_records, feature_set, repair_model, damage_model, primary_budget, config
        )
        prediction_cache[(feature_set, model_name)] = scored
        for repair_minimum in search["repair_probability_minimums"]:
            for damage_penalty in search["damage_penalties"]:
                for threshold in search["safe_score_thresholds"]:
                    predictions = _selector_predictions(
                        scored, budget=primary_budget, method="v2_selector",
                        repair_minimum=float(repair_minimum), damage_penalty=float(damage_penalty),
                        threshold=float(threshold), tolerance=tolerance,
                    )
                    macro = _six_cell_macro(predictions, "v2_selector", primary_budget)
                    pooled = _pooled(predictions, "v2_selector", primary_budget)
                    gain = macro - keep_macro
                    break_rate = pooled["break_rate"]
                    eligible = break_rate is not None and break_rate <= maximum_break + tolerance
                    gate = eligible and gain + tolerance >= minimum_gain and int(pooled["repairs"]) >= minimum_repairs
                    key = (feature_set, model_name, float(repair_minimum), float(damage_penalty), float(threshold))
                    row = {
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "repair_probability_minimum": float(repair_minimum),
                        "damage_penalty": float(damage_penalty),
                        "safe_score_threshold": float(threshold),
                        "six_cell_macro_accuracy": macro,
                        "six_cell_net_repair_gain": gain,
                        "pooled_original_accuracy": pooled["original_accuracy"],
                        "pooled_final_accuracy": pooled["final_accuracy"],
                        "repairs": pooled["repairs"],
                        "damage": pooled["damage"],
                        "fix_rate": pooled["fix_rate"],
                        "break_rate": break_rate,
                        "correction_abstentions": sum(int(item["correction_abstained"]) for item in predictions),
                        "eligible_under_break_constraint": eligible,
                        "gate_passed": gate,
                        "_key": key,
                    }
                    leaderboard.append(row)
    eligible_rows = [row for row in leaderboard if row["eligible_under_break_constraint"]]
    if not eligible_rows:
        raise PlanBError("No V2 operating point satisfies the validation BreakRate constraint")
    eligible_rows.sort(key=_leaderboard_key)
    selected = eligible_rows[0]
    selected_scored = prediction_cache[(str(selected["feature_set"]), str(selected["model_name"]))]
    selected_predictions = _selector_predictions(
        selected_scored,
        budget=primary_budget,
        method="v2_selector",
        repair_minimum=float(selected["repair_probability_minimum"]),
        damage_penalty=float(selected["damage_penalty"]),
        threshold=float(selected["safe_score_threshold"]),
        tolerance=tolerance,
    )
    ranked = sorted(leaderboard, key=lambda row: (not row["eligible_under_break_constraint"], *_leaderboard_key(row)))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row.pop("_key", None)
    selected.pop("_key", None)
    validation_predictions = baseline_predictions + selected_predictions
    validation_metrics = _slice_metrics(validation_predictions)
    leaderboard_path = training_dir / "validation_ablation_leaderboard.csv"
    predictions_path = training_dir / "validation_predictions.csv"
    metrics_path = training_dir / "validation_metrics.csv"
    dependency_path = training_dir / "environment_preflight.json"
    atomic_write_csv(leaderboard_path, ranked, LEADERBOARD_FIELDS)
    atomic_write_csv(predictions_path, validation_predictions, PREDICTION_FIELDS)
    if not validation_metrics:
        raise PlanBError("Missing V2 validation metrics")
    atomic_write_csv(metrics_path, validation_metrics, tuple(validation_metrics[0].keys()))
    atomic_write_json(
        dependency_path,
        with_self_hash({"schema_version": "plan_b_v2_environment_v1", **dependencies}),
    )
    artifacts.extend([leaderboard_path, predictions_path, metrics_path, dependency_path])
    selected_feature = str(selected["feature_set"])
    selected_model = str(selected["model_name"])
    _, _, selected_repair_path, selected_damage_path = fitted[(selected_feature, selected_model)]
    gate_passed = bool(selected["gate_passed"])
    freeze = with_self_hash(
        {
            "schema_version": "plan_b_v2_freeze_v1",
            "status": "FROZEN_VALIDATION_GATE_PASSED" if gate_passed else "FROZEN_VALIDATION_GATE_FAILED",
            "experiment_status": config["status"],
            "post_hoc_exploratory": True,
            "config_sha256": config_hash,
            "v1_backbone_config_sha256": config["v1_backbone_config_sha256"],
            "v1_audit_report_sha256": audit["report_sha256"],
            "input_inventory": inventory,
            "dependencies": dependencies,
            "training_summaries": training_summaries,
            "selection_objective": search["selection_objective"],
            "abstention_action": search["abstention_action"],
            "selected_operating_point": selected,
            "selected_feature_names": list(feature_names(selected_feature)),
            "selected_models": {
                "repair": {"path": selected_repair_path.relative_to(output_dir).as_posix(), "sha256": sha256_file(selected_repair_path)},
                "damage": {"path": selected_damage_path.relative_to(output_dir).as_posix(), "sha256": sha256_file(selected_damage_path)},
            },
            "validation": {
                "eligible_model_examples": len(validation_records),
                "primary_budget": primary_budget,
                "keep_original_six_cell_macro_accuracy": keep_macro,
                "gate_passed": gate_passed,
                "candidate_operating_points": len(leaderboard),
                "break_rate_constraint": maximum_break,
                "minimum_gain_gate": minimum_gain,
            },
            "cached_confidence_warning": (
                "Probe confidence values are reconstructed from cached shifts and are an unverified proxy; "
                "results selecting that feature set require explicit caveating."
            ),
            "calibration_split_accessed": False,
            "test_split_accessed": False,
            "shift_split_accessed": False,
            "new_mllm_calls": 0,
            "training_artifacts": _artifact_inventory(output_dir, artifacts),
        }
    )
    atomic_write_json(freeze_path, freeze)
    return freeze


def run_evaluate(
    *, repo_root: Path, config: Mapping[str, Any], output_dir: Path, resume: bool, overwrite: bool,
    confirm_locked_evaluation: bool,
) -> dict[str, Any]:
    if not confirm_locked_evaluation:
        raise PlanBError("Locked V2 evaluation requires --confirm-locked-evaluation")
    freeze_path = output_dir / "training" / "freeze_manifest.json"
    if not freeze_path.is_file():
        raise PlanBError("Run V2 prepare before locked evaluation")
    freeze = load_json(freeze_path)
    verify_self_hash(freeze)
    if freeze.get("status") != "FROZEN_VALIDATION_GATE_PASSED":
        raise PlanBError("V2 validation gate failed; locked test/shift remain closed")
    report_path = output_dir / "locked_evaluation" / "evaluation_report.json"
    existing = _stage_guard(report_path, resume=resume, overwrite=overwrite)
    if existing is not None:
        return existing
    _, v1_config = load_v1_config(repo_root, config)
    records, _ = load_and_verify_inputs(repo_root, v1_config)
    cohort, _ = build_cohort_index(records)
    lookup = _cohort_lookup(cohort)
    evaluation_records = [
        row for row in records
        if str(row["split"]) in {"test", "shift"}
        and int(lookup[(str(row["instance_id"]), str(row["model_id"]))]["closed_answer_eligible"]) == 1
    ]
    operating = freeze["selected_operating_point"]
    repair_entry = freeze["selected_models"]["repair"]
    damage_entry = freeze["selected_models"]["damage"]
    repair_path = output_dir / repair_entry["path"]
    damage_path = output_dir / damage_entry["path"]
    if sha256_file(repair_path) != repair_entry["sha256"] or sha256_file(damage_path) != damage_entry["sha256"]:
        raise PlanBError("Frozen V2 model artifact hash mismatch")
    repair_model = load_estimator(repair_path)
    damage_model = load_estimator(damage_path)
    budgets = [int(value) for value in config["evaluation_budgets"]]
    predictions = _baseline_predictions(evaluation_records, budgets, config)
    for budget in budgets:
        scored = _score_records(
            evaluation_records, str(operating["feature_set"]), repair_model, damage_model, budget, config
        )
        predictions.extend(
            _selector_predictions(
                scored, budget=budget, method="v2_selector",
                repair_minimum=float(operating["repair_probability_minimum"]),
                damage_penalty=float(operating["damage_penalty"]),
                threshold=float(operating["safe_score_threshold"]),
                tolerance=float(config["safe_correction_search"]["tie_tolerance"]),
            )
        )
    metrics = _slice_metrics(predictions)
    evaluation_dir = output_dir / "locked_evaluation"
    prediction_path = evaluation_dir / "predictions.csv"
    metric_path = evaluation_dir / "metrics.csv"
    bootstrap_path = evaluation_dir / "cluster_bootstrap.json"
    atomic_write_csv(prediction_path, predictions, PREDICTION_FIELDS)
    atomic_write_csv(metric_path, metrics, tuple(metrics[0].keys()))
    primary_budget = int(config["primary_budget"])
    bootstraps = [
        _bootstrap_delta(predictions, split, "v2_selector", primary_budget, config)
        for split in ("test", "shift")
    ]
    atomic_write_json(
        bootstrap_path,
        with_self_hash({"schema_version": "plan_b_v2_bootstrap_v1", "comparisons": bootstraps}),
    )
    summaries = {}
    for split in ("test", "shift"):
        split_rows = [row for row in predictions if row["split"] == split]
        selector = _pooled(split_rows, "v2_selector", primary_budget)
        keep = _pooled(split_rows, "keep_original", primary_budget)
        summaries[split] = {
            "v2_selector": selector,
            "keep_original": keep,
            "pooled_net_repair_gain": selector["final_accuracy"] - keep["final_accuracy"],
            "break_constraint_passed": selector["break_rate"] is not None and selector["break_rate"] <= float(config["gates"]["maximum_test_break_rate"]),
        }
    report = with_self_hash(
        {
            "schema_version": "plan_b_v2_locked_evaluation_v1",
            "status": "COMPLETE_POST_HOC_EXPLORATORY",
            "freeze_manifest_sha256": freeze["report_sha256"],
            "selected_operating_point": operating,
            "summaries": summaries,
            "no_post_test_tuning": True,
            "test_split_accessed": True,
            "shift_split_accessed": True,
            "new_mllm_calls": 0,
            "artifacts": _artifact_inventory(output_dir, [prediction_path, metric_path, bootstrap_path]),
        }
    )
    atomic_write_json(report_path, report)
    return report


def validate_outputs(output_dir: Path, require_evaluation: bool) -> dict[str, Any]:
    errors = []
    freeze_path = output_dir / "training" / "freeze_manifest.json"
    freeze = None
    if not freeze_path.is_file():
        errors.append("Missing V2 freeze manifest")
    else:
        try:
            freeze = load_json(freeze_path)
            verify_self_hash(freeze)
            for artifact in freeze.get("training_artifacts", []):
                path = output_dir / artifact["path"]
                if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                    errors.append(f"Training artifact drift: {artifact['path']}")
        except PlanBError as exc:
            errors.append(str(exc))
    evaluation = None
    if require_evaluation:
        report_path = output_dir / "locked_evaluation" / "evaluation_report.json"
        if not report_path.is_file():
            errors.append("Missing required V2 locked evaluation report")
        else:
            try:
                evaluation = load_json(report_path)
                verify_self_hash(evaluation)
                for artifact in evaluation.get("artifacts", []):
                    path = output_dir / artifact["path"]
                    if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                        errors.append(f"Evaluation artifact drift: {artifact['path']}")
            except PlanBError as exc:
                errors.append(str(exc))
    return {
        "mode": "validate",
        "is_valid": not errors,
        "errors": errors,
        "freeze_status": freeze.get("status") if freeze else None,
        "evaluation_status": evaluation.get("status") if evaluation else None,
    }
