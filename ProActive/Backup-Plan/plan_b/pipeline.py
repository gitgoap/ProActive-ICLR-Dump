"""Audited train/validation/test pipeline for the Plan B selector."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core import (
    ABSTAIN,
    FEATURE_NAMES,
    PlanBError,
    acquired_observations,
    atomic_write_csv,
    atomic_write_json,
    build_candidates,
    build_cohort_index,
    candidate_features,
    canonical_json_sha256,
    correctness,
    image_key,
    is_closed_answer,
    is_usable_normalized,
    load_and_verify_inputs,
    load_json,
    metric_summary,
    observation_from_row,
    operational_cost,
    record_gold_norm,
    record_normalizer_type,
    recompute_clean_correctness,
    select_ground,
    select_keep,
    select_majority,
    select_oracle,
    select_supported_ground,
    select_with_scores,
    sha256_file,
    verify_self_hash,
    with_self_hash,
)


COHORT_FIELDS = (
    "instance_id", "group_id", "image_key", "dataset", "split", "model_id",
    "answer_type", "closed_answer_eligible", "image_excluded_from_fit", "proposed_fit_eligible",
)
PREDICTION_FIELDS = (
    "split", "dataset", "model_id", "instance_id", "group_id", "image_key",
    "budget", "method", "selected_source", "selected_norm_answer", "selected_raw_answer", "correct",
    "original_correct", "repair", "damage", "abstain", "matched_cost",
    "operational_cost", "candidate_count",
)
METRIC_FIELDS = (
    "split", "budget", "method", "slice_type", "slice_value", "n",
    "original_accuracy", "final_accuracy", "repairs", "damage", "fix_rate",
    "break_rate", "abstentions", "candidate_coverage", "mean_matched_cost",
    "mean_operational_cost",
)
INPUT_INVENTORY_FIELDS = ("path", "sha256", "rows", "bytes")


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != "plan_b_config_v1":
        raise PlanBError("Unsupported Plan B config schema")
    if tuple(config.get("feature_names", ())) != FEATURE_NAMES:
        raise PlanBError("Config feature order does not match implementation")
    order = config.get("acquisition_order")
    if order != ["grounding", "blur", "crop", "brightness", "noise", "blank"]:
        raise PlanBError("Acquisition order drift")
    if config.get("decisions", {}).get("test_or_shift_tuning") is not False:
        raise PlanBError("Config must forbid test/shift tuning")
    return config


def _git_revision(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_versions() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "sklearn"):
        try:
            module = __import__(package)
            result[package] = module.__version__
        except ImportError:
            result[package] = None
    return result


def _stage_guard(path: Path, *, resume: bool, overwrite: bool) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if overwrite:
        return None
    if not resume:
        raise PlanBError(f"Output already exists; use --resume or --overwrite: {path}")
    payload = load_json(path)
    verify_self_hash(payload)
    return payload


def _cohort_lookup(cohort: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["instance_id"]), str(row["model_id"])): row for row in cohort}


def _image_overlap_report(
    records: Sequence[Mapping[str, Any]],
    cohort: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    splits = ("train", "val", "cal", "test", "shift")
    split_by_image: dict[str, set[str]] = defaultdict(set)
    all_images: dict[str, set[str]] = {split: set() for split in splits}
    for row in records:
        split = str(row["split"])
        key = image_key(row)
        split_by_image[key].add(split)
        all_images.setdefault(split, set()).add(key)
    pair_counts: dict[str, int] = {}
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1:]:
            pair_counts[f"{left}|{right}"] = len(all_images[left] & all_images[right])

    eligible_images = {
        split: {
            str(row["image_key"])
            for row in cohort
            if row["split"] == split and int(row["proposed_fit_eligible"]) == 1
        }
        for split in ("train", "val")
    }
    protected_intersections = {
        "eligible_train|val": len(eligible_images["train"] & all_images["val"]),
        "eligible_train|cal": len(eligible_images["train"] & all_images["cal"]),
        "eligible_train|test": len(eligible_images["train"] & all_images["test"]),
        "eligible_train|shift": len(eligible_images["train"] & all_images["shift"]),
        "eligible_val|cal": len(eligible_images["val"] & all_images["cal"]),
        "eligible_val|test": len(eligible_images["val"] & all_images["test"]),
        "eligible_val|shift": len(eligible_images["val"] & all_images["shift"]),
    }
    errors = [
        f"Protected image intersection is nonzero: {name}={count}"
        for name, count in protected_intersections.items()
        if count
    ]
    return with_self_hash(
        {
            "schema_version": "plan_b_image_overlap_v1",
            "identity_scope": "metadata_only_not_pixel_or_perceptual_deduplication",
            "unique_image_keys": len(split_by_image),
            "metadata_cross_split_images": sum(len(values) > 1 for values in split_by_image.values()),
            "original_split_pair_intersections": pair_counts,
            "eligible_fit_unique_images": {
                split: len(values) for split, values in eligible_images.items()
            },
            "post_purge_protected_intersections": protected_intersections,
            "is_valid": not errors,
            "errors": errors,
        }
    )


def run_audit(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    resume: bool,
    overwrite: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    audit_dir = output_dir / ("audit" if limit is None else f"audit_smoke_limit{limit}")
    report_path = audit_dir / "audit_report.json"
    existing = _stage_guard(report_path, resume=resume, overwrite=overwrite)
    config_hash = canonical_json_sha256(config)
    if existing is not None:
        if existing.get("config_sha256") != config_hash:
            raise PlanBError("Unsafe audit resume refused: config hash changed")
        for item in existing["input_inventory"]:
            path = repo_root / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise PlanBError(f"Unsafe audit resume refused: input drift for {item['path']}")
        for artifact in existing.get("audit_artifacts", []):
            path = output_dir / artifact["path"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise PlanBError(f"Unsafe audit resume refused: output drift for {artifact['path']}")
        return existing

    records, inventory = load_and_verify_inputs(repo_root, config, limit=limit)
    cohort, exclusions = build_cohort_index(records)
    cohort_lookup = _cohort_lookup(cohort)
    split_counts = Counter(str(row.get("split")) for row in records)
    dataset_split_counts = Counter((str(row.get("split")), str(row.get("dataset"))) for row in records)
    fit_counts = Counter(
        str(row["split"]) for row in cohort if int(row["proposed_fit_eligible"]) == 1
    )
    closed_test = sum(int(row["closed_answer_eligible"]) for row in cohort if row["split"] == "test")
    closed_shift = sum(int(row["closed_answer_eligible"]) for row in cohort if row["split"] == "shift")
    image_report = _image_overlap_report(records, cohort)
    if not image_report["is_valid"]:
        raise PlanBError("; ".join(image_report["errors"]))

    mismatches: list[dict[str, Any]] = []
    unusable: list[dict[str, Any]] = []
    for record in records:
        recomputed = recompute_clean_correctness(record)
        saved = record.get("clean", {}).get("correct")
        if saved not in {0, 1, False, True}:
            raise PlanBError(f"Invalid saved clean correctness for {record['instance_id']}")
        if recomputed != int(saved):
            mismatches.append(
                {
                    "instance_id": record["instance_id"],
                    "model_id": record["model_id"],
                    "dataset": record["dataset"],
                    "saved": int(saved),
                    "recomputed": recomputed,
                    "normalizer_type": record_normalizer_type(record),
                }
            )
        for source, payload in record.get("probes", {}).items():
            if not isinstance(payload, Mapping):
                raise PlanBError(f"Malformed probe payload: {record['instance_id']} {source}")
            if payload.get("valid") is True and payload.get("applicable") is True:
                norm = payload.get("norm_answer")
                if not is_usable_normalized(norm):
                    unusable.append(
                        {
                            "instance_id": record["instance_id"],
                            "model_id": record["model_id"],
                            "dataset": record["dataset"],
                            "split": record["split"],
                            "probe": source,
                            "norm_answer": norm if isinstance(norm, str) else "",
                            "reason": "valid_applicable_but_unusable_normalized_answer",
                        }
                    )

    expected = config["expected"]
    observed = {
        "all_records": len(records),
        "train_records": split_counts["train"],
        "validation_records": split_counts["val"],
        "calibration_records": split_counts["cal"],
        "test_records": split_counts["test"],
        "shift_records": split_counts["shift"],
        "fit_train_records": fit_counts["train"],
        "fit_validation_records": fit_counts["val"],
        "closed_test_records": closed_test,
        "closed_shift_records": closed_shift,
        "metadata_cross_split_images": image_report["metadata_cross_split_images"],
        "unusable_probe_answers": len(unusable),
        "clean_correctness_mismatches": len(mismatches),
    }
    errors: list[str] = []
    if limit is None:
        for key, expected_value in expected.items():
            if observed.get(key) != expected_value:
                errors.append(f"{key}: expected={expected_value} observed={observed.get(key)}")
    if mismatches:
        errors.append(f"Recomputed clean correctness disagrees for {len(mismatches)} rows")

    cohort_path = audit_dir / "cohort_index.csv"
    exclusions_path = audit_dir / "image_exclusions.csv"
    unusable_path = audit_dir / "unusable_probe_answers.csv"
    mismatches_path = audit_dir / "clean_correctness_mismatches.csv"
    inventory_path = audit_dir / "input_inventory.csv"
    image_report_path = audit_dir / "image_overlap_report.json"
    atomic_write_csv(inventory_path, inventory, INPUT_INVENTORY_FIELDS)
    atomic_write_csv(cohort_path, cohort, COHORT_FIELDS)
    atomic_write_csv(
        exclusions_path,
        exclusions,
        ("image_key", "excluded_split", "other_splits", "reason"),
    )
    atomic_write_csv(
        unusable_path,
        unusable,
        ("instance_id", "model_id", "dataset", "split", "probe", "norm_answer", "reason"),
    )
    atomic_write_csv(
        mismatches_path,
        mismatches,
        ("instance_id", "model_id", "dataset", "saved", "recomputed", "normalizer_type"),
    )
    atomic_write_json(image_report_path, image_report)
    report = with_self_hash(
        {
            "schema_version": "plan_b_audit_v1",
            "is_valid": not errors and limit is None,
            "scientific_run": limit is None,
            "config_sha256": config_hash,
            "source_cache_revision": config["source_cache_revision"],
            "input_inventory": inventory,
            "observed": observed,
            "expected": expected,
            "dataset_split_counts": {
                f"{split}|{dataset}": count
                for (split, dataset), count in sorted(dataset_split_counts.items())
            },
            "unique_model_examples": len({(row["instance_id"], row["model_id"]) for row in records}),
            "unique_image_clusters": len({image_key(row) for row in records}),
            "image_exclusion_rows": len(exclusions),
            "clean_correctness_audit_scope": "all_cached_records_with_declared_contract",
            "audit_artifacts": _artifact_inventory(
                output_dir,
                [
                    inventory_path,
                    cohort_path,
                    exclusions_path,
                    unusable_path,
                    mismatches_path,
                    image_report_path,
                ],
            ),
            "errors": errors,
        }
    )
    atomic_write_json(report_path, report)
    if errors:
        raise PlanBError("Audit failed: " + "; ".join(errors))
    return report


def _training_rows(
    records: Sequence[Mapping[str, Any]],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[list[float]], list[int], list[float], dict[str, Any]]:
    lookup = _cohort_lookup(cohort)
    eligible = [
        row for row in records
        if lookup[(str(row["instance_id"]), str(row["model_id"]))]["split"] == "train"
        and int(lookup[(str(row["instance_id"]), str(row["model_id"]))]["proposed_fit_eligible"]) == 1
    ]
    group_sizes = Counter(str(row["group_id"]) for row in eligible)
    features: list[list[float]] = []
    labels: list[int] = []
    raw_weights: list[float] = []
    no_candidate_prefixes = 0
    class_counts: Counter[int] = Counter()
    for row in eligible:
        gold_norm = record_gold_norm(row)
        for budget in config["training_budgets"]:
            observations, cost = acquired_observations(row, int(budget), config["acquisition_order"])
            candidates = build_candidates(observations, cost)
            if not candidates:
                no_candidate_prefixes += 1
                continue
            group_size = group_sizes[str(row["group_id"])]
            if group_size <= 0:
                raise PlanBError("Invalid training group size")
            weight = 1.0 / (group_size * len(config["training_budgets"]) * len(candidates))
            for candidate in candidates:
                feature = candidate_features(candidate, candidates, cost)
                label = int(candidate.norm_answer == gold_norm)
                features.append(list(feature))
                labels.append(label)
                raw_weights.append(weight)
                class_counts[label] += 1
    if not features or set(labels) != {0, 1}:
        raise PlanBError("Training candidates do not contain both classes")
    mean_weight = sum(raw_weights) / len(raw_weights)
    weights = [weight / mean_weight for weight in raw_weights]
    return features, labels, weights, {
        "eligible_model_examples": len(eligible),
        "candidate_rows": len(features),
        "candidate_class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "no_candidate_prefixes": no_candidate_prefixes,
        "mean_normalized_sample_weight": sum(weights) / len(weights),
    }


def _manual_scores(model: Mapping[str, Any], features: Sequence[Sequence[float]]) -> list[float]:
    coefficients = model["coefficients"]
    intercept = float(model["intercept"])
    if len(coefficients) != len(FEATURE_NAMES):
        raise PlanBError("Stored selector coefficient dimension mismatch")
    result: list[float] = []
    for row in features:
        value = intercept + sum(float(coef) * float(feature) for coef, feature in zip(coefficients, row))
        probability = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, value))))
        result.append(probability)
    return result


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    margin: float,
    budgets: Sequence[int],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selector_config = config["selector"]
    for record in records:
        gold_norm = record_gold_norm(record)
        clean = observation_from_row(record, "clean")
        original_correct = int(clean.norm_answer == gold_norm)
        for budget in budgets:
            observations, matched_cost = acquired_observations(record, int(budget), config["acquisition_order"])
            candidates = build_candidates(observations, matched_cost)
            features = [candidate_features(candidate, candidates, matched_cost) for candidate in candidates]
            scores = _manual_scores(model, features) if features else []
            selections = {
                "keep_original": select_keep(observations),
                "always_ground": select_ground(observations),
                "majority": select_majority(observations),
                "supported_ground": select_supported_ground(observations),
                "selector": select_with_scores(
                    candidates,
                    scores,
                    margin,
                    float(selector_config["switch_probability_threshold"]),
                    float(config["decisions"]["floating_tie_tolerance"]),
                ),
                "oracle": select_oracle(observations, gold_norm),
            }
            for method in methods:
                selection = selections[method]
                final_correct = correctness(selection, gold_norm)
                rows.append(
                    {
                        "split": record["split"],
                        "dataset": record["dataset"],
                        "model_id": record["model_id"],
                        "instance_id": record["instance_id"],
                        "group_id": record["group_id"],
                        "image_key": image_key(record),
                        "budget": int(budget),
                        "method": method,
                        "selected_source": selection.source,
                        "selected_norm_answer": selection.norm_answer,
                        "selected_raw_answer": selection.raw_answer,
                        "correct": final_correct,
                        "original_correct": original_correct,
                        "repair": int(not original_correct and final_correct),
                        "damage": int(original_correct and not final_correct),
                        "abstain": int(selection.norm_answer == ABSTAIN),
                        "matched_cost": matched_cost,
                        "operational_cost": operational_cost(method, observations, matched_cost),
                        "candidate_count": len(candidates),
                    }
                )
    return rows


def _slice_metrics(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        base = (str(row["split"]), int(row["budget"]), str(row["method"]))
        groups[base + ("pooled", "all")].append(row)
        groups[base + ("dataset", str(row["dataset"]))].append(row)
        groups[base + ("model", str(row["model_id"]))].append(row)
        groups[base + ("dataset_model", f"{row['dataset']}|{row['model_id']}")].append(row)
    output: list[dict[str, Any]] = []
    for (split, budget, method, slice_type, slice_value), rows in sorted(groups.items()):
        output.append(
            {
                "split": split,
                "budget": budget,
                "method": method,
                "slice_type": slice_type,
                "slice_value": slice_value,
                **metric_summary(rows),
            }
        )
    return output


def _six_cell_macro(predictions: Sequence[Mapping[str, Any]], method: str, budget: int) -> float:
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row["method"] == method and int(row["budget"]) == budget:
            cells[(str(row["dataset"]), str(row["model_id"]))].append(row)
    if len(cells) != 6 or any(not rows for rows in cells.values()):
        raise PlanBError(f"Expected six dataset/model cells for {method}, found {len(cells)}")
    return sum(metric_summary(rows)["final_accuracy"] for rows in cells.values()) / 6.0


def _pooled_metric(predictions: Sequence[Mapping[str, Any]], method: str, budget: int) -> dict[str, Any]:
    rows = [row for row in predictions if row["method"] == method and int(row["budget"]) == budget]
    return metric_summary(rows)


def run_prepare(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    audit = run_audit(
        repo_root=repo_root,
        config=config,
        output_dir=output_dir,
        resume=resume,
        overwrite=overwrite,
    )
    if not audit["is_valid"]:
        raise PlanBError("Preparation requires a complete valid audit")
    freeze_path = output_dir / "training" / "freeze_manifest.json"
    existing = _stage_guard(freeze_path, resume=resume, overwrite=overwrite)
    config_hash = canonical_json_sha256(config)
    if existing is not None:
        if existing.get("config_sha256") != config_hash or existing.get("audit_report_sha256") != audit["report_sha256"]:
            raise PlanBError("Unsafe preparation resume refused: provenance drift")
        for artifact in existing.get("training_artifacts", []):
            path = output_dir / artifact["path"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise PlanBError(f"Unsafe preparation resume refused: output drift for {artifact['path']}")
        return existing

    try:
        import numpy as np
        import sklearn
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise PlanBError("Preparation requires NumPy and scikit-learn in the active CPU environment") from exc

    environment = _runtime_versions()
    if environment.get("numpy") is None or environment.get("sklearn") is None:
        raise PlanBError("NumPy/scikit-learn runtime versions could not be recorded")
    training_dir = output_dir / "training"
    environment_path = training_dir / "environment_preflight.json"
    atomic_write_json(
        environment_path,
        with_self_hash(
            {
                "schema_version": "plan_b_environment_v1",
                "recorded_before_fit": True,
                "runtime_versions": environment,
            }
        ),
    )

    records, _ = load_and_verify_inputs(repo_root, config)
    cohort, _ = build_cohort_index(records)
    features, labels, weights, training_summary = _training_rows(records, cohort, config)
    settings = config["selector"]
    estimator = LogisticRegression(
        penalty=settings["penalty"],
        C=float(settings["C"]),
        fit_intercept=bool(settings["fit_intercept"]),
        solver=settings["solver"],
        tol=float(settings["tolerance"]),
        max_iter=int(settings["max_iter"]),
        random_state=int(config["seed"]),
        class_weight=settings["class_weight"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(np.asarray(features, dtype=np.float64), np.asarray(labels), sample_weight=np.asarray(weights))
    convergence = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)]
    if convergence or int(estimator.n_iter_[0]) >= int(settings["max_iter"]):
        raise PlanBError("Logistic regression did not converge under the frozen solver contract")
    if estimator.classes_.tolist() != [0, 1] or estimator.coef_.shape != (1, len(FEATURE_NAMES)):
        raise PlanBError("Unexpected logistic-regression class or coefficient shape")
    model = {
        "format_version": "plan_b_logistic_v1",
        "feature_names": list(FEATURE_NAMES),
        "classes": [0, 1],
        "coefficients": estimator.coef_[0].astype(float).tolist(),
        "intercept": float(estimator.intercept_[0]),
        "n_iter": int(estimator.n_iter_[0]),
        "settings": settings,
    }
    model["model_sha256"] = canonical_json_sha256(model)

    validation_records = []
    lookup = _cohort_lookup(cohort)
    for row in records:
        entry = lookup[(str(row["instance_id"]), str(row["model_id"]))]
        if entry["split"] == "val" and int(entry["proposed_fit_eligible"]) == 1:
            validation_records.append(row)
    primary_budget = int(config["primary_budget"])
    baseline_methods = ("keep_original", "always_ground", "majority", "supported_ground")
    baseline_predictions = evaluate_records(
        validation_records, config, model, 0.0, [primary_budget], baseline_methods
    )
    baseline_priority = {"keep_original": 0, "majority": 1, "supported_ground": 2, "always_ground": 3}
    comparator = None
    comparator_macro = -math.inf
    tolerance = float(config["decisions"]["floating_tie_tolerance"])
    baseline_summary: list[dict[str, Any]] = []
    for method in baseline_methods:
        macro = _six_cell_macro(baseline_predictions, method, primary_budget)
        pooled = _pooled_metric(baseline_predictions, method, primary_budget)
        baseline_summary.append({"method": method, "six_cell_macro_accuracy": macro, **pooled})
        if macro > comparator_macro + tolerance or (
            abs(macro - comparator_macro) <= tolerance
            and comparator is not None
            and baseline_priority[method] < baseline_priority[comparator]
        ):
            comparator = method
            comparator_macro = macro
    if comparator is None:
        raise PlanBError("Unable to select a validation comparator")

    margin_summary: list[dict[str, Any]] = []
    best_margin: float | None = None
    best_macro = -math.inf
    best_damage = sys.maxsize
    best_break: float | None = None
    best_predictions: list[dict[str, Any]] = []
    for margin_value in settings["switch_margins"]:
        margin = float(margin_value)
        predictions = evaluate_records(validation_records, config, model, margin, [primary_budget], ["selector"])
        macro = _six_cell_macro(predictions, "selector", primary_budget)
        pooled = _pooled_metric(predictions, "selector", primary_budget)
        break_rate = pooled["break_rate"]
        allowed = break_rate is not None and break_rate <= float(config["gates"]["maximum_validation_break_rate"]) + tolerance
        margin_summary.append(
            {
                "margin": margin,
                "six_cell_macro_accuracy": macro,
                "eligible_under_break_constraint": allowed,
                **pooled,
            }
        )
        if not allowed:
            continue
        better = macro > best_macro + tolerance
        tied = abs(macro - best_macro) <= tolerance
        if better or (tied and (pooled["damage"] < best_damage or (pooled["damage"] == best_damage and (best_margin is None or margin > best_margin)))):
            best_margin = margin
            best_macro = macro
            best_damage = pooled["damage"]
            best_break = break_rate
            best_predictions = predictions
    if best_margin is None or best_break is None:
        raise PlanBError("No switching margin satisfies the validation BreakRate constraint")

    validation_gain = best_macro - comparator_macro
    validation_passed = (
        validation_gain + tolerance >= float(config["gates"]["minimum_validation_macro_gain"])
        and best_break <= float(config["gates"]["maximum_validation_break_rate"]) + tolerance
    )
    all_validation_predictions = baseline_predictions + best_predictions
    validation_metrics = _slice_metrics(all_validation_predictions)
    model_path = training_dir / "selector_model.json"
    validation_predictions_path = training_dir / "validation_predictions.csv"
    validation_metrics_path = training_dir / "validation_metrics.csv"
    atomic_write_json(model_path, model)
    atomic_write_csv(validation_predictions_path, all_validation_predictions, PREDICTION_FIELDS)
    atomic_write_csv(validation_metrics_path, validation_metrics, METRIC_FIELDS)

    if _runtime_versions() != environment:
        raise PlanBError("Runtime environment changed during selector preparation")
    freeze = with_self_hash(
        {
            "schema_version": "plan_b_freeze_v1",
            "status": "FROZEN_VALIDATION_GATE_PASSED" if validation_passed else "FROZEN_VALIDATION_GATE_FAILED",
            "config_sha256": config_hash,
            "audit_report_sha256": audit["report_sha256"],
            "input_inventory": audit["input_inventory"],
            "implementation_git_revision": _git_revision(repo_root),
            "runtime_versions": environment,
            "feature_names": list(FEATURE_NAMES),
            "model": model,
            "training_artifacts": _artifact_inventory(
                output_dir,
                [
                    environment_path,
                    model_path,
                    validation_predictions_path,
                    validation_metrics_path,
                ],
            ),
            "training_summary": training_summary,
            "validation": {
                "primary_budget": primary_budget,
                "comparator": comparator,
                "comparator_six_cell_macro_accuracy": comparator_macro,
                "selected_margin": best_margin,
                "selector_six_cell_macro_accuracy": best_macro,
                "selector_pooled_break_rate": best_break,
                "selector_gain": validation_gain,
                "gate_passed": validation_passed,
                "baseline_candidates": baseline_summary,
                "margin_candidates": margin_summary,
            },
            "calibration_split_accessed": false_value(),
            "test_split_accessed": false_value(),
            "shift_split_accessed": false_value(),
        }
    )
    atomic_write_json(freeze_path, freeze)
    return freeze


def false_value() -> bool:
    """Make provenance booleans visually explicit at construction sites."""
    return False


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _cluster_statistics(
    predictions: Sequence[Mapping[str, Any]],
    selector: str,
    comparators: Sequence[str],
    budget: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanBError("Statistical evaluation requires NumPy") from exc
    relevant = [row for row in predictions if int(row["budget"]) == budget]
    by_method_key = {
        (str(row["method"]), str(row["instance_id"]), str(row["model_id"])): row
        for row in relevant
    }
    selector_rows = [row for row in relevant if row["method"] == selector]
    if not selector_rows:
        raise PlanBError("Missing selector rows for statistics")
    clusters = sorted({str(row["image_key"]) for row in selector_rows})
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    cells = sorted({(str(row["dataset"]), str(row["model_id"])) for row in selector_rows})
    if len(cells) != 6:
        raise PlanBError(f"Primary statistics require six cells, found {len(cells)}")
    cell_index = {cell: index for index, cell in enumerate(cells)}
    bootstrap_count = int(config["statistics"]["bootstrap_resamples"])
    randomization_count = int(config["statistics"]["randomization_draws"])
    confidence = float(config["statistics"]["confidence_level"])
    alpha_tail = (1.0 - confidence) / 2.0
    tolerance = float(config["decisions"]["floating_tie_tolerance"])
    raw_p: dict[str, float] = {}
    output: list[dict[str, Any]] = []

    for comparator in comparators:
        if comparator == selector:
            continue
        n_clusters = len(clusters)
        selector_correct = np.zeros((n_clusters, 6), dtype=np.float64)
        comparator_correct = np.zeros((n_clusters, 6), dtype=np.float64)
        counts = np.zeros((n_clusters, 6), dtype=np.float64)
        selector_repairs = np.zeros(n_clusters, dtype=np.float64)
        selector_damage = np.zeros(n_clusters, dtype=np.float64)
        originally_wrong = np.zeros(n_clusters, dtype=np.float64)
        originally_correct = np.zeros(n_clusters, dtype=np.float64)
        for row in selector_rows:
            key = (comparator, str(row["instance_id"]), str(row["model_id"]))
            other = by_method_key.get(key)
            if other is None:
                raise PlanBError(f"Unpaired comparator row: {key}")
            cluster = cluster_index[str(row["image_key"])]
            cell = cell_index[(str(row["dataset"]), str(row["model_id"]))]
            selector_correct[cluster, cell] += int(row["correct"])
            comparator_correct[cluster, cell] += int(other["correct"])
            counts[cluster, cell] += 1
            selector_repairs[cluster] += int(row["repair"])
            selector_damage[cluster] += int(row["damage"])
            originally_wrong[cluster] += 1 - int(row["original_correct"])
            originally_correct[cluster] += int(row["original_correct"])

        total_counts = counts.sum(axis=0)
        observed_macro = float(np.mean(selector_correct.sum(axis=0) / total_counts) - np.mean(comparator_correct.sum(axis=0) / total_counts))
        observed_pooled = float((selector_correct.sum() - comparator_correct.sum()) / counts.sum())
        observed_fix = float(selector_repairs.sum() / originally_wrong.sum())
        observed_break = float(selector_damage.sum() / originally_correct.sum())

        bootstrap_rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
        boot_macro: list[float] = []
        boot_pooled: list[float] = []
        boot_fix: list[float] = []
        boot_break: list[float] = []
        attempts = 0
        while len(boot_macro) < bootstrap_count:
            attempts += 1
            if attempts > bootstrap_count * 20:
                raise PlanBError("Too many invalid bootstrap draws")
            draw = bootstrap_rng.integers(0, n_clusters, size=n_clusters)
            multiplicity = np.bincount(draw, minlength=n_clusters).astype(np.float64)
            drawn_counts = multiplicity @ counts
            if np.any(drawn_counts == 0):
                continue
            selected = multiplicity @ selector_correct
            compared = multiplicity @ comparator_correct
            wrong_den = float(multiplicity @ originally_wrong)
            correct_den = float(multiplicity @ originally_correct)
            if wrong_den <= 0 or correct_den <= 0:
                continue
            boot_macro.append(float(np.mean(selected / drawn_counts) - np.mean(compared / drawn_counts)))
            boot_pooled.append(float((selected.sum() - compared.sum()) / drawn_counts.sum()))
            boot_fix.append(float((multiplicity @ selector_repairs) / wrong_den))
            boot_break.append(float((multiplicity @ selector_damage) / correct_den))

        percentile_method = str(config["statistics"]["percentile_method"])
        def interval(values: Sequence[float]) -> tuple[float, float]:
            low, high = np.quantile(
                np.asarray(values), [alpha_tail, 1.0 - alpha_tail], method=percentile_method
            )
            return float(low), float(high)

        random_rng = np.random.default_rng(int(config["statistics"]["randomization_seed"]))
        extreme = 0
        delta = selector_correct - comparator_correct
        for _ in range(randomization_count):
            signs = random_rng.choice(np.asarray([-1.0, 1.0]), size=n_clusters)
            permuted = (signs[:, None] * delta).sum(axis=0) / total_counts
            statistic = float(np.mean(permuted))
            if statistic + tolerance >= observed_macro:
                extreme += 1
        p_value = (1.0 + extreme) / (randomization_count + 1.0)
        raw_p[comparator] = p_value
        macro_ci = interval(boot_macro)
        pooled_ci = interval(boot_pooled)
        fix_ci = interval(boot_fix)
        break_ci = interval(boot_break)
        output.append(
            {
                "selector": selector,
                "comparator": comparator,
                "budget": budget,
                "clusters": n_clusters,
                "model_examples": len(selector_rows),
                "observed_six_cell_macro_difference": observed_macro,
                "macro_ci_low": macro_ci[0],
                "macro_ci_high": macro_ci[1],
                "observed_pooled_accuracy_difference": observed_pooled,
                "pooled_ci_low": pooled_ci[0],
                "pooled_ci_high": pooled_ci[1],
                "selector_fix_rate": observed_fix,
                "fix_rate_ci_low": fix_ci[0],
                "fix_rate_ci_high": fix_ci[1],
                "selector_break_rate": observed_break,
                "break_rate_ci_low": break_ci[0],
                "break_rate_ci_high": break_ci[1],
                "randomization_p_one_sided": p_value,
                "holm_adjusted_p": None,
            }
        )
    adjusted = _holm_adjust(raw_p)
    for row in output:
        row["holm_adjusted_p"] = adjusted[str(row["comparator"])]
    return output


def _artifact_inventory(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    inventory = []
    for path in paths:
        if not path.is_file():
            raise PlanBError(f"Expected artifact missing: {path}")
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return inventory


def run_evaluate(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    resume: bool,
    overwrite: bool,
    confirm_locked_evaluation: bool,
) -> dict[str, Any]:
    if not confirm_locked_evaluation:
        raise PlanBError("Locked evaluation requires --confirm-locked-evaluation")
    freeze_path = output_dir / "training" / "freeze_manifest.json"
    if not freeze_path.is_file():
        raise PlanBError("Run preparation before locked evaluation")
    freeze = load_json(freeze_path)
    verify_self_hash(freeze)
    if freeze.get("status") != "FROZEN_VALIDATION_GATE_PASSED":
        raise PlanBError("Validation gate did not pass; locked evaluation is prohibited")
    final_path = output_dir / "evaluation" / "final_report.json"
    existing = _stage_guard(final_path, resume=resume, overwrite=overwrite)
    if existing is not None:
        if existing.get("freeze_manifest_sha256") != freeze["report_sha256"]:
            raise PlanBError("Unsafe evaluation resume refused: freeze drift")
        # A crash after writing the report but before writing the signed run
        # manifest must not be mistaken for a complete locked evaluation.
        run_manifest_path = output_dir / "run_manifest.json"
        if run_manifest_path.is_file():
            validation = validate_outputs(output_dir, require_evaluation=True)
            if not validation["is_valid"]:
                raise PlanBError(
                    "Unsafe evaluation resume refused: " + "; ".join(validation["errors"])
                )
            return existing

    config_hash = canonical_json_sha256(config)
    if freeze["config_sha256"] != config_hash:
        raise PlanBError("Freeze/config mismatch")
    audit = load_json(output_dir / "audit" / "audit_report.json")
    verify_self_hash(audit)
    if freeze["audit_report_sha256"] != audit["report_sha256"]:
        raise PlanBError("Freeze/audit mismatch")
    records, inventory = load_and_verify_inputs(repo_root, config)
    if inventory != audit["input_inventory"]:
        raise PlanBError("Evaluation input inventory differs from audit")
    test_records = [row for row in records if row.get("split") == "test" and is_closed_answer(row)]
    shift_records = [row for row in records if row.get("split") == "shift" and is_closed_answer(row)]
    if len(test_records) != int(config["expected"]["closed_test_records"]):
        raise PlanBError("Closed test cohort count drift")
    if len(shift_records) != int(config["expected"]["closed_shift_records"]):
        raise PlanBError("Closed shift cohort count drift")
    methods = ("keep_original", "always_ground", "majority", "supported_ground", "selector", "oracle")
    budgets = [int(value) for value in config["evaluation_budgets"]]
    margin = float(freeze["validation"]["selected_margin"])
    model = freeze["model"]
    predictions = evaluate_records(test_records + shift_records, config, model, margin, budgets, methods)
    evaluation_dir = output_dir / "evaluation"
    predictions_path = evaluation_dir / "predictions.csv"
    metrics_path = evaluation_dir / "metrics.csv"
    stats_path = evaluation_dir / "paired_statistics.csv"
    oracle_path = evaluation_dir / "oracle_bound.csv"
    atomic_write_csv(predictions_path, predictions, PREDICTION_FIELDS)
    metrics = _slice_metrics(predictions)
    atomic_write_csv(metrics_path, metrics, METRIC_FIELDS)
    oracle_metrics = [row for row in metrics if row["method"] == "oracle"]
    atomic_write_csv(oracle_path, oracle_metrics, METRIC_FIELDS)

    primary_budget = int(config["primary_budget"])
    comparator = str(freeze["validation"]["comparator"])
    test_predictions = [row for row in predictions if row["split"] == "test"]
    selector_macro = _six_cell_macro(test_predictions, "selector", primary_budget)
    comparator_macro = _six_cell_macro(test_predictions, comparator, primary_budget)
    keep_macro = _six_cell_macro(test_predictions, "keep_original", primary_budget)
    selector_pooled = _pooled_metric(test_predictions, "selector", primary_budget)
    reference_methods = ("keep_original", "always_ground", "majority", "oracle")
    reference_baselines: dict[str, dict[str, Any]] = {}
    for method in reference_methods:
        pooled = _pooled_metric(test_predictions, method, primary_budget)
        correct_count = round(float(pooled["final_accuracy"]) * int(pooled["n"]))
        expected_correct = int(config["reference_core_test_b2_correct"][method])
        if correct_count != expected_correct:
            raise PlanBError(
                f"Reference baseline drift for {method}: "
                f"expected_correct={expected_correct} observed_correct={correct_count}"
            )
        reference_baselines[method] = {
            "correct": correct_count,
            "pooled": pooled,
            "six_cell_macro_accuracy": _six_cell_macro(
                test_predictions, method, primary_budget
            ),
        }
    keep_cells: dict[tuple[str, str], float] = {}
    selector_cells: dict[tuple[str, str], float] = {}
    for dataset in sorted({str(row["dataset"]) for row in test_records}):
        for model_id in sorted({str(row["model_id"]) for row in test_records}):
            key = (dataset, model_id)
            keep_rows = [row for row in test_predictions if row["dataset"] == dataset and row["model_id"] == model_id and row["method"] == "keep_original" and int(row["budget"]) == primary_budget]
            selector_rows = [row for row in test_predictions if row["dataset"] == dataset and row["model_id"] == model_id and row["method"] == "selector" and int(row["budget"]) == primary_budget]
            if keep_rows and selector_rows:
                keep_cells[key] = metric_summary(keep_rows)["final_accuracy"]
                selector_cells[key] = metric_summary(selector_rows)["final_accuracy"]
    comparators = ["keep_original"]
    if comparator != "keep_original":
        comparators.append(comparator)
    statistics = _cluster_statistics(test_predictions, "selector", comparators, primary_budget, config)
    stat_fields = tuple(statistics[0].keys()) if statistics else ()
    atomic_write_csv(stats_path, statistics, stat_fields)
    statistic_by_comparator = {row["comparator"]: row for row in statistics}
    all_statistical = all(
        float(row["macro_ci_low"]) > 0.0
        and float(row["holm_adjusted_p"]) < float(config["statistics"]["familywise_alpha"])
        for row in statistics
    )
    no_cell_regression = all(
        selector_cells[key] + float(config["gates"]["maximum_cell_loss_vs_keep"]) >= keep_cells[key]
        for key in keep_cells
    )
    test_gain = selector_macro - comparator_macro
    test_gate = (
        test_gain >= float(config["gates"]["minimum_test_macro_gain"])
        and selector_pooled["break_rate"] is not None
        and selector_pooled["break_rate"] <= float(config["gates"]["maximum_test_break_rate"])
        and no_cell_regression
        and all_statistical
    )
    shift_predictions = [row for row in predictions if row["split"] == "shift"]
    shift_primary = {
        method: _pooled_metric(shift_predictions, method, primary_budget)
        for method in methods
    }

    conclusion_path = evaluation_dir / "conclusion.md"
    conclusion = [
        "# ProActive Plan B conclusion",
        "",
        f"**Exploratory gate:** {'PASSED' if test_gate else 'FAILED'}",
        "",
        f"The validation-frozen comparator was `{comparator}`. At the primary budget B={primary_budget}, "
        f"the selector six-cell macro accuracy was {selector_macro:.4f}, versus {comparator_macro:.4f} "
        f"for the comparator and {keep_macro:.4f} for keep-original.",
        "",
        f"The paired primary gain over the frozen comparator was {test_gain:+.4f}. "
        f"Pooled BreakRate was {selector_pooled['break_rate']:.4f}.",
        "",
        "This is a cached-answer selection experiment. It does not establish that the learned ProActive "
        "acquisition policy improves answer correction, and held-out shift is descriptive because it was "
        "already inspected before this protocol was created.",
    ]
    conclusion_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = conclusion_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    temporary.replace(conclusion_path)

    report_payload = {
        "schema_version": "plan_b_final_report_v1",
        "status": "EXPLORATORY_GATE_PASSED" if test_gate else "EXPLORATORY_GATE_FAILED",
        "config_sha256": config_hash,
        "audit_report_sha256": audit["report_sha256"],
        "freeze_manifest_sha256": freeze["report_sha256"],
        "test_and_shift_opened_once": True,
        "post_evaluation_tuning_used": False,
        "primary": {
            "budget": primary_budget,
            "selector_six_cell_macro_accuracy": selector_macro,
            "comparator": comparator,
            "comparator_six_cell_macro_accuracy": comparator_macro,
            "keep_six_cell_macro_accuracy": keep_macro,
            "selector_gain_over_comparator": test_gain,
            "selector_pooled": selector_pooled,
            "cell_differences_vs_keep": {
                f"{dataset}|{model_id}": selector_cells[(dataset, model_id)] - keep_cells[(dataset, model_id)]
                for dataset, model_id in sorted(keep_cells)
            },
            "all_paired_tests_pass": all_statistical,
            "no_cell_regression_over_limit": no_cell_regression,
            "gate_passed": test_gate,
        },
        "reference_core_test_baselines_reproduced": reference_baselines,
        "descriptive_heldout_shift_at_primary_budget": {
            "budget": primary_budget,
            "metrics": shift_primary,
            "selector_accuracy_gain_vs_keep": (
                shift_primary["selector"]["final_accuracy"]
                - shift_primary["keep_original"]["final_accuracy"]
            ),
            "used_for_tuning_or_gate": False,
        },
        "statistics": statistics,
    }
    report = with_self_hash(report_payload)
    atomic_write_json(final_path, report)
    artifacts = _artifact_inventory(
        output_dir,
        [
            output_dir / "audit" / "audit_report.json",
            output_dir / "audit" / "input_inventory.csv",
            output_dir / "audit" / "cohort_index.csv",
            output_dir / "audit" / "image_exclusions.csv",
            output_dir / "audit" / "image_overlap_report.json",
            output_dir / "audit" / "unusable_probe_answers.csv",
            output_dir / "training" / "selector_model.json",
            output_dir / "training" / "environment_preflight.json",
            output_dir / "training" / "validation_predictions.csv",
            output_dir / "training" / "validation_metrics.csv",
            freeze_path,
            predictions_path,
            metrics_path,
            stats_path,
            oracle_path,
            conclusion_path,
            final_path,
        ],
    )
    manifest = with_self_hash(
        {
            "schema_version": "plan_b_run_manifest_v1",
            "experiment_id": config["experiment_id"],
            "config_sha256": config_hash,
            "implementation_git_revision": _git_revision(repo_root),
            "runtime_versions": _runtime_versions(),
            "artifacts": artifacts,
        }
    )
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return report


def validate_outputs(output_dir: Path, require_evaluation: bool) -> dict[str, Any]:
    errors: list[str] = []
    for relative in ("audit/audit_report.json", "training/freeze_manifest.json"):
        path = output_dir / relative
        if not path.is_file():
            errors.append(f"Missing {relative}")
            continue
        try:
            payload = load_json(path)
            verify_self_hash(payload)
            artifact_key = "audit_artifacts" if relative.startswith("audit/") else "training_artifacts"
            for artifact in payload.get(artifact_key, []):
                artifact_path = output_dir / artifact["path"]
                if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
                    errors.append(f"Artifact hash mismatch: {artifact['path']}")
        except PlanBError as exc:
            errors.append(f"{relative}: {exc}")
    if require_evaluation:
        for relative in ("evaluation/final_report.json", "run_manifest.json"):
            path = output_dir / relative
            if not path.is_file():
                errors.append(f"Missing {relative}")
                continue
            try:
                payload = load_json(path)
                verify_self_hash(payload)
                if relative == "run_manifest.json":
                    for artifact in payload.get("artifacts", []):
                        artifact_path = output_dir / artifact["path"]
                        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
                            errors.append(f"Artifact hash mismatch: {artifact['path']}")
            except PlanBError as exc:
                errors.append(f"{relative}: {exc}")
    return {
        "schema_version": "plan_b_validation_v1",
        "is_valid": not errors,
        "require_evaluation": require_evaluation,
        "errors": errors,
    }
