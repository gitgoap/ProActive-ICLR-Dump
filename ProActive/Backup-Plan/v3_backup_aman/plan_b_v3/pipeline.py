"""Plan B V3: grouped-OOF hard negatives and active verification."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from plan_b.core import (
    ABSTAIN,
    Candidate,
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
    record_gold_norm,
    recompute_clean_correctness,
    select_keep,
    sha256_file,
    verify_self_hash,
    with_self_hash,
)
from plan_b.pipeline import _slice_metrics, run_audit
from plan_b_v2.features import feature_names, feature_vector
from plan_b_v2.models import (
    ModelSpec,
    dependency_report,
    fit_estimator,
    load_estimator,
    positive_probability,
    save_estimator,
)


@dataclass
class CandidateExample:
    row: Mapping[str, Any]
    observations: Sequence[Any]
    candidates: Sequence[Candidate]
    candidate: Candidate
    matched_cost: int
    features: list[float]
    original_correct: int
    candidate_correct: int
    repair_label: int
    damage_label: int
    base_weight: float
    fold: int = -1

    @property
    def record_key(self) -> tuple[str, str]:
        return str(self.row["instance_id"]), str(self.row["model_id"])


PREDICTION_FIELDS = (
    "split", "dataset", "model_id", "instance_id", "group_id", "image_key", "budget", "method",
    "selected_source", "selected_norm_answer", "selected_raw_answer", "correct", "original_correct",
    "repair", "damage", "abstain", "correction_abstained", "candidate_count", "alternative_count",
    "base_probe_cost", "matched_cost", "verifier_probe", "verifier_acquired", "verifier_agreed", "operational_cost",
    "ensemble_min_repair_probability", "ensemble_max_damage_probability", "ensemble_min_safe_score",
)

OOF_FIELDS = (
    "instance_id", "model_id", "dataset", "image_key", "fold", "candidate_norm_answer",
    "repair_label", "damage_label", "base_weight", "models_proposing", "hard_negative",
    "xgboost_default_repair", "xgboost_default_damage",
    "lightgbm_default_repair", "lightgbm_default_damage",
    "mlp_64_32_regularized_repair", "mlp_64_32_regularized_damage",
)

LEADERBOARD_FIELDS = (
    "rank", "repair_probability_minimum", "damage_probability_maximum", "safe_score_minimum",
    "six_cell_macro_accuracy", "six_cell_macro_gain", "pooled_original_accuracy", "pooled_final_accuracy",
    "repairs", "damage", "fix_rate", "break_rate", "ensemble_proposals", "verifier_acquisitions",
    "switches", "eligible_under_break_constraint", "gate_passed",
)


def load_v3_config(path: Path, repo_root: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != "plan_b_v3_config_v1":
        raise PlanBError("Unsupported Plan B V3 config schema")
    pinned = (
        ("v1_backbone_config", "v1_backbone_config_sha256"),
        ("v2_feature_module", "v2_feature_module_sha256"),
        ("v2_model_module", "v2_model_module_sha256"),
    )
    for path_key, hash_key in pinned:
        target = (repo_root / str(config[path_key])).resolve()
        if not target.is_file():
            raise PlanBError(f"Missing pinned V3 dependency: {target}")
        actual = sha256_file(target)
        if actual != config[hash_key]:
            raise PlanBError(f"Pinned V3 dependency drift for {config[path_key]}: {actual}")
    if config.get("base_budget") != 2 or config.get("base_probes") != ["grounding", "blur"]:
        raise PlanBError("V3 base evidence contract drift")
    if config.get("verification_probe_candidates") != ["crop", "brightness", "noise"]:
        raise PlanBError("V3 verifier candidates drift")
    if config.get("verifier_selection", {}).get("blank_forbidden_as_primary_verifier") is not True:
        raise PlanBError("V3 must forbid blank as the primary verifier")
    if config.get("feature_set") != "structural_plus_cached_confidence_proxy_v2":
        raise PlanBError("V3 feature contract drift")
    feature_names(str(config["feature_set"]))
    models = ensemble_specs(config)
    if [model.family for model in models] != ["xgboost", "lightgbm", "mlp"]:
        raise PlanBError("V3 requires exactly XGBoost, LightGBM, and MLP")
    if int(config["oof"]["folds"]) != 5:
        raise PlanBError("V3 requires five grouped OOF folds")
    constraints = config.get("scientific_constraints", {})
    required_true = (
        "post_hoc_after_v1_and_v2_validation", "hard_negative_mining_uses_train_oof_only",
        "validation_selects_one_operating_point", "calibration_is_confirmation_only",
        "no_retuning_after_confirmation", "locked_test_requires_validation_and_confirmation",
        "cached_confidence_is_unverified_proxy", "one_additional_probe_only_on_unanimous_correction_proposals",
        "no_new_mllm_calls", "no_dataset_or_model_identity_features",
    )
    if any(constraints.get(key) is not True for key in required_true):
        raise PlanBError("V3 scientific constraints are incomplete")
    if constraints.get("test_or_shift_tuning") is not False:
        raise PlanBError("V3 must forbid test/shift tuning")
    if config["validation_search"].get("requires_unanimous_models") is not True:
        raise PlanBError("V3 validation must require unanimous models")
    if config["validation_search"].get("requires_verifier_agreement") is not True:
        raise PlanBError("V3 validation must require verifier agreement")
    if config["validation_search"].get("abstention_action") != "KEEP_ORIGINAL":
        raise PlanBError("V3 abstention must keep the original answer")
    for gate in (
        "maximum_validation_break_rate", "maximum_confirmation_break_rate", "maximum_test_break_rate"
    ):
        value = float(config["gates"][gate])
        if value < 0.0 or value > 0.01:
            raise PlanBError(f"V3 {gate} may not exceed 1%")
    return config


def ensemble_specs(config: Mapping[str, Any]) -> list[ModelSpec]:
    specs = [
        ModelSpec(str(item["name"]), str(item["family"]), dict(item.get("parameters", {})))
        for item in config["ensemble_models"]
    ]
    if len(specs) != 3 or len({spec.name for spec in specs}) != 3:
        raise PlanBError("V3 ensemble must contain three uniquely named models")
    return specs


def load_v1_config(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = (repo_root / str(config["v1_backbone_config"])).resolve()
    payload = load_json(path)
    if payload.get("schema_version") != "plan_b_config_v1":
        raise PlanBError("Unsupported V1 config for V3")
    return payload


def _stage_guard(path: Path, *, resume: bool, overwrite: bool) -> dict[str, Any] | None:
    if not path.exists() or overwrite:
        return None
    if not resume:
        raise PlanBError(f"Output exists; use --resume or --overwrite: {path}")
    payload = load_json(path)
    verify_self_hash(payload)
    return payload


def _artifact_inventory(output_dir: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        if not path.is_file():
            raise PlanBError(f"Missing expected V3 artifact: {path}")
        result.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return result


def _implementation_inventory(repo_root: Path) -> list[dict[str, Any]]:
    relative_paths = (
        "Backup-Plan/v3_backup_aman/config/plan_b_v3_config.json",
        "Backup-Plan/v3_backup_aman/plan_b_v3/pipeline.py",
        "Backup-Plan/v3_backup_aman/run_plan_b_v3.py",
        "Backup-Plan/v3_backup_aman/PLAN_B_V3_PROTOCOL.md",
    )
    inventory = []
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise PlanBError(f"Missing V3 implementation file: {relative}")
        inventory.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return inventory


def _verify_implementation_inventory(repo_root: Path, inventory: Sequence[Mapping[str, Any]]) -> None:
    if not inventory:
        raise PlanBError("Frozen V3 manifest is missing implementation provenance")
    for item in inventory:
        path = repo_root / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != int(item["bytes"]):
            raise PlanBError(f"Frozen V3 implementation drift: {item['path']}")


def _cohort_lookup(cohort: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["instance_id"]), str(row["model_id"])): row for row in cohort}


def _records_for_split(
    records: Sequence[Mapping[str, Any]], cohort: Sequence[Mapping[str, Any]], split: str, *, fit_only: bool
) -> list[Mapping[str, Any]]:
    lookup = _cohort_lookup(cohort)
    output = []
    for row in records:
        entry = lookup[(str(row["instance_id"]), str(row["model_id"]))]
        if str(entry["split"]) != split or int(entry["closed_answer_eligible"]) != 1:
            continue
        if fit_only and int(entry["proposed_fit_eligible"]) != 1:
            continue
        output.append(row)
    return output


def assign_grouped_folds(records: Sequence[Mapping[str, Any]], folds: int, seed: int) -> dict[str, int]:
    groups = sorted(
        {image_key(row) for row in records},
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest(),
    )
    if len(groups) < folds:
        raise PlanBError("Insufficient image groups for V3 OOF")
    return {group: index % folds for index, group in enumerate(groups)}


def _build_examples(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], fold_by_image: Mapping[str, int] | None,
    *, require_both_classes: bool = False,
) -> list[CandidateExample]:
    budget = int(config["base_budget"])
    feature_set = str(config["feature_set"])
    group_sizes = Counter(str(row["group_id"]) for row in records)
    pending = []
    for row in records:
        observations, cost = acquired_observations(row, budget, config["base_probes"])
        candidates = build_candidates(observations, cost)
        clean = observation_from_row(row, "clean")
        alternatives = [candidate for candidate in candidates if not clean.usable or candidate.norm_answer != clean.norm_answer]
        if not alternatives:
            continue
        original_correct = recompute_clean_correctness(row)
        gold = record_gold_norm(row)
        denominator = group_sizes[str(row["group_id"])] * len(alternatives)
        for candidate in alternatives:
            candidate_correct = int(candidate.norm_answer == gold)
            pending.append(
                CandidateExample(
                    row=row,
                    observations=observations,
                    candidates=candidates,
                    candidate=candidate,
                    matched_cost=cost,
                    features=list(feature_vector(feature_set, row, candidate, candidates, observations, cost)),
                    original_correct=original_correct,
                    candidate_correct=candidate_correct,
                    repair_label=int(not original_correct and candidate_correct),
                    damage_label=int(original_correct and not candidate_correct),
                    base_weight=1.0 / denominator,
                    fold=fold_by_image[image_key(row)] if fold_by_image is not None else -1,
                )
            )
    if not pending:
        raise PlanBError("V3 found no alternative candidate examples")
    mean_weight = sum(example.base_weight for example in pending) / len(pending)
    for example in pending:
        example.base_weight /= mean_weight
    if require_both_classes and {example.repair_label for example in pending} != {0, 1}:
        raise PlanBError("V3 repair target lacks both classes")
    if require_both_classes and {example.damage_label for example in pending} != {0, 1}:
        raise PlanBError("V3 damage target lacks both classes")
    return pending


def _as_array(values: Sequence[Any]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanBError("V3 requires NumPy") from exc
    return np.asarray(values, dtype=np.float64)


def _fit_pair(
    spec: ModelSpec,
    examples: Sequence[CandidateExample],
    weights: Sequence[float],
    seed: int,
) -> tuple[Any, Any]:
    features = _as_array([example.features for example in examples])
    repair = [example.repair_label for example in examples]
    damage = [example.damage_label for example in examples]
    weight_array = _as_array(weights)
    return (
        fit_estimator(spec, features, repair, weight_array, seed=seed),
        fit_estimator(spec, features, damage, weight_array, seed=seed),
    )


def _weighted_examples(
    examples: Sequence[CandidateExample], config: Mapping[str, Any], hard_flags: Sequence[bool] | None = None
) -> list[float]:
    settings = config["cost_sensitive_training"]
    damage_weight = float(settings["damage_example_weight"])
    hard_multiplier = float(settings["hard_negative_additional_multiplier"])
    if hard_flags is not None and len(hard_flags) != len(examples):
        raise PlanBError("V3 hard-negative flag length mismatch")
    values = []
    for index, example in enumerate(examples):
        weight = example.base_weight * (damage_weight if example.damage_label else 1.0)
        if hard_flags is not None and hard_flags[index]:
            weight *= hard_multiplier
        values.append(weight)
    mean = sum(values) / len(values)
    return [value / mean for value in values]


def _oof_predictions(
    examples: Sequence[CandidateExample], config: Mapping[str, Any],
    hard_flags: Sequence[bool] | None = None,
) -> dict[str, dict[str, list[float]]]:
    folds = int(config["oof"]["folds"])
    output = {
        spec.name: {"repair": [math.nan] * len(examples), "damage": [math.nan] * len(examples)}
        for spec in ensemble_specs(config)
    }
    base_weights = _weighted_examples(examples, config, hard_flags)
    for fold in range(folds):
        train_indexes = [index for index, example in enumerate(examples) if example.fold != fold]
        hold_indexes = [index for index, example in enumerate(examples) if example.fold == fold]
        if not train_indexes or not hold_indexes:
            raise PlanBError(f"Empty V3 OOF fold {fold}")
        train_examples = [examples[index] for index in train_indexes]
        train_weights = [base_weights[index] for index in train_indexes]
        hold_features = [examples[index].features for index in hold_indexes]
        for spec in ensemble_specs(config):
            repair_model, damage_model = _fit_pair(
                spec, train_examples, train_weights, int(config["seed"]) + fold
            )
            repair = positive_probability(repair_model, hold_features)
            damage = positive_probability(damage_model, hold_features)
            for position, index in enumerate(hold_indexes):
                output[spec.name]["repair"][index] = repair[position]
                output[spec.name]["damage"][index] = damage[position]
    for model in output.values():
        for target in ("repair", "damage"):
            if any(not math.isfinite(value) for value in model[target]):
                raise PlanBError("Incomplete V3 OOF predictions")
    return output


def _qualifies(repair: float, damage: float, repair_minimum: float, damage_maximum: float, safe_minimum: float) -> bool:
    return repair >= repair_minimum and damage <= damage_maximum and repair - damage >= safe_minimum


def mine_hard_negatives(
    examples: Sequence[CandidateExample], oof: Mapping[str, Mapping[str, Sequence[float]]], config: Mapping[str, Any]
) -> tuple[list[bool], list[dict[str, Any]]]:
    definition = config["cost_sensitive_training"]["hard_negative_definition"]
    minimum = int(definition["minimum_models_proposing"])
    flags = []
    rows = []
    names = [spec.name for spec in ensemble_specs(config)]
    for index, example in enumerate(examples):
        proposing = sum(
            _qualifies(
                float(oof[name]["repair"][index]),
                float(oof[name]["damage"][index]),
                float(definition["repair_probability_minimum"]),
                float(definition["damage_probability_maximum"]),
                float(definition["safe_score_minimum"]),
            )
            for name in names
        )
        hard = bool(example.damage_label and proposing >= minimum)
        flags.append(hard)
        row = {
            "instance_id": example.row["instance_id"],
            "model_id": example.row["model_id"],
            "dataset": example.row["dataset"],
            "image_key": image_key(example.row),
            "fold": example.fold,
            "candidate_norm_answer": example.candidate.norm_answer,
            "repair_label": example.repair_label,
            "damage_label": example.damage_label,
            "base_weight": example.base_weight,
            "models_proposing": proposing,
            "hard_negative": int(hard),
        }
        for name in names:
            row[f"{name}_repair"] = oof[name]["repair"][index]
            row[f"{name}_damage"] = oof[name]["damage"][index]
        rows.append(row)
    return flags, rows


def _oof_rows_with_frozen_hard_flags(
    examples: Sequence[CandidateExample],
    oof: Mapping[str, Mapping[str, Sequence[float]]],
    hard_flags: Sequence[bool],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if len(examples) != len(hard_flags):
        raise PlanBError("V3 hardened OOF flag length mismatch")
    definition = config["cost_sensitive_training"]["hard_negative_definition"]
    names = [spec.name for spec in ensemble_specs(config)]
    rows = []
    for index, example in enumerate(examples):
        proposing = sum(
            _qualifies(
                float(oof[name]["repair"][index]),
                float(oof[name]["damage"][index]),
                float(definition["repair_probability_minimum"]),
                float(definition["damage_probability_maximum"]),
                float(definition["safe_score_minimum"]),
            )
            for name in names
        )
        row = {
            "instance_id": example.row["instance_id"],
            "model_id": example.row["model_id"],
            "dataset": example.row["dataset"],
            "image_key": image_key(example.row),
            "fold": example.fold,
            "candidate_norm_answer": example.candidate.norm_answer,
            "repair_label": example.repair_label,
            "damage_label": example.damage_label,
            "base_weight": example.base_weight,
            "models_proposing": proposing,
            "hard_negative": int(hard_flags[index]),
        }
        for name in names:
            row[f"{name}_repair"] = oof[name]["repair"][index]
            row[f"{name}_damage"] = oof[name]["damage"][index]
        rows.append(row)
    return rows


def _score_examples(
    examples: Sequence[CandidateExample], models: Mapping[str, tuple[Any, Any]], config: Mapping[str, Any]
) -> dict[str, dict[str, list[float]]]:
    features = [example.features for example in examples]
    output: dict[str, dict[str, list[float]]] = {}
    for spec in ensemble_specs(config):
        repair_model, damage_model = models[spec.name]
        output[spec.name] = {
            "repair": positive_probability(repair_model, features),
            "damage": positive_probability(damage_model, features),
        }
    return output


def _unanimous_proposal(
    indexes: Sequence[int],
    examples: Sequence[CandidateExample],
    scores: Mapping[str, Mapping[str, Sequence[float]]],
    config: Mapping[str, Any],
    *,
    repair_minimum: float,
    damage_maximum: float,
    safe_minimum: float,
) -> tuple[CandidateExample | None, float | None, float | None, float | None]:
    choices = []
    names = [spec.name for spec in ensemble_specs(config)]
    for name in names:
        eligible = []
        for index in indexes:
            repair = float(scores[name]["repair"][index])
            damage = float(scores[name]["damage"][index])
            if _qualifies(repair, damage, repair_minimum, damage_maximum, safe_minimum):
                candidate = examples[index].candidate
                eligible.append((index, repair - damage, repair, damage, candidate.earliest_rank, candidate.norm_answer))
        if not eligible:
            return None, None, None, None
        eligible.sort(key=lambda value: (-value[1], -value[2], value[3], value[4], value[5]))
        choices.append(eligible[0])
    answers = {examples[choice[0]].candidate.norm_answer for choice in choices}
    if len(answers) != 1:
        return None, None, None, None
    selected_index = choices[0][0]
    selected_answer = examples[selected_index].candidate.norm_answer
    corresponding = []
    for name in names:
        index = next(index for index in indexes if examples[index].candidate.norm_answer == selected_answer)
        corresponding.append(
            (
                float(scores[name]["repair"][index]),
                float(scores[name]["damage"][index]),
            )
        )
    return (
        examples[selected_index],
        min(value[0] for value in corresponding),
        max(value[1] for value in corresponding),
        min(value[0] - value[1] for value in corresponding),
    )


def _prediction_row(
    record: Mapping[str, Any], selection: Selection, *, method: str, candidate_count: int,
    alternative_count: int, base_probe_cost: int, verifier_probe: str | None,
    verifier_acquired: bool, verifier_agreed: bool, correction_abstained: bool,
    min_repair: float | None = None, max_damage: float | None = None, min_safe: float | None = None,
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
        "budget": 2,
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
        "candidate_count": candidate_count,
        "alternative_count": alternative_count,
        "base_probe_cost": base_probe_cost,
        "matched_cost": base_probe_cost,
        "verifier_probe": verifier_probe,
        "verifier_acquired": int(verifier_acquired),
        "verifier_agreed": int(verifier_agreed),
        "operational_cost": base_probe_cost + int(verifier_acquired),
        "ensemble_min_repair_probability": min_repair,
        "ensemble_max_damage_probability": max_damage,
        "ensemble_min_safe_score": min_safe,
    }


def _keep_predictions(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        clean = observation_from_row(record, "clean")
        selection = Selection("clean" if clean.usable else "abstain", clean.norm_answer if clean.usable else ABSTAIN, clean.raw_answer if clean.usable else "")
        output.append(
            _prediction_row(
                record, selection, method="keep_original", candidate_count=int(clean.usable), alternative_count=0,
                base_probe_cost=0, verifier_probe=None, verifier_acquired=False, verifier_agreed=False,
                correction_abstained=True,
            )
        )
    return output


def _ensemble_predictions(
    records: Sequence[Mapping[str, Any]],
    examples: Sequence[CandidateExample],
    scores: Mapping[str, Mapping[str, Sequence[float]]],
    config: Mapping[str, Any],
    *,
    verifier_probe: str,
    repair_minimum: float,
    damage_maximum: float,
    safe_minimum: float,
    method: str = "v3_selector",
) -> list[dict[str, Any]]:
    by_record: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_record[example.record_key].append(index)
    output = []
    for record in records:
        key = (str(record["instance_id"]), str(record["model_id"]))
        indexes = by_record.get(key, [])
        observations, base_cost = acquired_observations(record, int(config["base_budget"]), config["base_probes"])
        candidates = build_candidates(observations, base_cost)
        proposal, min_repair, max_damage, min_safe = _unanimous_proposal(
            indexes, examples, scores, config,
            repair_minimum=repair_minimum,
            damage_maximum=damage_maximum,
            safe_minimum=safe_minimum,
        ) if indexes else (None, None, None, None)
        verifier_acquired = proposal is not None
        verifier_agreed = False
        selection = select_keep(observations)
        if proposal is not None:
            verifier = observation_from_row(record, verifier_probe)
            verifier_agreed = bool(verifier.usable and verifier.norm_answer == proposal.candidate.norm_answer)
            if verifier_agreed:
                selection = Selection(
                    proposal.candidate.sources[0],
                    proposal.candidate.norm_answer,
                    proposal.candidate.raw_answer,
                )
        output.append(
            _prediction_row(
                record, selection, method=method, candidate_count=len(candidates), alternative_count=len(indexes),
                base_probe_cost=base_cost, verifier_probe=verifier_probe, verifier_acquired=verifier_acquired,
                verifier_agreed=verifier_agreed, correction_abstained=not verifier_agreed,
                min_repair=min_repair, max_damage=max_damage, min_safe=min_safe,
            )
        )
    return output


def _six_cell_macro(predictions: Sequence[Mapping[str, Any]], method: str) -> float:
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        if str(row["method"]) == method:
            cells[(str(row["dataset"]), str(row["model_id"]))].append(row)
    if len(cells) != 6 or any(not rows for rows in cells.values()):
        raise PlanBError(f"V3 expected six dataset/model cells for {method}, found {len(cells)}")
    return sum(metric_summary(rows)["final_accuracy"] for rows in cells.values()) / 6.0


def _pooled(predictions: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    rows = [row for row in predictions if str(row["method"]) == method]
    return metric_summary(rows)


def select_verifier_from_oof(
    records: Sequence[Mapping[str, Any]],
    examples: Sequence[CandidateExample],
    oof: Mapping[str, Mapping[str, Sequence[float]]],
    config: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    settings = config["verifier_selection"]
    candidates = []
    for order, verifier in enumerate(config["verification_probe_candidates"]):
        predictions = _ensemble_predictions(
            records, examples, oof, config,
            verifier_probe=str(verifier),
            repair_minimum=float(settings["repair_probability_minimum"]),
            damage_maximum=float(settings["damage_probability_maximum"]),
            safe_minimum=float(settings["safe_score_minimum"]),
            method="oof_selector",
        )
        pooled = _pooled(predictions, "oof_selector")
        utility = int(pooled["repairs"]) - 10 * int(pooled["damage"])
        candidates.append(
            {
                "verifier_probe": verifier,
                "configured_order": order,
                "repairs": pooled["repairs"],
                "damage": pooled["damage"],
                "fix_rate": pooled["fix_rate"],
                "break_rate": pooled["break_rate"],
                "ensemble_proposals": sum(int(row["verifier_acquired"]) for row in predictions),
                "switches": sum(int(row["verifier_agreed"]) for row in predictions),
                "damage_adjusted_utility": utility,
            }
        )
    candidates.sort(
        key=lambda row: (
            -int(row["damage_adjusted_utility"]),
            -int(row["repairs"]),
            int(row["damage"]),
            -int(row["switches"]),
            int(row["configured_order"]),
        )
    )
    return str(candidates[0]["verifier_probe"]), candidates


def _leaderboard_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["six_cell_macro_gain"]),
        -int(row["repairs"]),
        int(row["damage"]),
        -float(row["repair_probability_minimum"]),
        float(row["damage_probability_maximum"]),
        -float(row["safe_score_minimum"]),
    )


def _search_validation(
    records: Sequence[Mapping[str, Any]],
    examples: Sequence[CandidateExample],
    scores: Mapping[str, Mapping[str, Sequence[float]]],
    verifier: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keep = _keep_predictions(records)
    keep_macro = _six_cell_macro(keep, "keep_original")
    maximum_break = float(config["gates"]["maximum_validation_break_rate"])
    minimum_gain = float(config["gates"]["minimum_validation_six_cell_macro_gain"])
    minimum_repairs = int(config["gates"]["minimum_validation_repairs"])
    tolerance = float(config["validation_search"]["tie_tolerance"])
    leaderboard = []
    for repair_minimum in config["validation_search"]["repair_probability_minimums"]:
        for damage_maximum in config["validation_search"]["damage_probability_maximums"]:
            for safe_minimum in config["validation_search"]["safe_score_minimums"]:
                predictions = _ensemble_predictions(
                    records, examples, scores, config,
                    verifier_probe=verifier,
                    repair_minimum=float(repair_minimum),
                    damage_maximum=float(damage_maximum),
                    safe_minimum=float(safe_minimum),
                )
                macro = _six_cell_macro(predictions, "v3_selector")
                pooled = _pooled(predictions, "v3_selector")
                gain = macro - keep_macro
                break_rate = pooled["break_rate"]
                eligible = break_rate is not None and break_rate <= maximum_break + tolerance
                gate = eligible and gain + tolerance >= minimum_gain and int(pooled["repairs"]) >= minimum_repairs
                leaderboard.append(
                    {
                        "repair_probability_minimum": float(repair_minimum),
                        "damage_probability_maximum": float(damage_maximum),
                        "safe_score_minimum": float(safe_minimum),
                        "six_cell_macro_accuracy": macro,
                        "six_cell_macro_gain": gain,
                        "pooled_original_accuracy": pooled["original_accuracy"],
                        "pooled_final_accuracy": pooled["final_accuracy"],
                        "repairs": pooled["repairs"],
                        "damage": pooled["damage"],
                        "fix_rate": pooled["fix_rate"],
                        "break_rate": break_rate,
                        "ensemble_proposals": sum(int(row["verifier_acquired"]) for row in predictions),
                        "verifier_acquisitions": sum(int(row["verifier_acquired"]) for row in predictions),
                        "switches": sum(int(row["verifier_agreed"]) for row in predictions),
                        "eligible_under_break_constraint": eligible,
                        "gate_passed": gate,
                    }
                )
    eligible = [row for row in leaderboard if row["eligible_under_break_constraint"]]
    if not eligible:
        raise PlanBError("No V3 validation point satisfies the BreakRate constraint")
    eligible.sort(key=_leaderboard_key)
    selected = dict(eligible[0])
    ranked = sorted(leaderboard, key=lambda row: (not row["eligible_under_break_constraint"], *_leaderboard_key(row)))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    selected_predictions = _ensemble_predictions(
        records, examples, scores, config,
        verifier_probe=verifier,
        repair_minimum=float(selected["repair_probability_minimum"]),
        damage_maximum=float(selected["damage_probability_maximum"]),
        safe_minimum=float(selected["safe_score_minimum"]),
    )
    return selected, ranked, keep + selected_predictions, _slice_metrics(keep + selected_predictions)


def _fixed_split_evaluation(
    records: Sequence[Mapping[str, Any]],
    examples: Sequence[CandidateExample],
    scores: Mapping[str, Mapping[str, Sequence[float]]],
    verifier: str,
    operating: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    keep = _keep_predictions(records)
    selected = _ensemble_predictions(
        records,
        examples,
        scores,
        config,
        verifier_probe=verifier,
        repair_minimum=float(operating["repair_probability_minimum"]),
        damage_maximum=float(operating["damage_probability_maximum"]),
        safe_minimum=float(operating["safe_score_minimum"]),
    )
    predictions = keep + selected
    pooled_keep = _pooled(predictions, "keep_original")
    pooled_selected = _pooled(predictions, "v3_selector")
    return predictions, _slice_metrics(predictions), {
        "keep_original": pooled_keep,
        "v3_selector": pooled_selected,
        "pooled_accuracy_gain": pooled_selected["final_accuracy"] - pooled_keep["final_accuracy"],
        "verifier_acquisitions": sum(int(row["verifier_acquired"]) for row in selected),
        "switches": sum(int(row["verifier_agreed"]) for row in selected),
    }


def _confirmation_gate(summary: Mapping[str, Any], macro_gain: float, config: Mapping[str, Any]) -> bool:
    selected = summary["v3_selector"]
    break_rate = selected["break_rate"]
    tolerance = float(config["validation_search"]["tie_tolerance"])
    return bool(
        break_rate is not None
        and float(break_rate) <= float(config["gates"]["maximum_confirmation_break_rate"]) + tolerance
        and macro_gain + tolerance >= float(config["gates"]["minimum_confirmation_six_cell_macro_gain"])
        and int(selected["repairs"]) >= int(config["gates"]["minimum_confirmation_repairs"])
    )


def _bootstrap_delta(
    predictions: Sequence[Mapping[str, Any]], split: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanBError("V3 bootstrap requires NumPy") from exc
    relevant = [row for row in predictions if str(row["split"]) == split]
    selected = {
        (str(row["instance_id"]), str(row["model_id"])): row
        for row in relevant if row["method"] == "v3_selector"
    }
    kept = {
        (str(row["instance_id"]), str(row["model_id"])): row
        for row in relevant if row["method"] == "keep_original"
    }
    if selected.keys() != kept.keys() or not selected:
        raise PlanBError(f"Unpaired V3 bootstrap rows for {split}")
    clusters: dict[str, list[float]] = defaultdict(list)
    for key, row in selected.items():
        clusters[str(row["image_key"])].append(float(row["correct"]) - float(kept[key]["correct"]))
    names = sorted(clusters)
    values = [sum(clusters[name]) for name in names]
    counts = [len(clusters[name]) for name in names]
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
        "method": "v3_selector",
        "comparator": "keep_original",
        "model_example_pairs": len(selected),
        "image_clusters": len(names),
        "pooled_accuracy_delta": sum(values) / sum(counts),
        "cluster_bootstrap_ci": [
            float(np.quantile(draws, alpha)),
            float(np.quantile(draws, 1.0 - alpha)),
        ],
    }


def _load_frozen_models(output_dir: Path, freeze: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    models = {}
    entries = freeze.get("ensemble_models", {})
    expected_names = [spec.name for spec in ensemble_specs(config)]
    if sorted(entries) != sorted(expected_names):
        raise PlanBError("Frozen V3 ensemble membership mismatch")
    for name in expected_names:
        entry = entries[name]
        repair_path = output_dir / str(entry["repair"]["path"])
        damage_path = output_dir / str(entry["damage"]["path"])
        if sha256_file(repair_path) != entry["repair"]["sha256"]:
            raise PlanBError(f"Frozen V3 repair-model hash mismatch: {name}")
        if sha256_file(damage_path) != entry["damage"]["sha256"]:
            raise PlanBError(f"Frozen V3 damage-model hash mismatch: {name}")
        models[name] = (load_estimator(repair_path), load_estimator(damage_path))
    return models


def run_prepare(
    *, repo_root: Path, config: Mapping[str, Any], output_dir: Path, resume: bool, overwrite: bool
) -> dict[str, Any]:
    v1_config = load_v1_config(repo_root, config)
    dependencies = dependency_report(ensemble_specs(config))
    if dependencies["missing"]:
        raise PlanBError("Missing V3 dependencies: " + ", ".join(dependencies["missing"]))
    audit = run_audit(
        repo_root=repo_root,
        config=v1_config,
        output_dir=output_dir,
        resume=resume,
        overwrite=overwrite,
    )
    if not audit.get("is_valid"):
        raise PlanBError("V3 preparation requires a complete valid input audit")
    training_dir = output_dir / "training"
    freeze_path = training_dir / "freeze_manifest.json"
    existing = _stage_guard(freeze_path, resume=resume, overwrite=overwrite)
    config_hash = canonical_json_sha256(config)
    if existing is not None:
        if existing.get("config_sha256") != config_hash or existing.get("v1_audit_report_sha256") != audit["report_sha256"]:
            raise PlanBError("Unsafe V3 preparation resume refused: provenance drift")
        for artifact in existing.get("training_artifacts", []):
            path = output_dir / artifact["path"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise PlanBError(f"Unsafe V3 resume refused: output drift for {artifact['path']}")
        _verify_implementation_inventory(repo_root, existing.get("implementation_inventory", []))
        return existing

    records, inventory = load_and_verify_inputs(repo_root, v1_config)
    cohort, _ = build_cohort_index(records)
    train_records = _records_for_split(records, cohort, "train", fit_only=True)
    validation_records = _records_for_split(records, cohort, "val", fit_only=True)
    fold_by_image = assign_grouped_folds(train_records, int(config["oof"]["folds"]), int(config["seed"]))
    train_examples = _build_examples(
        train_records, config, fold_by_image, require_both_classes=True
    )
    preliminary_oof = _oof_predictions(train_examples, config)
    hard_flags, oof_rows = mine_hard_negatives(train_examples, preliminary_oof, config)
    hardened_oof = _oof_predictions(train_examples, config, hard_flags)
    hardened_oof_rows = _oof_rows_with_frozen_hard_flags(
        train_examples, hardened_oof, hard_flags, config
    )
    selected_verifier, verifier_rows = select_verifier_from_oof(
        train_records, train_examples, hardened_oof, config
    )

    fold_path = training_dir / "oof_fold_assignments.csv"
    oof_path = training_dir / "oof_candidate_predictions.csv"
    hardened_oof_path = training_dir / "hardened_oof_candidate_predictions.csv"
    verifier_csv_path = training_dir / "verifier_selection.csv"
    verifier_json_path = training_dir / "verifier_selection.json"
    environment_path = training_dir / "environment_preflight.json"
    atomic_write_csv(
        fold_path,
        [{"image_key": key, "fold": fold_by_image[key]} for key in sorted(fold_by_image)],
        ("image_key", "fold"),
    )
    atomic_write_csv(oof_path, oof_rows, OOF_FIELDS)
    atomic_write_csv(hardened_oof_path, hardened_oof_rows, OOF_FIELDS)
    atomic_write_csv(verifier_csv_path, verifier_rows, tuple(verifier_rows[0].keys()))
    atomic_write_json(
        verifier_json_path,
        with_self_hash(
            {
                "schema_version": "plan_b_v3_verifier_selection_v1",
                "selection_data": "grouped_out_of_fold_train_predictions_only",
                "selected_verifier": selected_verifier,
                "candidates": verifier_rows,
                "blank_forbidden": True,
            }
        ),
    )
    atomic_write_json(
        environment_path,
        with_self_hash({"schema_version": "plan_b_v3_environment_v1", **dependencies}),
    )

    weights = _weighted_examples(train_examples, config, hard_flags)
    model_dir = training_dir / "models"
    fitted: dict[str, tuple[Any, Any]] = {}
    model_entries: dict[str, Any] = {}
    artifacts: list[Path] = [
        fold_path, oof_path, hardened_oof_path, verifier_csv_path, verifier_json_path, environment_path
    ]
    for spec in ensemble_specs(config):
        pair = _fit_pair(spec, train_examples, weights, int(config["seed"]))
        repair_path = model_dir / f"{spec.name}.repair.joblib"
        damage_path = model_dir / f"{spec.name}.damage.joblib"
        save_estimator(pair[0], repair_path)
        save_estimator(pair[1], damage_path)
        fitted[spec.name] = pair
        model_entries[spec.name] = {
            "family": spec.family,
            "parameters": dict(spec.parameters),
            "repair": {"path": repair_path.relative_to(output_dir).as_posix(), "sha256": sha256_file(repair_path)},
            "damage": {"path": damage_path.relative_to(output_dir).as_posix(), "sha256": sha256_file(damage_path)},
        }
        artifacts.extend([repair_path, damage_path])

    validation_examples = _build_examples(validation_records, config, None)
    validation_scores = _score_examples(validation_examples, fitted, config)
    selected, leaderboard, validation_predictions, validation_metrics = _search_validation(
        validation_records, validation_examples, validation_scores, selected_verifier, config
    )
    leaderboard_path = training_dir / "validation_leaderboard.csv"
    validation_prediction_path = training_dir / "validation_predictions.csv"
    validation_metric_path = training_dir / "validation_metrics.csv"
    atomic_write_csv(leaderboard_path, leaderboard, LEADERBOARD_FIELDS)
    atomic_write_csv(validation_prediction_path, validation_predictions, PREDICTION_FIELDS)
    atomic_write_csv(validation_metric_path, validation_metrics, tuple(validation_metrics[0].keys()))
    artifacts.extend([leaderboard_path, validation_prediction_path, validation_metric_path])

    validation_passed = bool(selected["gate_passed"])
    confirmation: dict[str, Any] = {
        "accessed": False,
        "gate_passed": False,
        "reason": "validation_gate_failed",
    }
    if validation_passed:
        calibration_records = _records_for_split(records, cohort, "cal", fit_only=False)
        calibration_examples = _build_examples(calibration_records, config, None)
        calibration_scores = _score_examples(calibration_examples, fitted, config)
        calibration_predictions, calibration_metrics, calibration_summary = _fixed_split_evaluation(
            calibration_records,
            calibration_examples,
            calibration_scores,
            selected_verifier,
            selected,
            config,
        )
        keep_macro = _six_cell_macro(calibration_predictions, "keep_original")
        selected_macro = _six_cell_macro(calibration_predictions, "v3_selector")
        macro_gain = selected_macro - keep_macro
        confirmation_passed = _confirmation_gate(calibration_summary, macro_gain, config)
        confirmation_prediction_path = training_dir / "confirmation_predictions.csv"
        confirmation_metric_path = training_dir / "confirmation_metrics.csv"
        confirmation_report_path = training_dir / "confirmation_report.json"
        atomic_write_csv(confirmation_prediction_path, calibration_predictions, PREDICTION_FIELDS)
        atomic_write_csv(confirmation_metric_path, calibration_metrics, tuple(calibration_metrics[0].keys()))
        confirmation = {
            "accessed": True,
            "role": "no_retuning_confirmation_only",
            "gate_passed": confirmation_passed,
            "selected_operating_point_unchanged": True,
            "keep_original_six_cell_macro_accuracy": keep_macro,
            "v3_selector_six_cell_macro_accuracy": selected_macro,
            "six_cell_macro_gain": macro_gain,
            **calibration_summary,
        }
        atomic_write_json(
            confirmation_report_path,
            with_self_hash({"schema_version": "plan_b_v3_confirmation_v1", **confirmation}),
        )
        artifacts.extend([confirmation_prediction_path, confirmation_metric_path, confirmation_report_path])

    if not validation_passed:
        status = "FROZEN_VALIDATION_GATE_FAILED"
    elif not confirmation["gate_passed"]:
        status = "FROZEN_CONFIRMATION_GATE_FAILED"
    else:
        status = "FROZEN_CONFIRMATION_GATE_PASSED"
    freeze = with_self_hash(
        {
            "schema_version": "plan_b_v3_freeze_v1",
            "status": status,
            "experiment_status": config["status"],
            "post_hoc_exploratory": True,
            "config_sha256": config_hash,
            "v1_backbone_config_sha256": config["v1_backbone_config_sha256"],
            "v2_feature_module_sha256": config["v2_feature_module_sha256"],
            "v2_model_module_sha256": config["v2_model_module_sha256"],
            "v1_audit_report_sha256": audit["report_sha256"],
            "implementation_inventory": _implementation_inventory(repo_root),
            "input_inventory": inventory,
            "dependencies": dependencies,
            "training": {
                "eligible_records": len(train_records),
                "candidate_examples": len(train_examples),
                "image_groups": len(fold_by_image),
                "oof_folds": int(config["oof"]["folds"]),
                "oof_passes": ["preliminary_hard_negative_mining", "hard_negative_weighted_selection"],
                "hard_negative_count": sum(hard_flags),
                "damage_example_weight": config["cost_sensitive_training"]["damage_example_weight"],
                "hard_negative_additional_multiplier": config["cost_sensitive_training"]["hard_negative_additional_multiplier"],
            },
            "selected_verifier": selected_verifier,
            "verifier_selection_data": "grouped_out_of_fold_train_predictions_only",
            "ensemble_models": model_entries,
            "selected_operating_point": selected,
            "validation": {
                "eligible_records": len(validation_records),
                "candidate_operating_points": len(leaderboard),
                "gate_passed": validation_passed,
                "keep_original_six_cell_macro_accuracy": _six_cell_macro(validation_predictions, "keep_original"),
                "v3_selector_six_cell_macro_accuracy": _six_cell_macro(validation_predictions, "v3_selector"),
            },
            "confirmation": confirmation,
            "calibration_split_accessed": bool(confirmation["accessed"]),
            "test_split_accessed": False,
            "shift_split_accessed": False,
            "no_retuning_after_validation": True,
            "new_mllm_calls": 0,
            "cached_confidence_warning": (
                "Probe confidence values are reconstructed from cached shifts and are an unverified proxy."
            ),
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
        raise PlanBError("Locked V3 evaluation requires --confirm-locked-evaluation")
    freeze_path = output_dir / "training" / "freeze_manifest.json"
    if not freeze_path.is_file():
        raise PlanBError("Run V3 prepare before locked evaluation")
    freeze = load_json(freeze_path)
    verify_self_hash(freeze)
    if freeze.get("status") != "FROZEN_CONFIRMATION_GATE_PASSED":
        raise PlanBError("V3 validation/confirmation gate failed; locked test and shift remain closed")
    if freeze.get("config_sha256") != canonical_json_sha256(config):
        raise PlanBError("Frozen V3 config hash mismatch")
    _verify_implementation_inventory(repo_root, freeze.get("implementation_inventory", []))
    report_path = output_dir / "locked_evaluation" / "evaluation_report.json"
    existing = _stage_guard(report_path, resume=resume, overwrite=overwrite)
    if existing is not None:
        if existing.get("freeze_manifest_sha256") != freeze["report_sha256"]:
            raise PlanBError("Unsafe V3 evaluation resume refused: freeze drift")
        for artifact in existing.get("artifacts", []):
            path = output_dir / artifact["path"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise PlanBError(f"Unsafe V3 evaluation resume refused: {artifact['path']}")
        return existing

    v1_config = load_v1_config(repo_root, config)
    records, _ = load_and_verify_inputs(repo_root, v1_config)
    cohort, _ = build_cohort_index(records)
    models = _load_frozen_models(output_dir, freeze, config)
    operating = freeze["selected_operating_point"]
    verifier = str(freeze["selected_verifier"])
    all_predictions: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split in ("test", "shift"):
        split_records = _records_for_split(records, cohort, split, fit_only=False)
        split_examples = _build_examples(split_records, config, None)
        scores = _score_examples(split_examples, models, config)
        predictions, _, summary = _fixed_split_evaluation(
            split_records, split_examples, scores, verifier, operating, config
        )
        if split == "test":
            keep_macro = _six_cell_macro(predictions, "keep_original")
            selected_macro = _six_cell_macro(predictions, "v3_selector")
            summary["keep_original_six_cell_macro_accuracy"] = keep_macro
            summary["v3_selector_six_cell_macro_accuracy"] = selected_macro
            summary["six_cell_macro_gain"] = selected_macro - keep_macro
            break_rate = summary["v3_selector"]["break_rate"]
            summary["test_gate_passed"] = bool(
                break_rate is not None
                and float(break_rate) <= float(config["gates"]["maximum_test_break_rate"])
                and summary["six_cell_macro_gain"] >= float(config["gates"]["minimum_test_six_cell_macro_gain"])
            )
        summaries[split] = summary
        all_predictions.extend(predictions)

    metrics = _slice_metrics(all_predictions)
    evaluation_dir = output_dir / "locked_evaluation"
    prediction_path = evaluation_dir / "predictions.csv"
    metric_path = evaluation_dir / "metrics.csv"
    bootstrap_path = evaluation_dir / "cluster_bootstrap.json"
    atomic_write_csv(prediction_path, all_predictions, PREDICTION_FIELDS)
    atomic_write_csv(metric_path, metrics, tuple(metrics[0].keys()))
    atomic_write_json(
        bootstrap_path,
        with_self_hash(
            {
                "schema_version": "plan_b_v3_bootstrap_v1",
                "comparisons": [_bootstrap_delta(all_predictions, split, config) for split in ("test", "shift")],
            }
        ),
    )
    report = with_self_hash(
        {
            "schema_version": "plan_b_v3_locked_evaluation_v1",
            "status": "COMPLETE_POST_HOC_EXPLORATORY",
            "freeze_manifest_sha256": freeze["report_sha256"],
            "selected_verifier": verifier,
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


def validate_outputs(output_dir: Path, require_evaluation: bool, repo_root: Path | None = None) -> dict[str, Any]:
    errors = []
    freeze = None
    freeze_path = output_dir / "training" / "freeze_manifest.json"
    if not freeze_path.is_file():
        errors.append("Missing V3 freeze manifest")
    else:
        try:
            freeze = load_json(freeze_path)
            verify_self_hash(freeze)
            if repo_root is not None:
                _verify_implementation_inventory(repo_root, freeze.get("implementation_inventory", []))
            for artifact in freeze.get("training_artifacts", []):
                path = output_dir / artifact["path"]
                if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                    errors.append(f"Training artifact drift: {artifact['path']}")
        except (PlanBError, OSError, KeyError) as exc:
            errors.append(str(exc))
    evaluation = None
    report_path = output_dir / "locked_evaluation" / "evaluation_report.json"
    if require_evaluation or report_path.is_file():
        if not report_path.is_file():
            errors.append("Missing required V3 locked evaluation report")
        else:
            try:
                evaluation = load_json(report_path)
                verify_self_hash(evaluation)
                if freeze is None or evaluation.get("freeze_manifest_sha256") != freeze.get("report_sha256"):
                    errors.append("V3 evaluation does not match the freeze manifest")
                for artifact in evaluation.get("artifacts", []):
                    path = output_dir / artifact["path"]
                    if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                        errors.append(f"Evaluation artifact drift: {artifact['path']}")
            except (PlanBError, OSError, KeyError) as exc:
                errors.append(str(exc))
    return {
        "mode": "validate",
        "is_valid": not errors,
        "errors": errors,
        "freeze_status": freeze.get("status") if freeze else None,
        "evaluation_status": evaluation.get("status") if evaluation else None,
    }
