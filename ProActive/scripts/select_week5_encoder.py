#!/usr/bin/env python3
"""Apply the predeclared Week 5 gate and freeze the primary encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping

import yaml

from proactive.train.checkpoints import load_checkpoint, write_freeze_manifest
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


CONDITIONS = {
    "clean_mlp": ("standard",),
    "gru": ("canonical", "random_permutation"),
    "masked_slot_mlp": ("standard",),
    "deep_sets": ("standard",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/diag_bakeoff.yaml")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--approve_selection", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def _validate_signed_report(path: Path, hash_field: str = "report_sha256") -> Dict[str, Any]:
    value = _read_json(path)
    expected = value.get(hash_field)
    unsigned = {key: item for key, item in value.items() if key != hash_field}
    if expected != hash_dict(unsigned):
        raise SystemExit(f"Report self-hash mismatch: {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.device != "cpu" or args.limit is not None:
        raise SystemExit("Week 5 selection is a complete CPU-only gate and refuses --limit")
    config_path = Path(args.config)
    manifest_path = Path(args.manifest_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("metadata", {}).get("approval_status") != "APPROVED" and not args.dry_run:
        raise SystemExit("Week 5 settings are not APPROVED")
    checkpoint_dir = Path(args.checkpoint_dir or config["outputs"]["checkpoint_dir"])
    report_dir = Path(args.report_dir or config["outputs"]["report_dir"])
    output_dir = Path(args.output_dir or report_dir)
    selection_path = output_dir / "week5_encoder_selection.json"
    freeze_path = output_dir / "week5_encoder_freeze.json"
    seeds = [int(value) for value in config["seeds"]]
    primary_seed = int(config["primary_seed"])
    if primary_seed not in seeds:
        raise SystemExit("primary_seed must be one of the declared seeds")
    metric_reports: Dict[str, list[Dict[str, Any]]] = {}
    checkpoints: Dict[str, Dict[int, Path]] = {}
    permutation_reports: Dict[str, Dict[int, tuple[Path, Dict[str, Any]]]] = {}
    missing = []
    for encoder, conditions in CONDITIONS.items():
        for condition in conditions:
            key = f"{encoder}:{condition}"
            metric_reports[key] = []
            checkpoints[key] = {}
            if encoder != "clean_mlp":
                permutation_reports[key] = {}
            for seed in seeds:
                stem = f"diag_{encoder}_{condition}_seed{seed}"
                checkpoint_path = checkpoint_dir / f"{stem}.best.pt"
                metric_path = checkpoint_dir / f"{stem}.validation.json"
                permutation_path = report_dir / f"permutation_week5_{encoder}_{condition}_seed{seed}.json"
                if not checkpoint_path.exists() or not metric_path.exists():
                    missing.append(str(checkpoint_path if not checkpoint_path.exists() else metric_path))
                    continue
                report = _validate_signed_report(metric_path)
                checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
                if report.get("checkpoint_sha256") != file_sha256(checkpoint_path):
                    raise SystemExit(f"Metric/checkpoint hash mismatch: {metric_path}")
                if checkpoint.get("source_manifest_sha256") != file_sha256(manifest_path):
                    raise SystemExit(f"Checkpoint/vector manifest mismatch: {checkpoint_path}")
                if report.get("pilot") is not False or not checkpoint.get("scientifically_valid"):
                    raise SystemExit(f"Pilot/invalid checkpoint cannot enter Week 5 selection: {checkpoint_path}")
                metric_reports[key].append(report)
                checkpoints[key][seed] = checkpoint_path
                if encoder != "clean_mlp":
                    if not permutation_path.exists():
                        missing.append(str(permutation_path))
                    else:
                        permutation = _validate_signed_report(permutation_path)
                        if permutation.get("checkpoint_sha256") != file_sha256(checkpoint_path):
                            raise SystemExit(f"Permutation/checkpoint hash mismatch: {permutation_path}")
                        permutation_reports[key][seed] = (permutation_path, permutation)
    shortcut_path = report_dir / "shortcut_controls.json"
    aps_path = report_dir / f"temporary_aps_deep_sets_standard_seed{primary_seed}.json"
    for path in (shortcut_path, aps_path):
        if not path.exists():
            missing.append(str(path))
    if missing:
        if args.dry_run:
            print(json.dumps({"is_ready": False, "missing": missing, "selection_output": str(selection_path)}, indent=2))
            return
        raise SystemExit("Week 5 selection artifacts missing:\n" + "\n".join(missing))
    shortcut = _validate_signed_report(shortcut_path)
    aps = _validate_signed_report(aps_path)
    if shortcut.get("is_valid") is not True:
        raise SystemExit("Shortcut-control audit is not valid")
    primary_checkpoint = checkpoints["deep_sets:standard"][primary_seed]
    if aps.get("checkpoint_sha256") != file_sha256(primary_checkpoint):
        raise SystemExit("Primary Deep Sets checkpoint/temporary APS mismatch")

    primary_metric = str(config["training_proposal"]["selection_primary"])
    tiebreak_metric = str(config["training_proposal"]["selection_tiebreak"])

    def metric(key: str, name: str) -> float:
        return mean(float(report["metrics"][name]) for report in metric_reports[key])

    def drift(key: str) -> float:
        return mean(
            float(value[1]["summary"]["js_drift"]["mean"])
            for value in permutation_reports[key].values()
        )

    deep_primary = metric("deep_sets:standard", primary_metric)
    gru_key = max(
        ("gru:canonical", "gru:random_permutation"),
        key=lambda key: (metric(key, primary_metric), metric(key, tiebreak_metric)),
    )
    gru_primary = metric(gru_key, primary_metric)
    deep_drift = drift("deep_sets:standard")
    gru_drift = drift(gru_key)
    gate = config["selection_gate"]
    equals_or_better = deep_primary >= gru_primary
    within_tolerance = deep_primary >= gru_primary - float(gate["macro_f1_tolerance"])
    substantially_lower = (
        deep_drift <= float(gate["invariant_tolerance"])
        and gru_drift > float(gate["invariant_tolerance"])
        and deep_drift <= float(gate["substantial_drift_ratio"]) * gru_drift
    )
    completion_gate = equals_or_better or (within_tolerance and substantially_lower)
    set_transformer_trigger = gru_primary - deep_primary >= float(gate["set_transformer_gap_trigger"])
    if set_transformer_trigger:
        completion_gate = False
    raps_config = config["raps_gate"]
    raps_items = [
        aps["thresholds"][str(budget)][format(0.90, ".12g")]
        for budget in config["temporary_aps"]["budgets"]
    ]
    mean_set_size = mean(float(item["validation_metrics"]["average_set_size"]) for item in raps_items)
    mean_singleton_rate = mean(float(item["validation_metrics"]["singleton_rate"]) for item in raps_items)
    coverage_near_target = all(
        abs(
            float(item["validation_metrics"]["coverage"])
            - float(item["target_coverage"])
        )
        <= float(raps_config["coverage_near_target_tolerance"])
        for item in raps_items
    )
    raps_trigger = bool(
        mean_set_size > float(raps_config["average_set_size_trigger"])
        or (
            mean_singleton_rate < float(raps_config["singleton_rate_trigger"])
            and coverage_near_target
        )
    )
    identity_macro_f1 = float(
        shortcut["identity_clean_control"]["pooled"][primary_metric]
    )
    shortcut_gap = deep_primary - identity_macro_f1
    shortcut_gate_passed = shortcut_gap >= float(
        config["shortcut_gate"]["minimum_main_minus_identity_macro_f1"]
    )
    if not shortcut_gate_passed:
        completion_gate = False
    selection: Dict[str, Any] = {
        "format_version": "week5_encoder_selection_v1",
        "status": "SELECTED" if completion_gate else "GATE_FAILED",
        "fit_split": "train",
        "selection_split": "val",
        "calibration_used": False,
        "test_used": False,
        "selected_encoder": "deep_sets" if completion_gate else None,
        "selected_seed": primary_seed if completion_gate else None,
        "selected_checkpoint_path": str(primary_checkpoint) if completion_gate else None,
        "selected_checkpoint_sha256": file_sha256(primary_checkpoint) if completion_gate else None,
        "selection_primary_metric": primary_metric,
        "selection_tiebreak_metric": tiebreak_metric,
        "deep_sets_primary_mean": deep_primary,
        "deep_sets_tiebreak_mean": metric("deep_sets:standard", tiebreak_metric),
        "best_gru_condition": gru_key.split(":", 1)[1],
        "best_gru_primary_mean": gru_primary,
        "best_gru_tiebreak_mean": metric(gru_key, tiebreak_metric),
        "deep_sets_js_drift_mean": deep_drift,
        "best_gru_js_drift_mean": gru_drift,
        "completion_gate_passed": completion_gate,
        "set_transformer_gate": "TRIGGERED_REQUIRES_OWNER_REVIEW" if set_transformer_trigger else "NOT_TRIGGERED",
        "raps_gate": {
            "status": "TRIGGERED_APPENDIX_ABLATION" if raps_trigger else "NOT_TRIGGERED",
            "mean_validation_set_size_at_90": mean_set_size,
            "mean_validation_singleton_rate_at_90": mean_singleton_rate,
            "coverage_near_target": coverage_near_target,
            "config": dict(raps_config),
        },
        "shortcut_gate": {
            "status": "PASSED" if shortcut_gate_passed else "FAILED_REQUIRES_SCOPE_REVIEW",
            "deep_sets_source_bit_macro_f1_mean": deep_primary,
            "identity_clean_source_bit_macro_f1": identity_macro_f1,
            "main_minus_identity_gap": shortcut_gap,
            "config": dict(config["shortcut_gate"]),
        },
        "gate_config": dict(gate),
        "config_sha256": file_sha256(config_path),
        "vector_manifest_sha256": file_sha256(manifest_path),
        "temporary_aps_sha256": file_sha256(aps_path),
        "shortcut_control_sha256": file_sha256(shortcut_path),
    }
    selection["report_sha256"] = hash_dict(selection)
    if args.dry_run:
        print(json.dumps(selection, indent=2))
        return
    write_json(selection, selection_path, overwrite=(args.overwrite or args.resume) and selection_path.exists())
    if not completion_gate:
        raise SystemExit(
            "Week 5 encoder/shortcut/Set-Transformer gate did not permit freezing"
        )
    if not args.approve_selection:
        raise SystemExit(
            f"Selection evidence written to {selection_path}. Re-run with --approve_selection after owner review."
        )
    additional: Dict[str, Path] = {}
    for key, candidates in checkpoints.items():
        if key == "deep_sets:standard":
            continue
        safe_key = key.replace(":", "_")
        additional[f"diagnostic_checkpoint_{safe_key}"] = candidates[primary_seed]
    for key, values in permutation_reports.items():
        additional[f"permutation_{key.replace(':', '_')}"] = values[primary_seed][0]
    write_freeze_manifest(
        diagnostic_checkpoint=primary_checkpoint,
        policy_checkpoint=None,
        selection_report=selection_path,
        config_paths={
            "diagnostic_config": config_path,
            "temporary_aps": aps_path,
            "shortcut_controls": shortcut_path,
            "vector_manifest": manifest_path,
        },
        additional_artifacts=additional,
        output_path=freeze_path,
        approved_by_owner=True,
        overwrite=(args.overwrite or args.resume) and freeze_path.exists(),
    )
    print(json.dumps(_read_json(freeze_path), indent=2))


if __name__ == "__main__":
    main()
