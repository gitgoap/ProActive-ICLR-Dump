#!/usr/bin/env python3
"""Materialize predeclared, approval-bound Week 8 training configs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.train.ablations import validate_week8_ablation_authorization
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json, write_text


TRAINED_ABLATIONS = (
    "no_budget_embedding",
    "no_answer_flip",
    "no_confidence_shift",
    "no_semantic_match",
    "no_relation_probe",
    "no_signature_regression",
    "shared_vs_independent_heads",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week8_config", default="configs/experiments/week8_evaluation.yaml")
    parser.add_argument("--diag_config", default="configs/experiments/diag_bakeoff.yaml")
    parser.add_argument("--policy_config", default="configs/experiments/policy_train.yaml")
    parser.add_argument("--output_dir", default="outputs/week8_data/ablation_configs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected YAML mapping: {path}")
    return value


def _dump(value: Dict[str, Any]) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    week8_path = Path(args.week8_config)
    diag_path = Path(args.diag_config)
    policy_path = Path(args.policy_config)
    week8 = _load(week8_path)
    if week8.get("metadata", {}).get("approval_status") != "APPROVED":
        raise SystemExit(
            "Week 8 config is not owner-approved; ablation training configs remain locked"
        )
    try:
        validate_week8_ablation_authorization(week8)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    declared = set(week8["ablation_execution"]["mandatory"])
    if not set(TRAINED_ABLATIONS).issubset(declared):
        raise SystemExit("Week 8 mandatory-ablation registry drift")
    base_diag = _load(diag_path)
    base_policy = _load(policy_path)
    output_dir = Path(args.output_dir)
    artifacts = []
    rendered = []
    for ablation in TRAINED_ABLATIONS:
        root = Path("outputs/week8_ablations") / ablation
        diag = copy.deepcopy(base_diag)
        policy = copy.deepcopy(base_policy)
        diag["metadata"] = {
            **diag.get("metadata", {}),
            "version": f"week8_{ablation}_v1",
            "approval_status": "APPROVED",
            "approved_scope": f"predeclared_week8_ablation:{ablation}",
            "parent_week8_config_sha256": file_sha256(week8_path),
        }
        diag["seeds"] = [int(week8["seed"])]
        diag["primary_seed"] = int(week8["seed"])
        diag["outputs"] = {
            **diag["outputs"],
            "checkpoint_dir": str(root / "diagnostic_checkpoints"),
            "report_dir": str(root / "diagnostic_reports"),
        }
        if ablation == "no_budget_embedding":
            diag["architecture"]["budget_embedding_dim"] = 0
        elif ablation == "no_signature_regression":
            diag["loss"]["signature_weight"] = 0.0
        elif ablation == "shared_vs_independent_heads":
            diag["architecture"]["multi_task_mode"] = "independent_source_encoders"

        diag_output = output_dir / f"{ablation}_diag.yaml"
        policy["metadata"] = {
            **policy.get("metadata", {}),
            "version": f"week8_{ablation}_policy_v1",
            "approval_status": "APPROVED",
            "approved_scope": f"predeclared_week8_ablation:{ablation}",
            "parent_week8_config_sha256": file_sha256(week8_path),
        }
        policy["diagnostic_config"] = str(diag_output)
        policy["seeds"] = [int(week8["seed"])]
        policy["primary_seed"] = int(week8["seed"])
        policy["cost_multiplier_grid"] = [0.0]
        policy["week5_freeze_manifest"] = str(
            root / "freeze" / "week8_diagnostic_freeze.json"
        )
        policy["outputs"] = {
            "voi_dir": str(root / "voi"),
            "checkpoint_dir": str(root / "policy_checkpoints"),
            "report_dir": str(root / "policy_reports"),
        }
        if ablation == "no_budget_embedding":
            policy["policy"]["budget_embedding_dim"] = 0
        policy_output = output_dir / f"{ablation}_policy.yaml"
        rendered.append((diag_output, _dump(diag)))
        rendered.append((policy_output, _dump(policy)))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "is_valid": True,
                    "ablations": list(TRAINED_ABLATIONS),
                    "files": [str(path) for path, _ in rendered],
                },
                indent=2,
            )
        )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in rendered:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {path}; use --overwrite")
        write_text(content, path, overwrite=args.overwrite and path.exists())
        artifacts.append({"path": str(path), "sha256": file_sha256(path)})
    report: Dict[str, Any] = {
        "format_version": "week8_ablation_config_bundle_v1",
        "status": "APPROVED",
        "week8_config_sha256": file_sha256(week8_path),
        "base_diag_config_sha256": file_sha256(diag_path),
        "base_policy_config_sha256": file_sha256(policy_path),
        "ablations": list(TRAINED_ABLATIONS),
        "artifacts": artifacts,
        "calibration_used": False,
        "test_used": False,
    }
    report["report_sha256"] = hash_dict(report)
    report_path = output_dir / "ablation_config_bundle.json"
    write_json(report, report_path, overwrite=args.overwrite and report_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
