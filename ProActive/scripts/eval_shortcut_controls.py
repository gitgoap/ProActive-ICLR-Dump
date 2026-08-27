#!/usr/bin/env python3
"""Fit reviewer-facing identity controls on train and report validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from proactive.eval.diagnostic_metrics import diagnostic_metrics, within_group_metrics
from proactive.eval.shortcut_controls import IdentityVocabulary, control_features
from proactive.train.vectorized import load_vectorized_split
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/diag_bakeoff.yaml")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", default="outputs/week5_reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _slice(values: Any, limit: int | None) -> Any:
    return values if limit is None else values[:limit]


def _probabilities(features: np.ndarray, bits: np.ndarray, six: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    bit_probabilities = []
    for index in range(bits.shape[1]):
        if np.unique(bits[:, index]).size != 2:
            raise ValueError(f"Source bit {index} has only one training class")
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(features, bits[:, index])
        bit_probabilities.append(model)
    if np.unique(six).size != 6:
        raise ValueError("All six reporting classes must appear in training")
    six_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    six_model.fit(features, six)
    return np.asarray(bit_probabilities, dtype=object), np.asarray([six_model], dtype=object)


def _evaluate(train: Any, val: Any, *, evidence: bool, seed: int, vocabulary: IdentityVocabulary) -> Dict[str, Any]:
    kwargs_train = {
        "clean_features": train.model_input["clean_features"].numpy(),
        "metadata": train.audit_metadata,
        "vocabulary": vocabulary,
    }
    kwargs_val = {
        "clean_features": val.model_input["clean_features"].numpy(),
        "metadata": val.audit_metadata,
        "vocabulary": vocabulary,
    }
    if evidence:
        kwargs_train.update(
            probe_numeric=train.model_input["probe_numeric"].numpy(),
            acquired_mask=train.model_input["acquired_mask"].numpy(),
        )
        kwargs_val.update(
            probe_numeric=val.model_input["probe_numeric"].numpy(),
            acquired_mask=val.model_input["acquired_mask"].numpy(),
        )
    x_train = control_features(**kwargs_train)
    x_val = control_features(**kwargs_val)
    bit_models, six_models = _probabilities(
        x_train,
        train.targets["source_bits"].numpy().astype(np.int64),
        train.targets["six_way"].numpy().astype(np.int64),
        seed,
    )
    bit_probability = np.column_stack([model.predict_proba(x_val)[:, 1] for model in bit_models])
    six_model = six_models[0]
    raw_six = six_model.predict_proba(x_val)
    six_probability = np.zeros((x_val.shape[0], 6), dtype=np.float64)
    six_probability[:, six_model.classes_.astype(int)] = raw_six
    metric_inputs = {
        "bit_probabilities": bit_probability,
        "six_way_probabilities": six_probability,
        "bit_targets": val.targets["source_bits"].numpy().astype(np.int64),
        "six_way_targets": val.targets["six_way"].numpy().astype(np.int64),
    }
    return {
        "feature_count": int(x_train.shape[1]),
        "pooled": diagnostic_metrics(**metric_inputs),
        "within_dataset": within_group_metrics(
            group_values=val.audit_metadata["dataset"], minimum_rows=25, metric_inputs=metric_inputs
        ),
        "within_model": within_group_metrics(
            group_values=val.audit_metadata["model_id"], minimum_rows=25, metric_inputs=metric_inputs
        ),
    }


def main() -> None:
    args = parse_args()
    if args.device != "cpu":
        raise SystemExit("Shortcut controls are CPU-only")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config_path = Path(args.config)
    manifest_path = Path(args.manifest_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["splits"]["train"] != "train" or config["splits"]["selection"] != "val":
        raise SystemExit("Shortcut-control split firewall requires train/val")
    output_path = Path(args.output_dir) / "shortcut_controls.json"
    if output_path.exists() and args.resume:
        with open(output_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("manifest_sha256") != file_sha256(manifest_path):
            raise SystemExit("Shortcut-control resume refused: manifest drift")
        print(json.dumps(existing, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({"train_split": "train", "evaluation_split": "val", "output": str(output_path)}, indent=2))
        return
    train = load_vectorized_split(manifest_path, "train")
    val = load_vectorized_split(manifest_path, "val")
    if args.limit is not None:
        # Pilot-only deterministic prefix; the report is marked invalid below.
        from proactive.train.vectorized import IndexedStateDataset
        train = IndexedStateDataset(train, range(min(args.limit, len(train))))
        val = IndexedStateDataset(val, range(min(args.limit, len(val))))
        raise SystemExit("Shortcut-control --limit is reserved for readiness; use --dry_run")
    vocabulary = IdentityVocabulary.fit(train.audit_metadata)
    result: Dict[str, Any] = {
        "format_version": "week5_shortcut_controls_v1",
        "is_valid": True,
        "fit_split": "train",
        "evaluation_split": "val",
        "calibration_used": False,
        "test_used": False,
        "main_learner_received_identity": False,
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "vocabulary": vocabulary.to_dict(),
        "identity_clean_control": _evaluate(train, val, evidence=False, seed=args.seed, vocabulary=vocabulary),
        "identity_clean_plus_probe_control": _evaluate(train, val, evidence=True, seed=args.seed, vocabulary=vocabulary),
    }
    result["report_sha256"] = hash_dict(result)
    write_json(result, output_path, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
