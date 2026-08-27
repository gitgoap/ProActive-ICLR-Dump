#!/usr/bin/env python3
"""Train one action-conditioned VOI policy with a frozen diagnostic encoder."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import yaml
from torch.utils.data import DataLoader

from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.voi import ActionConditionedVOIHead, FrozenDiagnosticPolicy
from proactive.train.checkpoints import (
    POLICY_CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
    validate_freeze_manifest,
)
from proactive.train.policy import VOITargetDataset, load_voi_records, policy_epoch
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import load_vectorized_split
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json
from proactive.utils.seed import set_global_seed


LOGGER = logging.getLogger("train_policy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/policy_train.yaml")
    parser.add_argument("--manifest_path", required=True, help="Week 5 vectorized manifest")
    parser.add_argument("--voi_manifest", required=True)
    parser.add_argument("--diagnostic_checkpoint", required=True)
    parser.add_argument("--freeze_manifest", default=None)
    parser.add_argument("--cost_multiplier", type=float, required=True)
    parser.add_argument("--target_kind", choices=("voi", "entropy_reduction"), default="voi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_unapproved_pilot", action="store_true")
    return parser.parse_args()


def _optimizer(parameters: Any, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    if str(config.get("optimizer", "adamw")).lower() != "adamw":
        raise SystemExit("Only the declared AdamW policy optimizer is supported")
    return torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config_path = Path(args.config)
    vector_manifest_path = Path(args.manifest_path)
    voi_manifest_path = Path(args.voi_manifest)
    diagnostic_path = Path(args.diagnostic_checkpoint)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    approval = config.get("metadata", {}).get("approval_status")
    pilot = args.limit is not None and args.limit <= 100
    if approval != "APPROVED" and not args.dry_run and not (pilot and args.allow_unapproved_pilot):
        raise SystemExit("Week 6 settings are not APPROVED; only dry-run or explicit <=100-row pilot is allowed")
    grid = [float(value) for value in config["cost_multiplier_grid"]]
    if args.target_kind == "voi" and args.cost_multiplier not in grid:
        raise SystemExit(f"Cost multiplier {args.cost_multiplier} is outside the declared validation grid")
    if args.target_kind == "entropy_reduction" and args.cost_multiplier != 0.0:
        raise SystemExit("Entropy-greedy baseline must use --cost_multiplier 0")
    freeze_path = Path(args.freeze_manifest or config["week5_freeze_manifest"])
    freeze = validate_freeze_manifest(freeze_path, require_policy=False)
    diagnostic_sha = file_sha256(diagnostic_path)
    if not any(
        name.startswith("diagnostic_checkpoint") and item.get("sha256") == diagnostic_sha
        for name, item in freeze["artifacts"].items()
    ):
        raise SystemExit("Policy diagnostic checkpoint is not included in the Week 5 freeze")
    diagnostic_checkpoint = load_checkpoint(diagnostic_path, map_location="cpu")
    if not diagnostic_checkpoint.get("scientifically_valid") and not pilot:
        raise SystemExit("Full policy training requires a scientifically valid diagnostic checkpoint")
    if diagnostic_checkpoint.get("source_manifest_sha256") != file_sha256(vector_manifest_path):
        raise SystemExit("Policy checkpoint/vectorized-manifest drift")
    voi_manifest, train_records = load_voi_records(
        voi_manifest_path, "train", require_complete=not pilot
    )
    _, val_records = load_voi_records(voi_manifest_path, "val", require_complete=not pilot)
    if voi_manifest.get("checkpoint_sha256") != diagnostic_sha:
        raise SystemExit("VOI targets were built with a different diagnostic checkpoint")
    output_dir = Path(args.output_dir or config["outputs"]["checkpoint_dir"])
    multiplier_name = format(args.cost_multiplier, ".12g").replace(".", "p")
    encoder = diagnostic_checkpoint["encoder_name"]
    condition = diagnostic_checkpoint.get("gru_condition", "standard")
    prefix = "policy" if args.target_kind == "voi" else "uncertainty"
    stem = f"{prefix}_{encoder}_{condition}_lambda{multiplier_name}_seed{args.seed}"
    best_path = output_dir / f"{stem}.best.pt"
    last_path = output_dir / f"{stem}.last.pt"
    report_path = output_dir / f"{stem}.validation.json"
    history_path = output_dir / f"{stem}.history.json"
    summary = {
        "encoder": encoder,
        "gru_condition": condition,
        "cost_multiplier": args.cost_multiplier,
        "target_kind": args.target_kind,
        "limit": args.limit,
        "seed": args.seed,
        "pilot": pilot,
        "config_approval": approval,
        "output": str(best_path),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return
    if any(path.exists() for path in (best_path, last_path, report_path, history_path)) and not (args.resume or args.overwrite):
        raise SystemExit(f"Policy outputs already exist for {stem}; use --resume or --overwrite")
    set_global_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA requested but unavailable: {args.device}")
    train_base = load_vectorized_split(vector_manifest_path, "train")
    val_base = load_vectorized_split(vector_manifest_path, "val")
    train_data = VOITargetDataset(
        train_base,
        train_records,
        split="train",
        cost_multiplier=args.cost_multiplier,
        target_kind=args.target_kind,
        limit=args.limit,
    )
    val_data = VOITargetDataset(
        val_base,
        val_records,
        split="val",
        cost_multiplier=args.cost_multiplier,
        target_kind=args.target_kind,
        limit=args.limit,
    )
    training = config["training_proposal"]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=min(int(training["batch_size"]), len(train_data)),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=min(int(training["batch_size"]), len(val_data)),
        shuffle=False,
        num_workers=0,
    )
    diagnostic = build_diagnostic_model(
        diagnostic_checkpoint["encoder_name"], diagnostic_checkpoint["architecture"]
    )
    diagnostic.load_state_dict(diagnostic_checkpoint["model_state_dict"])
    policy_config = config["policy"]
    head = ActionConditionedVOIHead(
        state_dim=int(diagnostic_checkpoint["architecture"]["state_dim"]),
        action_embedding_dim=int(policy_config["action_embedding_dim"]),
        budget_embedding_dim=int(policy_config["budget_embedding_dim"]),
        hidden_dim=int(policy_config["hidden_dim"]),
        dropout=float(policy_config["dropout"]),
        max_budget=int(config["full_budget"]),
    )
    model = FrozenDiagnosticPolicy(diagnostic, head).to(device)
    normalizer = FeatureNormalizer.from_state_dict(diagnostic_checkpoint["normalizer"])
    optimizer = _optimizer(model.voi_head.parameters(), training)
    start_epoch = 0
    best_score = (float("inf"), -float("inf"))
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[Dict[str, Any]] = []
    common_provenance = {
        "config_sha256": file_sha256(config_path),
        "source_manifest_sha256": file_sha256(voi_manifest_path),
        "diagnostic_checkpoint_sha256": diagnostic_sha,
        "vector_manifest_sha256": file_sha256(vector_manifest_path),
        "week5_freeze_sha256": file_sha256(freeze_path),
        "seed": args.seed,
        "cost_multiplier": args.cost_multiplier,
        "target_kind": args.target_kind,
        "limit": args.limit,
    }
    if args.resume and last_path.exists():
        last = load_checkpoint(
            last_path,
            expected_version=POLICY_CHECKPOINT_VERSION,
            map_location=device,
        )
        for key, value in common_provenance.items():
            if last.get(key) != value:
                raise SystemExit(f"Unsafe policy resume: {key} drift")
        model.voi_head.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        start_epoch = int(last["epoch"]) + 1
        best_score = tuple(last["best_score"])
        best_epoch = int(last["best_epoch"])
        epochs_without_improvement = int(last["epochs_without_improvement"])
        history = list(last["history"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    for epoch in range(start_epoch, max_epochs):
        train_metrics = policy_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            normalizer=normalizer,
            device=device,
            margin=float(policy_config["ranking_margin"]),
            mse_weight=float(policy_config["mse_weight"]),
            action_ce_weight=float(policy_config["action_ce_weight"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        val_metrics = policy_epoch(
            model,
            val_loader,
            optimizer=None,
            normalizer=normalizer,
            device=device,
            margin=float(policy_config["ranking_margin"]),
            mse_weight=float(policy_config["mse_weight"]),
            action_ce_weight=float(policy_config["action_ce_weight"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        score = (float(val_metrics["total"]), -float(val_metrics["best_action_accuracy"]))
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        payload: Dict[str, Any] = {
            "format_version": POLICY_CHECKPOINT_VERSION,
            "stage": "week6_policy" if args.target_kind == "voi" else "week6_uncertainty_baseline",
            **common_provenance,
            "encoder_name": encoder,
            "gru_condition": condition,
            "policy_architecture": dict(policy_config),
            "diagnostic_architecture": diagnostic_checkpoint["architecture"],
            "normalizer": diagnostic_checkpoint["normalizer"],
            "model_state_dict": model.voi_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scientifically_valid": not pilot,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": list(best_score),
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        save_checkpoint(payload, last_path, overwrite=True)
        if improved:
            save_checkpoint(payload, best_path, overwrite=True)
        LOGGER.info("epoch=%s val_loss=%.6f oracle_agreement=%.4f", epoch, val_metrics["total"], val_metrics["best_action_accuracy"])
        if epochs_without_improvement >= patience:
            break
    best = load_checkpoint(best_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location=device)
    model.voi_head.load_state_dict(best["model_state_dict"])
    metrics = policy_epoch(
        model,
        val_loader,
        optimizer=None,
        normalizer=normalizer,
        device=device,
        margin=float(policy_config["ranking_margin"]),
        mse_weight=float(policy_config["mse_weight"]),
        action_ce_weight=float(policy_config["action_ce_weight"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    report: Dict[str, Any] = {
        "format_version": "week6_policy_validation_v1",
        "is_valid": not pilot,
        "fit_split": "train",
        "selection_split": "val",
        "calibration_used": False,
        "test_used": False,
        "encoder_name": encoder,
        "gru_condition": condition,
        "seed": args.seed,
        "cost_multiplier": args.cost_multiplier,
        "target_kind": args.target_kind,
        "train_rows": len(train_data),
        "validation_rows": len(val_data),
        "best_epoch": int(best["best_epoch"]),
        "metrics": metrics,
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": file_sha256(best_path),
        **common_provenance,
    }
    report["report_sha256"] = hash_dict(report)
    write_json(history, history_path, overwrite=(args.overwrite or args.resume) and history_path.exists())
    write_json(report, report_path, overwrite=(args.overwrite or args.resume) and report_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
