#!/usr/bin/env python3
"""Train one mandatory diagnostic encoder/seed with strict split guards."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import yaml
from torch.utils.data import DataLoader

from proactive.networks.diagnostic import ENCODER_NAMES, build_diagnostic_model
from proactive.networks.losses import compute_training_weights
from proactive.train.checkpoints import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
    validate_completed_training_report,
    validate_early_stopping_history,
)
from proactive.train.diagnostic import (
    evaluate_predictions,
    predict,
    stratified_prediction_metrics,
    train_epoch,
)
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import (
    BudgetProjectedDataset,
    empty_state_view,
    load_vectorized_split,
)
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json
from proactive.utils.seed import set_global_seed


LOGGER = logging.getLogger("train_diag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/diag_bakeoff.yaml")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--encoder", required=True, choices=ENCODER_NAMES)
    parser.add_argument("--gru_condition", choices=("canonical", "random_permutation"), default="canonical")
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


def _optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    if name != "adamw":
        raise SystemExit(f"Unsupported optimizer: {name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


def _scientific_metric(predictions: Mapping[str, Any], pilot: bool) -> Dict[str, Any]:
    try:
        return {"is_valid": True, **evaluate_predictions(predictions)}
    except ValueError as exc:
        if not pilot:
            raise
        return {"is_valid": False, "pilot_only_reason": str(exc)}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise SystemExit("Diagnostic config must be a mapping")
    approval = config.get("metadata", {}).get("approval_status")
    pilot = args.limit is not None and args.limit <= 100
    if approval != "APPROVED" and not args.dry_run:
        if not (args.allow_unapproved_pilot and pilot):
            raise SystemExit(
                "Week 5 training settings are not APPROVED. Only --dry_run or an explicit "
                "--allow_unapproved_pilot with --limit <= 100 is permitted."
            )
    if args.encoder != "gru" and args.gru_condition != "canonical":
        raise SystemExit("--gru_condition applies only to --encoder gru")
    if args.seed not in [int(value) for value in config["seeds"]] and not pilot:
        raise SystemExit(f"Seed {args.seed} is outside the declared three-seed comparison")
    manifest_path = Path(args.manifest_path or Path(config["outputs"]["vectorized_dir"]) / "vectorized_manifest.json")
    output_dir = Path(args.output_dir or config["outputs"]["checkpoint_dir"])
    condition = args.gru_condition if args.encoder == "gru" else "standard"
    stem = f"diag_{args.encoder}_{condition}_seed{args.seed}"
    best_path = output_dir / f"{stem}.best.pt"
    last_path = output_dir / f"{stem}.last.pt"
    history_path = output_dir / f"{stem}.history.json"
    metrics_path = output_dir / f"{stem}.validation.json"
    config_sha = file_sha256(config_path)
    manifest_sha = file_sha256(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        vector_manifest = json.load(handle)
    if vector_manifest.get("status") != "COMPLETE" and not pilot:
        raise SystemExit("Full Week 5 training requires a COMPLETE vectorized manifest")
    print(json.dumps({
        "encoder": args.encoder,
        "gru_condition": condition,
        "seed": args.seed,
        "budgets": config["initial_budgets"],
        "config_approval": approval,
        "pilot": pilot,
        "manifest": str(manifest_path),
        "output": str(best_path),
    }, indent=2))
    if args.dry_run:
        return

    if args.resume:
        try:
            completed = validate_completed_training_report(
                metrics_path,
                best_path,
                expected_report={
                    "encoder_name": args.encoder,
                    "gru_condition": condition,
                    "seed": args.seed,
                    "config_sha256": config_sha,
                    "source_manifest_sha256": manifest_sha,
                },
                expected_checkpoint={
                    "config_sha256": config_sha,
                    "source_manifest_sha256": manifest_sha,
                    "encoder_name": args.encoder,
                    "gru_condition": condition,
                    "seed": args.seed,
                    "limit": args.limit,
                },
                expected_version=CHECKPOINT_VERSION,
            )
            if completed is not None:
                if not history_path.is_file():
                    raise ValueError("Completed diagnostic history is missing")
                with open(history_path, "r", encoding="utf-8") as handle:
                    completed_history = json.load(handle)
                validate_early_stopping_history(
                    [
                        (
                            row["validation"].get("source_bit_macro_f1", -float("inf")),
                            row["validation"].get("six_way_macro_f1", -float("inf")),
                        )
                        for row in completed_history
                    ],
                    patience=int(config["training_proposal"]["early_stopping_patience"]),
                    minimize=False,
                    reported_best_epoch=int(completed["best_epoch"]),
                )
                LOGGER.info("Completed diagnostic run verified; training is unchanged")
                print(json.dumps(completed, indent=2))
                return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Unsafe diagnostic resume: {exc}") from exc

    set_global_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA requested but unavailable: {args.device}")
    train_base = load_vectorized_split(manifest_path, config["splits"]["train"])
    val_base = load_vectorized_split(manifest_path, config["splits"]["selection"])
    if train_base.split != "train" or val_base.split != "val":
        raise SystemExit("Week 5 split firewall requires train/val exactly")
    budgets = [int(value) for value in config["initial_budgets"]]
    # The clean baseline gets one empty state per model-instance. Training it
    # on every partial-state copy would silently overweight relation-capable
    # instances even though the evidence is intentionally ignored.
    train_source = empty_state_view(train_base) if args.encoder == "clean_mlp" else train_base
    train_budgets = [budgets[0]] if args.encoder == "clean_mlp" else budgets
    train_data = BudgetProjectedDataset(train_source, train_budgets, limit=args.limit)
    val_source = empty_state_view(val_base) if args.encoder == "clean_mlp" else val_base
    val_data = BudgetProjectedDataset(val_source, budgets, limit=args.limit)
    training = config["training_proposal"]
    batch_size = min(int(training["batch_size"]), len(train_data))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=min(int(training["batch_size"]), len(val_data)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
    )
    empty_train = empty_state_view(train_base)
    empty_indices = torch.tensor(empty_train.indices, dtype=torch.long)
    acquired_mask = train_base.model_input["acquired_mask"].bool()
    normalizer = FeatureNormalizer.fit_components(
        clean_features=train_base.model_input["clean_features"][empty_indices],
        acquired_probe_features=train_base.model_input["probe_numeric"][acquired_mask],
    )
    weight_indices = empty_indices if args.encoder == "clean_mlp" else torch.arange(len(train_base))
    try:
        bit_pos_weight, class_weight = compute_training_weights(
            train_base.targets["source_bits"][weight_indices],
            train_base.targets["six_way"][weight_indices],
        )
        scientifically_weighted = True
    except ValueError:
        if not pilot:
            raise
        bit_pos_weight = torch.ones(3)
        class_weight = torch.ones(6)
        scientifically_weighted = False

    architecture = {**config["architecture"], "max_budget": int(config["full_budget"])}
    model = build_diagnostic_model(args.encoder, architecture).to(device)
    optimizer = _optimizer(model, training)
    start_epoch = 0
    best_score = (-float("inf"), -float("inf"))
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    if last_path.exists() and args.resume:
        checkpoint = load_checkpoint(last_path, map_location=device)
        expected = {
            "config_sha256": config_sha,
            "source_manifest_sha256": manifest_sha,
            "encoder_name": args.encoder,
            "gru_condition": condition,
            "seed": args.seed,
            "limit": args.limit,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise SystemExit(f"Unsafe diagnostic resume: {key} drift")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = tuple(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint["history"])
        normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
        LOGGER.info("Resuming at epoch %s", start_epoch)
    elif any(path.exists() for path in (best_path, last_path, history_path, metrics_path)) and not args.overwrite:
        raise SystemExit(f"Outputs for {stem} already exist; use --resume or --overwrite")

    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    permutation_generator = torch.Generator().manual_seed(args.seed + 10_000)
    for epoch in range(start_epoch, max_epochs):
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            normalizer=normalizer,
            bit_pos_weight=bit_pos_weight,
            class_weight=class_weight,
            six_way_weight=float(config["loss"]["six_way_weight"]),
            signature_weight=float(config["loss"]["signature_weight"]),
            device=device,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            randomize_gru_order=args.encoder == "gru" and condition == "random_permutation",
            permutation_generator=permutation_generator,
        )
        predictions = predict(model, val_loader, normalizer=normalizer, device=device)
        metrics = _scientific_metric(predictions, pilot)
        score = (
            float(metrics.get("source_bit_macro_f1", -float("inf"))),
            float(metrics.get("six_way_macro_f1", -float("inf"))),
        )
        improved = score > best_score or (pilot and best_epoch < 0)
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch, "train_loss": train_losses, "validation": metrics})
        payload: Dict[str, Any] = {
            "format_version": CHECKPOINT_VERSION,
            "stage": "week5_diagnostic",
            "encoder_name": args.encoder,
            "gru_condition": condition,
            "seed": args.seed,
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": manifest_sha,
            "architecture": architecture,
            "budgets": budgets,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "bit_pos_weight": bit_pos_weight,
            "class_weight": class_weight,
            "scientifically_weighted": scientifically_weighted,
            "scientifically_valid": not pilot and metrics.get("is_valid") is True,
            "limit": args.limit,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": list(best_score),
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        save_checkpoint(payload, last_path, overwrite=True)
        if improved:
            save_checkpoint(payload, best_path, overwrite=True)
        LOGGER.info("epoch=%s train_total=%.6f score=%s", epoch, train_losses["total"], score)
        if epochs_without_improvement >= patience:
            break

    best = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    final_predictions = predict(model, val_loader, normalizer=normalizer, device=device)
    final_metrics = _scientific_metric(final_predictions, pilot)
    stratified = (
        stratified_prediction_metrics(final_predictions)
        if not pilot
        else {"pooled": final_metrics, "status": "PILOT_STRATA_NOT_REPORTED"}
    )
    report = {
        "format_version": "week5_validation_report_v1",
        "encoder_name": args.encoder,
        "gru_condition": condition,
        "seed": args.seed,
        "pilot": pilot,
        "scientifically_weighted": scientifically_weighted,
        "train_rows": len(train_data),
        "validation_rows": len(val_data),
        "best_epoch": int(best["best_epoch"]),
        "metrics": final_metrics,
        "stratified_metrics": stratified,
        "config_sha256": config_sha,
        "source_manifest_sha256": manifest_sha,
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": file_sha256(best_path),
    }
    report["report_sha256"] = hash_dict(report)
    write_json(history, history_path, overwrite=args.overwrite or args.resume or history_path.exists())
    write_json(report, metrics_path, overwrite=args.overwrite or args.resume or metrics_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
