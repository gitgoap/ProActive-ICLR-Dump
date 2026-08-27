#!/usr/bin/env python3
"""Train scalar-confidence and one-pass-distilled Week 6 baselines."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from proactive.networks.controls import ScalarConfidenceDiagnostic
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.losses import diagnostic_loss
from proactive.train.checkpoints import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint, validate_freeze_manifest
from proactive.train.diagnostic import evaluate_predictions, predict
from proactive.train.distillation import FullEvidencePairDataset, distillation_loss
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import BudgetProjectedDataset, empty_state_view, load_vectorized_split
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json
from proactive.utils.seed import set_global_seed


LOGGER = logging.getLogger("train_one_pass_baselines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/policy_train.yaml")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--diagnostic_checkpoint", required=True)
    parser.add_argument("--freeze_manifest", default=None)
    parser.add_argument("--baseline", choices=("scalar", "distilled"), required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_unapproved_pilot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config_path = Path(args.config)
    manifest_path = Path(args.manifest_path)
    diagnostic_path = Path(args.diagnostic_checkpoint)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    pilot = args.limit is not None and args.limit <= 100
    if config.get("metadata", {}).get("approval_status") != "APPROVED" and not args.dry_run and not (pilot and args.allow_unapproved_pilot):
        raise SystemExit("Week 6 settings are not APPROVED")
    freeze_path = Path(args.freeze_manifest or config["week5_freeze_manifest"])
    freeze = validate_freeze_manifest(freeze_path, require_policy=False)
    teacher_checkpoint = load_checkpoint(diagnostic_path, map_location="cpu")
    diagnostic_sha = file_sha256(diagnostic_path)
    if not any(
        name.startswith("diagnostic_checkpoint") and item.get("sha256") == diagnostic_sha
        for name, item in freeze["artifacts"].items()
    ):
        raise SystemExit("One-pass baselines require a diagnostic checkpoint frozen in Week 5")
    if teacher_checkpoint.get("source_manifest_sha256") != file_sha256(manifest_path):
        raise SystemExit("Diagnostic/vectorized-manifest mismatch")
    output_dir = Path(args.output_dir or config["outputs"]["checkpoint_dir"])
    stem = f"baseline_{args.baseline}_seed{args.seed}"
    best_path = output_dir / f"{stem}.best.pt"
    last_path = output_dir / f"{stem}.last.pt"
    report_path = output_dir / f"{stem}.validation.json"
    print(json.dumps({"baseline": args.baseline, "pilot": pilot, "output": str(best_path)}, indent=2))
    if args.dry_run:
        return
    if any(path.exists() for path in (best_path, last_path, report_path)) and not (args.resume or args.overwrite):
        raise SystemExit(f"Baseline outputs exist for {stem}; use --resume or --overwrite")
    set_global_seed(args.seed)
    device = torch.device(args.device)
    train_base = load_vectorized_split(manifest_path, "train")
    val_base = load_vectorized_split(manifest_path, "val")
    train_pairs = FullEvidencePairDataset(train_base)
    if args.limit is not None:
        from torch.utils.data import Subset

        train_pairs = Subset(train_pairs, range(min(args.limit, len(train_pairs))))
    val_empty = BudgetProjectedDataset(empty_state_view(val_base), [7], limit=args.limit)
    training = config["training_proposal"]
    train_loader = DataLoader(
        train_pairs,
        batch_size=min(int(training["batch_size"]), len(train_pairs)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_empty,
        batch_size=min(int(training["batch_size"]), len(val_empty)),
        shuffle=False,
        num_workers=0,
    )
    teacher = build_diagnostic_model(teacher_checkpoint["encoder_name"], teacher_checkpoint["architecture"])
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.to(device).eval()
    if args.baseline == "scalar":
        student = ScalarConfidenceDiagnostic(
            state_dim=int(teacher_checkpoint["architecture"]["state_dim"]),
            dropout=float(teacher_checkpoint["architecture"]["dropout"]),
        )
    else:
        student = build_diagnostic_model("clean_mlp", teacher_checkpoint["architecture"])
    student.to(device)
    normalizer = FeatureNormalizer.from_state_dict(teacher_checkpoint["normalizer"])
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    start_epoch = 0
    best_score = -float("inf")
    best_epoch = -1
    stale = 0
    history = []
    provenance = {
        "config_sha256": file_sha256(config_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "diagnostic_checkpoint_sha256": file_sha256(diagnostic_path),
        "week5_freeze_sha256": file_sha256(freeze_path),
        "seed": args.seed,
        "baseline_type": args.baseline,
        "limit": args.limit,
    }
    if args.resume and last_path.exists():
        last = load_checkpoint(last_path, map_location=device)
        for key, value in provenance.items():
            if last.get(key) != value:
                raise SystemExit(f"Unsafe baseline resume: {key} drift")
        student.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        start_epoch = int(last["epoch"]) + 1
        best_score = float(last["best_score"])
        best_epoch = int(last["best_epoch"])
        stale = int(last["epochs_without_improvement"])
        history = list(last["history"])
    for epoch in range(start_epoch, int(training["max_epochs"])):
        student.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            empty = normalizer.transform({key: value.to(device) for key, value in batch["empty_model_input"].items()})
            full = normalizer.transform({key: value.to(device) for key, value in batch["full_model_input"].items()})
            targets = {key: value.to(device) for key, value in batch["targets"].items()}
            optimizer.zero_grad(set_to_none=True)
            output = student(empty)
            if args.baseline == "distilled":
                with torch.no_grad():
                    teacher_output = teacher(full)
                loss = distillation_loss(output, teacher_output)
            else:
                loss = diagnostic_loss(
                    output,
                    targets,
                    bit_pos_weight=teacher_checkpoint["bit_pos_weight"],
                    class_weight=teacher_checkpoint["class_weight"],
                    reduction="mean",
                ).total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            batch_count = int(targets["six_way"].shape[0])
            total += float(loss.detach().cpu()) * batch_count
            count += batch_count
        predictions = predict(student, val_loader, normalizer=normalizer, device=device)
        metrics = evaluate_predictions(predictions)
        score = float(metrics["source_bit_macro_f1"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total / count, "validation": metrics})
        payload: Dict[str, Any] = {
            "format_version": CHECKPOINT_VERSION,
            "stage": f"week6_{args.baseline}_baseline",
            **provenance,
            "architecture": teacher_checkpoint["architecture"],
            "normalizer": teacher_checkpoint["normalizer"],
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scientifically_valid": not pilot,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "epochs_without_improvement": stale,
            "history": history,
        }
        save_checkpoint(payload, last_path, overwrite=True)
        if improved:
            save_checkpoint(payload, best_path, overwrite=True)
        LOGGER.info("epoch=%s train_loss=%.6f val_macro_f1=%.4f", epoch, total / count, score)
        if stale >= int(training["early_stopping_patience"]):
            break
    best = load_checkpoint(best_path, map_location=device)
    student.load_state_dict(best["model_state_dict"])
    metrics = evaluate_predictions(predict(student, val_loader, normalizer=normalizer, device=device))
    report: Dict[str, Any] = {
        "format_version": "week6_one_pass_baseline_validation_v1",
        "is_valid": not pilot,
        "fit_split": "train",
        "selection_split": "val",
        "calibration_used": False,
        "test_used": False,
        "baseline_type": args.baseline,
        "metrics": metrics,
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": file_sha256(best_path),
        **provenance,
    }
    report["report_sha256"] = hash_dict(report)
    write_json(report, report_path, overwrite=(args.overwrite or args.resume) and report_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
