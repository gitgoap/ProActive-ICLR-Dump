#!/usr/bin/env python3
"""
Week 3 Pilot Analysis and Configuration Generation Tool (Plan §14, §25.5).

Generates:
1. Grouped severity statistics across (dataset, model, probe, severity).
2. Canonical severity selection (minimizing saturation while targeting sensitivity).
3. All 4 required distribution plots + source bit activation rates:
   - answer_flip_distributions.png
   - confidence_shift_distributions.png
   - entropy_shift_distributions.png
   - latency_distributions.png
   - source_bit_rates_by_dataset.png
4. Exported sample inspection images.
5. pilot_analysis_summary.json.
6. candidate/frozen Week 3 probe configs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from PIL import Image

from proactive.audits.confound_audit import audit_dataset_and_model_confounds
from proactive.audits.pilot_analysis import (
    compute_probe_statistics,
    compute_severity_grid_statistics,
    select_canonical_severities,
    generate_candidate_week3_config,
    compute_full_run_estimates,
)
from proactive.data.manifests import load_manifest
from proactive.probes.image_transforms import (
    CANONICAL_SEVERITIES,
    PILOT_SEVERITIES,
    apply_image_transform,
    derive_transform_seed,
    export_sample_transformed_images,
)
from proactive.utils.io import ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_pilot")


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load lines from JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_all_pilot_plots(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    """Generate all 4 required distribution plots + source bit activations (Plan §25.5)."""
    plot_files = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ensure_dir(output_dir)

        grid_res = compute_severity_grid_statistics(records)
        by_probe_sev = grid_res.get("by_probe_severity", {})

        # Extract per-probe and per-severity data
        probe_keys = list(by_probe_sev.keys())
        if not probe_keys:
            logger.warning("No probe observations available for plotting.")
            return []

        # -------------------------------------------------------------
        # Plot 1: Answer flip distributions (answer_flip_distributions.png)
        # -------------------------------------------------------------
        flip_rates = [by_probe_sev[k]["flip_rate"] for k in probe_keys]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(probe_keys)), flip_rates, color="steelblue", edgecolor="black", alpha=0.85)
        plt.xticks(range(len(probe_keys)), probe_keys, rotation=45, ha="right", fontsize=9)
        plt.ylabel("Answer Flip Rate", fontsize=11)
        plt.ylim(0, 1.0)
        plt.title("Answer Flip Rate Distribution Across Probes & Severities", fontsize=12)
        plt.axhline(0.25, color="red", linestyle="--", alpha=0.7, label="Target Rate (0.25)")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        p1 = output_dir / "answer_flip_distributions.png"
        plt.savefig(p1, dpi=150)
        plt.close()
        plot_files.append(p1)

        # -------------------------------------------------------------
        # Plot 2: Confidence shift distributions (confidence_shift_distributions.png)
        # -------------------------------------------------------------
        conf_means = [by_probe_sev[k]["confidence_shift_distribution"]["mean"] for k in probe_keys]
        conf_stds = [by_probe_sev[k]["confidence_shift_distribution"]["std"] for k in probe_keys]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(probe_keys)), conf_means, yerr=conf_stds, capsize=4, color="coral", edgecolor="black", alpha=0.85)
        plt.xticks(range(len(probe_keys)), probe_keys, rotation=45, ha="right", fontsize=9)
        plt.ylabel("Mean Confidence Shift (Δc)", fontsize=11)
        plt.ylim(-1.0, 1.0)
        plt.axhline(0.0, color="gray", linestyle="-", alpha=0.5)
        plt.title("Confidence Shift Distributions (Mean ± Std)", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        p2 = output_dir / "confidence_shift_distributions.png"
        plt.savefig(p2, dpi=150)
        plt.close()
        plot_files.append(p2)

        # -------------------------------------------------------------
        # Plot 3: Entropy shift distributions (entropy_shift_distributions.png)
        # -------------------------------------------------------------
        ent_means = [by_probe_sev[k]["entropy_shift_distribution"]["mean"] for k in probe_keys]
        ent_stds = [by_probe_sev[k]["entropy_shift_distribution"]["std"] for k in probe_keys]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(probe_keys)), ent_means, yerr=ent_stds, capsize=4, color="mediumpurple", edgecolor="black", alpha=0.85)
        plt.xticks(range(len(probe_keys)), probe_keys, rotation=45, ha="right", fontsize=9)
        plt.ylabel("Mean Entropy Shift (ΔH)", fontsize=11)
        plt.axhline(0.0, color="gray", linestyle="-", alpha=0.5)
        plt.title("Entropy Shift Distributions (Mean ± Std)", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        p3 = output_dir / "entropy_shift_distributions.png"
        plt.savefig(p3, dpi=150)
        plt.close()
        plot_files.append(p3)

        # -------------------------------------------------------------
        # Plot 4: Latency distributions (latency_distributions.png)
        # -------------------------------------------------------------
        lat_means = [by_probe_sev[k]["latency_distribution"]["mean"] for k in probe_keys]
        lat_p95 = [by_probe_sev[k]["latency_distribution"]["p95"] for k in probe_keys]
        plt.figure(figsize=(10, 5))
        x = range(len(probe_keys))
        plt.plot(x, lat_means, marker="o", color="teal", label="Mean Latency (ms)", linewidth=2)
        plt.plot(x, lat_p95, marker="s", linestyle="--", color="crimson", label="p95 Latency (ms)", linewidth=1.5)
        plt.xticks(x, probe_keys, rotation=45, ha="right", fontsize=9)
        plt.ylabel("Latency (ms)", fontsize=11)
        plt.title("Probe Execution Latency Distributions", fontsize=12)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        p4 = output_dir / "latency_distributions.png"
        plt.savefig(p4, dpi=150)
        plt.close()
        plot_files.append(p4)

        # -------------------------------------------------------------
        # Plot 5: Source bit activation rates by dataset
        # -------------------------------------------------------------
        audit = audit_dataset_and_model_confounds(records)
        ds_names = list(audit["by_dataset"].keys())
        if ds_names:
            v_rates = [audit["by_dataset"][d]["visual_bit_rate"] for d in ds_names]
            l_rates = [audit["by_dataset"][d]["language_bit_rate"] for d in ds_names]
            a_rates = [audit["by_dataset"][d]["alignment_bit_rate"] for d in ds_names]

            x_idx = range(len(ds_names))
            width = 0.25
            plt.figure(figsize=(10, 5))
            plt.bar([i - width for i in x_idx], v_rates, width=width, label="b_V (Visual)", color="skyblue")
            plt.bar(x_idx, l_rates, width=width, label="b_L (Language)", color="orange")
            plt.bar([i + width for i in x_idx], a_rates, width=width, label="b_A (Alignment)", color="lightgreen")
            plt.xticks(x_idx, ds_names)
            plt.ylabel("Activation Rate")
            plt.ylim(0, 1.0)
            plt.title("Source Bit Activation Rates Across Datasets")
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            p5 = output_dir / "source_bit_rates_by_dataset.png"
            plt.savefig(p5, dpi=150)
            plt.close()
            plot_files.append(p5)

    except ImportError:
        logger.warning("matplotlib not available; skipping plot generation.")

    return plot_files


def export_pilot_inspection_images(
    records: List[Dict[str, Any]],
    output_dir: Path,
    count_per_probe: int = 50,
) -> Dict[str, List[Path]]:
    """Export transformed images from sample records for human visual review."""
    ensure_dir(output_dir)

    samples = []
    seen = set()
    for r in records:
        inst_id = r.get("instance_id")
        img_p = r.get("image_path")
        if img_p and inst_id not in seen and Path(img_p).exists():
            try:
                img = Image.open(img_p).convert("RGB")
                samples.append((inst_id, img))
                seen.add(inst_id)
                if len(samples) >= count_per_probe:
                    break
            except Exception as e:
                logger.warning(f"Could not load image {img_p}: {e}")

    if not samples:
        logger.warning("No local sample images found to export for inspection.")
        return {}

    return export_sample_transformed_images(samples, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze pilot cache and generate artifacts.")
    parser.add_argument("--pilot_dir", type=str, default="outputs/pilot_cache", help="Pilot cache directory.")
    parser.add_argument("--output_dir", type=str, default="outputs/pilot_reports", help="Reports output directory.")
    parser.add_argument("--freeze", action="store_true", help="Freeze candidate config to frozen_week3_config.yaml.")
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    jsonl_files = list(pilot_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {pilot_dir}")
        return

    all_records = []
    for f in jsonl_files:
        all_records.extend(load_jsonl(f))

    logger.info(f"Loaded {len(all_records)} pilot records across {len(jsonl_files)} files.")

    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    insp_dir = out_dir / "inspection"

    # Compute comprehensive stats
    grid_stats = compute_severity_grid_statistics(all_records)
    selected_severities = select_canonical_severities(grid_stats)
    probe_stats = compute_probe_statistics(all_records)
    audit = audit_dataset_and_model_confounds(all_records)
    plots = generate_all_pilot_plots(all_records, plots_dir)
    exported = export_pilot_inspection_images(all_records, insp_dir)
    estimates = compute_full_run_estimates(all_records)

    # Save summary report
    report = {
        "total_records": len(all_records),
        "selected_canonical_severities": selected_severities,
        "probe_statistics": probe_stats,
        "severity_grid_statistics": grid_stats,
        "confound_audit": audit,
        "plots_generated": [str(p) for p in plots],
        "full_run_estimates": estimates,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pilot_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved pilot analysis summary to {summary_path}")

    # Emit candidate configuration
    cand_path = Path("configs/probes/candidate_week3_config.yaml")
    generate_candidate_week3_config(probe_stats, audit, cand_path, selected_severities=selected_severities)

    if args.freeze:
        frozen_path = Path("configs/probes/frozen_week3_config.yaml")
        generate_candidate_week3_config(probe_stats, audit, frozen_path, selected_severities=selected_severities)
        logger.info(f"FROZEN Week 3 configuration saved to {frozen_path}")


if __name__ == "__main__":
    main()
