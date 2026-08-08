"""
Week 3 Pilot Analysis and Configuration Generation Core (Plan §14, §25.5).

Computes comprehensive distributional statistics grouped by (dataset, model, probe, severity),
evaluates saturation and invalid rates, implements automated canonical severity selection,
and outputs configuration and report artifacts.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from proactive.probes.image_transforms import (
    CANONICAL_SEVERITIES,
    PILOT_SEVERITIES,
)

logger = logging.getLogger(__name__)


def _percentiles(values: List[float]) -> Dict[str, float]:
    """Compute summary percentiles from a list of floats."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n
    variance = sum((x - mean_val) ** 2 for x in sorted_vals) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)

    def get_p(p: float) -> float:
        idx = int(p * (n - 1))
        return sorted_vals[idx]

    return {
        "mean": mean_val,
        "std": std_val,
        "min": sorted_vals[0],
        "p25": get_p(0.25),
        "median": get_p(0.50),
        "p75": get_p(0.75),
        "p95": get_p(0.95),
        "max": sorted_vals[-1],
    }


def compute_severity_grid_statistics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Group statistics by (dataset, model, probe, severity) and aggregate across probes."""
    # Key: (dataset, model_id, probe, severity)
    grouped: Dict[Tuple[str, str, str, Optional[float]], Dict[str, Any]] = defaultdict(
        lambda: {
            "total_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "flips": [],
            "conf_shifts": [],
            "entropy_shifts": [],
            "margin_shifts": [],
            "latencies": [],
            "saturated_count": 0,
        }
    )

    # Key: (probe, severity) aggregated across datasets/models
    agg_probe_sev: Dict[Tuple[str, Optional[float]], Dict[str, Any]] = defaultdict(
        lambda: {
            "total_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "flips": [],
            "conf_shifts": [],
            "entropy_shifts": [],
            "margin_shifts": [],
            "latencies": [],
            "saturated_count": 0,
        }
    )

    for r in records:
        dataset = r.get("dataset", "unknown")
        model_id = r.get("model_id", "unknown")
        is_grid_row = "pilot_severity_probe" in r
        grid_probe = r.get("pilot_severity_probe")
        grid_sev = r.get("pilot_severity_value")

        probes = r.get("probes", {})
        for p_name, p_obs in probes.items():
            # If grid row, only focus on the varied probe
            if is_grid_row and p_name != grid_probe:
                continue
            if not p_obs.get("applicable", True):
                continue

            sev = grid_sev if (is_grid_row and p_name == grid_probe) else p_obs.get("severity")
            is_valid = p_obs.get("valid", True)
            flip_val = 1.0 if p_obs.get("flip", False) else 0.0
            conf_shift = p_obs.get("conf_shift", 0.0)
            ent_shift = p_obs.get("entropy_shift", 0.0)
            mrg_shift = p_obs.get("margin_shift", 0.0)
            lat_ms = p_obs.get("latency_ms")

            # Saturation is an extreme confidence displacement, not a single
            # answer flip.  Treating every flip as saturation made the target
            # 15--40% sensitivity range mathematically self-defeating.
            is_saturated = bool(abs(conf_shift) > 0.90)

            for target_dict in [grouped[(dataset, model_id, p_name, sev)], agg_probe_sev[(p_name, sev)]]:
                target_dict["total_count"] += 1
                if is_valid:
                    target_dict["valid_count"] += 1
                    target_dict["flips"].append(flip_val)
                    target_dict["conf_shifts"].append(conf_shift)
                    target_dict["entropy_shifts"].append(ent_shift)
                    target_dict["margin_shifts"].append(mrg_shift)
                    if lat_ms is not None:
                        target_dict["latencies"].append(lat_ms)
                    if is_saturated:
                        target_dict["saturated_count"] += 1
                else:
                    target_dict["invalid_count"] += 1

    # Summarize aggregated probe-severity statistics
    summary_by_probe_sev: Dict[str, Dict[str, Any]] = {}
    for (p_name, sev), data in sorted(agg_probe_sev.items(), key=lambda x: (x[0][0], x[0][1] or 0.0)):
        sev_key = f"{p_name}@sev={sev}" if sev is not None else p_name
        vc = max(1, data["valid_count"])
        tc = max(1, data["total_count"])
        summary_by_probe_sev[sev_key] = {
            "probe": p_name,
            "severity": sev,
            "total_count": data["total_count"],
            "valid_count": data["valid_count"],
            "invalid_count": data["invalid_count"],
            "invalid_rate": data["invalid_count"] / tc,
            "saturation_rate": data["saturated_count"] / vc,
            "flip_rate": sum(data["flips"]) / vc if data["flips"] else 0.0,
            "flip_distribution": _percentiles(data["flips"]),
            "confidence_shift_distribution": _percentiles(data["conf_shifts"]),
            "entropy_shift_distribution": _percentiles(data["entropy_shifts"]),
            "margin_shift_distribution": _percentiles(data["margin_shifts"]),
            "latency_distribution": _percentiles(data["latencies"]),
        }

    return {
        "by_probe_severity": summary_by_probe_sev,
        "raw_grouped_count": len(grouped),
    }


def select_canonical_severities(
    grid_stats: Dict[str, Any],
    fallback_severities: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Select the severity s* that maximizes dynamic sensitivity without over-saturating.

    Selection criteria per probe:
    1. invalid_rate <= 0.05
    2. saturation_rate <= 0.15
    3. flip_rate in target range [0.15, 0.40] (or closest to 0.25)
    """
    fallback = fallback_severities or CANONICAL_SEVERITIES
    selected = dict(fallback)

    by_probe: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sev_key, info in grid_stats.get("by_probe_severity", {}).items():
        if info.get("severity") is not None:
            by_probe[info["probe"]].append(info)

    for probe, candidates in by_probe.items():
        if not candidates:
            continue

        # Invalid and saturation constraints are hard gates.  A soft penalty
        # could otherwise select a scientifically disallowed severity.
        eligible = [
            c for c in candidates
            if c["invalid_rate"] <= 0.05 and c["saturation_rate"] <= 0.15
        ]
        target_candidates = [c for c in eligible if 0.15 <= c["flip_rate"] <= 0.40]
        ranked = target_candidates or eligible
        best_candidate = None
        best_score = float("inf")
        if ranked:
            best = min(ranked, key=lambda c: (abs(c["flip_rate"] - 0.25), c["severity"]))
            best_candidate = best["severity"]
            best_score = abs(best["flip_rate"] - 0.25)
        else:
            logger.warning(
                "No eligible severity for '%s'; retaining pre-registered fallback %s",
                probe,
                fallback.get(probe),
            )

        if best_candidate is not None:
            selected[probe] = best_candidate
            logger.info(f"Selected canonical severity for '{probe}': {best_candidate} (score={best_score:.4f})")

    return selected


def compute_probe_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for each probe across all records."""
    grid_res = compute_severity_grid_statistics(records)
    summary_by_probe: Dict[str, Any] = {}

    # Extract single overall stats per probe
    for sev_key, info in grid_res["by_probe_severity"].items():
        p_name = info["probe"]
        if p_name not in summary_by_probe:
            summary_by_probe[p_name] = {
                "total_count": info["total_count"],
                "valid_count": info["valid_count"],
                "invalid_rate": info["invalid_rate"],
                "flip_rate": info["flip_rate"],
                "mean_conf_shift": info["confidence_shift_distribution"]["mean"],
                "mean_abs_conf_shift": abs(info["confidence_shift_distribution"]["mean"]),
                "mean_entropy_shift": info["entropy_shift_distribution"]["mean"],
                "mean_latency_ms": info["latency_distribution"]["mean"],
            }

    return summary_by_probe


def generate_candidate_week3_config(
    probe_stats: Dict[str, Any],
    confound_audit: Dict[str, Any],
    output_config_path: Path,
    selected_severities: Optional[Dict[str, float]] = None,
    semantic_threshold: float = 0.82,
    frozen: bool = False,
    semantic_calibration: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate candidate Week 3 configuration file."""
    severities = selected_severities or CANONICAL_SEVERITIES

    config = {
        "metadata": {
            "version": "v3.5_week3_frozen" if frozen else "v3.5_week3_candidate",
            "status": "FROZEN" if frozen else "CANDIDATE_FOR_AUDIT",
            "description": "Canonical probe severities, label thresholds, and semantic matching parameters.",
        },
        "probe_severities": severities,
        "pilot_severity_grid": PILOT_SEVERITIES,
        "label_thresholds": {
            "visual_flip_threshold": 0.25,
            "visual_conf_threshold": 0.15,
            "blank_semantic_match_required": True,
            "blank_conf_ratio_threshold": 0.80,
            "grounding_flip_triggers": True,
            "grounding_conf_threshold": 0.15,
        },
        "semantic_matching": {
            "binary_matching": "exact_normalized",
            "freeform_matching": "embedding_similarity",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "e4ce9877abf3edee10b0257f22713854020a4004",
            "threshold": semantic_threshold,
            "calibration_split": "train_val",
            "calibration_report": semantic_calibration,
        },
        "summary_statistics": probe_stats,
    }

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=output_config_path.parent,
        suffix=".tmp",
        prefix=output_config_path.stem + "_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_config_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    logger.info(f"Wrote candidate configuration to {output_config_path}")
    return output_config_path


def compute_full_run_estimates(
    records: List[Dict[str, Any]],
    total_manifest_examples: int = 8291,
    total_models: int = 3,
    gpus: int = 4,
    total_forward_passes: Optional[int] = None,
) -> Dict[str, Any]:
    """Project the core full run from measured per-forward-pass latency."""
    if gpus <= 0:
        raise ValueError("gpus must be positive")
    latencies_ms: List[float] = []
    passes_per_instance: List[int] = []
    for r in records:
        instance_passes = 0
        clean_latency = r.get("clean", {}).get("latency_ms")
        if is_finite_number(clean_latency) and clean_latency > 0:
            latencies_ms.append(float(clean_latency))
            instance_passes += 1
        for observation in r.get("probes", {}).values():
            latency = observation.get("latency_ms")
            if is_finite_number(latency) and latency > 0 and observation.get("applicable", True):
                latencies_ms.append(float(latency))
                instance_passes += 1
        if instance_passes:
            passes_per_instance.append(instance_passes)

    if not latencies_ms or not passes_per_instance:
        raise ValueError("No positive finite pilot latencies available for projection")

    mean_sec_per_pass = sum(latencies_ms) / len(latencies_ms) / 1000.0
    mean_passes_per_instance = sum(passes_per_instance) / len(passes_per_instance)
    if total_forward_passes is None:
        if total_manifest_examples == 8291 and total_models == 3:
            total_forward_passes = 178131
        else:
            total_forward_passes = math.ceil(
                total_manifest_examples * total_models * mean_passes_per_instance
            )

    projected_gpu_hours = total_forward_passes * mean_sec_per_pass / 3600.0
    total_hours = projected_gpu_hours / gpus

    return {
        "measured_mean_seconds_per_forward_pass": mean_sec_per_pass,
        "measured_mean_passes_per_instance": mean_passes_per_instance,
        "total_manifest_examples": total_manifest_examples,
        "total_models": total_models,
        "total_forward_passes": total_forward_passes,
        "gpu_count": gpus,
        "projected_gpu_hours": projected_gpu_hours,
        "projected_total_hours": total_hours,
        "projected_total_days": total_hours / 24.0,
    }


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
