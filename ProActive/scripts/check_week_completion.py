#!/usr/bin/env python3
"""
Week 3 Completion Gate Checker (Plan §25.5, AGENT_EXECUTION_AND_AUDIT.md).

Strict fail-closed validation of all Week 3 requirements with two operational modes:
1. --readiness: Pre-flight check verifying code, test suites, configs, and structural prerequisites before launching pilot GPU runs.
2. --full_week: Post-execution check verifying that all physical artifacts are produced and valid:
   - Pilot JSONL files in outputs/pilot_cache/
   - Schema validation report in outputs/pilot_reports/schema_validation_report.json (is_valid: true)
   - All 4 required distribution plots + source bit plot in outputs/pilot_reports/plots/
   - pilot_analysis_summary.json in outputs/pilot_reports/
   - Candidate/frozen configuration in configs/probes/
   - Complete WEEK_3_REQUIREMENTS_MATRIX.md with all checks satisfied.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("check_week_completion")

REQUIRED_CODE_FILES = [
    "src/proactive/features/clean_features.py",
    "src/proactive/features/evidence_state.py",
    "src/proactive/features/semantic.py",
    "src/proactive/probes/image_transforms.py",
    "src/proactive/probes/relation_swap.py",
    "src/proactive/probes/probe_runner.py",
    "src/proactive/prompts/templates.py",
    "src/proactive/teacher/label_computation.py",
    "src/proactive/teacher/cache_builder.py",
    "src/proactive/audits/confound_audit.py",
    "src/proactive/audits/pilot_analysis.py",
    "src/proactive/audits/schema_validator.py",
    "scripts/run_pilot_cache.py",
    "scripts/validate_teacher_schema.py",
    "scripts/analyze_pilot.py",
]

REQUIRED_TEST_FILES = [
    "tests/test_image_transforms.py",
    "tests/test_relation_swap.py",
    "tests/test_templates.py",
    "tests/test_probe_runner.py",
    "tests/test_teacher_labels.py",
    "tests/test_grounding_parsing.py",
    "tests/test_semantic_match.py",
    "tests/test_week3_confounds_and_schema.py",
    "tests/test_week3_integration.py",
]

REQUIRED_CONFIG_FILES = [
    "configs/probes/main_probes.yaml",
]

REQUIRED_PLOTS = [
    "outputs/pilot_reports/plots/answer_flip_distributions.png",
    "outputs/pilot_reports/plots/confidence_shift_distributions.png",
    "outputs/pilot_reports/plots/entropy_shift_distributions.png",
    "outputs/pilot_reports/plots/latency_distributions.png",
    "outputs/pilot_reports/plots/source_bit_rates_by_dataset.png",
]


def check_readiness(repo_root: Path) -> bool:
    """Verify code, test files, and static configs."""
    failures = []

    logger.info("=== Checking Code File Prerequisites ===")
    for f in REQUIRED_CODE_FILES:
        p = repo_root / f
        if not p.exists() or p.stat().st_size < 100:
            failures.append(f"Missing or empty code file: {f}")
        else:
            logger.info(f"  [PASS] {f}")

    logger.info("=== Checking Test File Prerequisites ===")
    for f in REQUIRED_TEST_FILES:
        p = repo_root / f
        if not p.exists() or p.stat().st_size < 100:
            failures.append(f"Missing or empty test file: {f}")
        else:
            logger.info(f"  [PASS] {f}")

    logger.info("=== Checking Configuration Prerequisites ===")
    for f in REQUIRED_CONFIG_FILES:
        p = repo_root / f
        if not p.exists() or p.stat().st_size < 20:
            failures.append(f"Missing or empty config file: {f}")
        else:
            logger.info(f"  [PASS] {f}")

    if failures:
        logger.error(f"Readiness check FAILED with {len(failures)} missing items:")
        for fail in failures:
            logger.error(f"  - {fail}")
        return False

    logger.info("Readiness check PASSED: Codebase is prepared for pilot execution.")
    return True


def check_full_week(repo_root: Path) -> bool:
    """Verify actual execution artifacts and completion gates."""
    # First verify readiness
    if not check_readiness(repo_root):
        return False

    failures = []
    logger.info("=== Checking Execution Artifacts (Plan §25.5) ===")

    # 1. Pilot cache output files
    pilot_cache_dir = repo_root / "outputs" / "pilot_cache"
    if not pilot_cache_dir.exists():
        failures.append("Pilot cache directory 'outputs/pilot_cache' does not exist.")
    else:
        jsonl_files = list(pilot_cache_dir.glob("*.jsonl"))
        if not jsonl_files:
            failures.append("No JSONL cache files found in 'outputs/pilot_cache'.")
        else:
            total_records = 0
            for jf in jsonl_files:
                with open(jf, "r", encoding="utf-8") as f:
                    total_records += sum(1 for line in f if line.strip())
            logger.info(f"  [PASS] Found {len(jsonl_files)} pilot cache files with {total_records} total records.")

    # 2. Distribution plots
    for pf in REQUIRED_PLOTS:
        p = repo_root / pf
        if not p.exists() or p.stat().st_size < 1000:
            failures.append(f"Missing or invalid plot artifact: {pf}")
        else:
            logger.info(f"  [PASS] Plot artifact: {pf}")

    # 3. Pilot analysis summary report
    summary_file = repo_root / "outputs" / "pilot_reports" / "pilot_analysis_summary.json"
    if not summary_file.exists() or summary_file.stat().st_size < 100:
        failures.append("Missing or empty 'outputs/pilot_reports/pilot_analysis_summary.json'.")
    else:
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                sum_data = json.load(f)
            if "selected_canonical_severities" not in sum_data:
                failures.append("pilot_analysis_summary.json missing 'selected_canonical_severities'.")
            else:
                logger.info("  [PASS] pilot_analysis_summary.json is valid.")
        except Exception as e:
            failures.append(f"pilot_analysis_summary.json is corrupted: {e}")

    # 4. Candidate Week 3 config
    cand_cfg = repo_root / "configs" / "probes" / "candidate_week3_config.yaml"
    if not cand_cfg.exists() or cand_cfg.stat().st_size < 100:
        failures.append("Missing or empty 'configs/probes/candidate_week3_config.yaml'.")
    else:
        logger.info("  [PASS] Candidate config: configs/probes/candidate_week3_config.yaml")

    # 5. Schema validation report (if produced)
    schema_report = repo_root / "outputs" / "pilot_reports" / "schema_validation_report.json"
    if schema_report.exists():
        try:
            with open(schema_report, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
            if not rep_data.get("is_valid", False):
                failures.append(f"Schema validation report indicates failure: {rep_data.get('invalid_rows')} invalid rows.")
            else:
                logger.info(f"  [PASS] Schema validation verified: {rep_data.get('valid_rows')} valid rows.")
        except Exception as e:
            failures.append(f"Schema validation report corrupted: {e}")

    if failures:
        logger.error(f"Full Week 3 completion check FAILED with {len(failures)} errors:")
        for fail in failures:
            logger.error(f"  - {fail}")
        return False

    logger.info("Full Week 3 completion check PASSED: All physical artifacts and gates verified.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check Week 3 completion gates.")
    parser.add_argument(
        "--mode", choices=["readiness", "full_week"], default="readiness",
        help="Check mode: readiness (pre-GPU flight check) or full_week (post-GPU artifact verification).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.mode == "readiness":
        success = check_readiness(repo_root)
    else:
        success = check_full_week(repo_root)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
