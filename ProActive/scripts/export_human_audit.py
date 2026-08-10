#!/usr/bin/env python3
"""Export the blinded 180-example Week 4 human-audit packet."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.teacher.cache_builder import _load_image_safely
from proactive.audits.human_audit import LABELS, audit_rank, select_audit_keys
from proactive.teacher.offline import teacher_key
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl, write_text


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_human_audit")

ANNOTATION_FIELDS = (
    "clean_correct",
    "visual_fragile",
    "language_persistence",
    "alignment_instability",
    "label6",
    "insufficient_or_contradictory",
)

README_TEXT = """# ProActive Week 4 human audit

Purpose: validate whether the rule-backed labels are understandable from the
observed model behaviour. This is a behavioural audit, not a causal-source
annotation task.

Use only `human_audit_blinded.csv` and the `images/` directory while annotating.
Do not open `human_audit_private_key.jsonl`; it contains the hidden dataset,
model, teacher scores, and teacher label used only for later analysis.

Three annotators independently fill their own six columns (`ann1_*`, `ann2_*`,
or `ann3_*`). Use 1/0 for the four binary questions and one of
`no-failure`, `visual`, `language-prior`, `alignment`, `mixed`, or `unclear`
for `label6`. Leave `adjudicated_label6` empty until independent annotation is
finished, then resolve disagreements without consulting the private key.

The images are losslessly materialized and renamed by audit ID so their source
dataset is not visible. This packet is for internal research annotation only;
do not redistribute dataset images.
"""


def _jsonl_files(path: Path, prefix: str) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob(f"{prefix}*.jsonl"))
        return files or sorted(path.glob("*.jsonl"))
    raise FileNotFoundError(path)


def _read_unique(path: Path, prefix: str) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Path, str]]:
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    hashes: Dict[Path, str] = {}
    for file_path in _jsonl_files(path, prefix):
        hashes[file_path] = file_sha256(file_path)
        for row in iter_jsonl(file_path):
            key = teacher_key(row)
            if key in rows:
                raise ValueError(f"Duplicate key across {prefix} shards: {key}")
            rows[key] = row
    return rows, hashes


def _probe_table(teacher: Mapping[str, Any]) -> str:
    rows = []
    for probe_name in sorted(teacher["probes"]):
        probe = teacher["probes"][probe_name]
        rows.append(
            {
                "probe": probe_name,
                "answer": probe.get("norm_answer", ""),
                "applicable": bool(probe.get("applicable")),
                "valid": bool(probe.get("valid")),
            }
        )
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _write_csv_atomic(rows: Sequence[Mapping[str, Any]], path: Path, fieldnames: Sequence[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem + "_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise


def _validate_resume(output_dir: Path) -> bool:
    manifest_path = output_dir / "human_audit_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Audit manifest is missing")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for relative, expected_hash in manifest.get("artifact_sha256", {}).items():
        path = output_dir / relative
        if not path.exists() or file_sha256(path) != expected_hash:
            raise ValueError(f"Audit artifact missing or changed: {relative}")
    if manifest.get("materialized_image_count") != manifest.get("selected_count"):
        raise ValueError("Audit packet does not contain all materialized images")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--manifest_path", help="Accepted for script-contract parity")
    parser.add_argument("--output_dir", default="outputs/human_audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Optional cap on eligible records for testing")
    parser.add_argument("--device", default="cpu", help="Accepted for script-contract parity")
    parser.add_argument("--no_materialize_images", action="store_true")
    parser.add_argument("--allow_incomplete_model_coverage", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    with open(experiment["frozen_probe_config"], "r", encoding="utf-8") as handle:
        frozen = yaml.safe_load(handle)
    audit_config = experiment["human_audit"]
    output_dir = Path(args.output_dir)
    if args.resume and output_dir.exists():
        try:
            _validate_resume(output_dir)
        except (ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Unsafe audit resume refused: {exc}") from exc
        logger.info("Validated complete audit packet; skipping")
        return
    existing = [
        output_dir / "human_audit_blinded.csv",
        output_dir / "human_audit_private_key.jsonl",
        output_dir / "human_audit_manifest.json",
    ]
    if any(path.exists() for path in existing) and not args.overwrite:
        raise SystemExit("Audit output exists. Use --resume or --overwrite.")

    teachers, teacher_hashes = _read_unique(Path(args.teacher_path), "teacher_")
    labels, label_hashes = _read_unique(Path(args.labels_path), "labels_")
    if args.limit is not None:
        keep = set(
            sorted(labels, key=lambda key: audit_rank(args.seed, "limit", key))[
                : args.limit
            ]
        )
        labels = {key: value for key, value in labels.items() if key in keep}

    observed_models = {label["model_id"] for label in labels.values()}
    required_model_keys = list(audit_config["required_model_keys"])
    configured_models = experiment["models"]
    required_model_ids = set()
    for model_key in required_model_keys:
        with open(configured_models[model_key], "r", encoding="utf-8") as handle:
            required_model_ids.add(yaml.safe_load(handle)["model_id"])
    missing_models = required_model_ids - observed_models
    if missing_models and not args.allow_incomplete_model_coverage:
        raise SystemExit(
            f"Audit requires all configured models; missing {sorted(missing_models)}. "
            "Use --allow_incomplete_model_coverage only for a clearly marked interim packet."
        )

    selected, provenance = select_audit_keys(
        teachers=teachers,
        labels=labels,
        total=int(audit_config["total_examples"]),
        natural_count=int(audit_config["natural_subset_examples"]),
        target_per_label=int(audit_config["target_per_label"]),
        seed=args.seed,
        thresholds=frozen["label_thresholds"],
        allowed_splits=set(audit_config["allowed_splits"]),
    )
    distribution = Counter(labels[key]["teacher_label6"] for key in selected)
    logger.info("Selected %s audit rows: %s", len(selected), dict(distribution))
    if args.dry_run:
        return

    image_dir = output_dir / "images"
    if not args.no_materialize_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    blinded_rows: List[Dict[str, Any]] = []
    private_rows: List[Dict[str, Any]] = []
    image_hashes: Dict[str, str] = {}
    for index, key in enumerate(selected, start=1):
        audit_id = f"audit_{index:04d}"
        teacher = teachers[key]
        label = labels[key]
        image_relative = f"images/{audit_id}.png"
        if not args.no_materialize_images:
            image = _load_image_safely(teacher["image_path"], teacher["dataset"])
            image_path = output_dir / image_relative
            fd, temporary_image = tempfile.mkstemp(
                dir=image_dir, prefix=audit_id + "_", suffix=".tmp.png"
            )
            os.close(fd)
            try:
                image.save(temporary_image, format="PNG")
                os.replace(temporary_image, image_path)
            except BaseException:
                if os.path.exists(temporary_image):
                    os.remove(temporary_image)
                raise
            image_hashes[image_relative] = file_sha256(image_path)

        blinded: Dict[str, Any] = {
            "audit_id": audit_id,
            "image_file": image_relative,
            "question": teacher.get("question", teacher["prompt_text"]),
            "gold_answer": teacher["gold_answer"],
            "clean_answer": teacher["clean"]["raw_answer"],
            "probe_results_json": _probe_table(teacher),
        }
        for annotator in ("ann1", "ann2", "ann3"):
            for field in ANNOTATION_FIELDS:
                blinded[f"{annotator}_{field}"] = ""
        blinded["adjudicated_label6"] = ""
        blinded_rows.append(blinded)

        private_rows.append(
            {
                "audit_id": audit_id,
                "instance_id": teacher["instance_id"],
                "group_id": teacher["group_id"],
                "dataset": teacher["dataset"],
                "split": teacher["split"],
                "model_id": teacher["model_id"],
                "model_revision": teacher["model_revision"],
                "source_image_path": teacher["image_path"],
                "teacher_signature": label["teacher_signature"],
                "teacher_bits": label["teacher_bits"],
                "teacher_label6": label["teacher_label6"],
                "selection_subset": provenance[key],
                "source_teacher_record_sha256": label["source_teacher_record_sha256"],
                "source_label_record_sha256": label["label_record_sha256"],
            }
        )

    fieldnames = list(blinded_rows[0])
    blinded_path = output_dir / "human_audit_blinded.csv"
    private_path = output_dir / "human_audit_private_key.jsonl"
    readme_path = output_dir / "README.md"
    _write_csv_atomic(blinded_rows, blinded_path, fieldnames, overwrite=args.overwrite)
    write_jsonl(private_rows, private_path, overwrite=args.overwrite and private_path.exists())
    write_text(
        README_TEXT,
        readme_path,
        overwrite=args.overwrite and readme_path.exists(),
    )

    artifact_hashes = {
        "human_audit_blinded.csv": file_sha256(blinded_path),
        "human_audit_private_key.jsonl": file_sha256(private_path),
        "README.md": file_sha256(readme_path),
        **image_hashes,
    }
    manifest = {
        "record_type": "human_audit_manifest",
        "selected_count": len(selected),
        "natural_subset_count": sum(value == "natural" for value in provenance.values()),
        "is_interim": bool(missing_models),
        "materialized_image_count": len(image_hashes),
        "teacher_source_sha256": sorted(teacher_hashes.values()),
        "label_source_sha256": sorted(label_hashes.values()),
        "artifact_sha256": artifact_hashes,
    }
    write_json(
        manifest,
        output_dir / "human_audit_manifest.json",
        overwrite=args.overwrite and (output_dir / "human_audit_manifest.json").exists(),
    )


if __name__ == "__main__":
    main()
