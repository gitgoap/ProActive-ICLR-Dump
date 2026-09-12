#!/usr/bin/env python3
"""Download, safely extract, and verify pinned Week 8 held-out releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.data.heldout import load_illusionbench_release, load_prehal_release
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


LOGGER = logging.getLogger("setup_heldout_datasets")
DATASET_CONFIGS = {
    "prehal": Path("configs/data/prehal.yaml"),
    "illusionbench": Path("configs/data/illusionbench.yaml"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_CONFIGS),
        default=list(DATASET_CONFIGS),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--output_report",
        default="outputs/week8_data/dataset_setup_report.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Dataset config is not a mapping: {path}")
    return value


def _target(data_root: Path, dataset: str) -> Path:
    return data_root / ("PREHAL" if dataset == "prehal" else "IllusionBench")


def _download(config: Mapping[str, Any], target: Path) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required only for --download. Install it with "
            "`python -m pip install huggingface_hub`."
        ) from exc
    target.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=str(config["repo_id"]),
        repo_type="dataset",
        revision=str(config["revision"]),
        local_dir=str(target),
        allow_patterns=list(config["download_allow_patterns"]),
    )
    return str(Path(downloaded).resolve())


def _safe_remove_directory(path: Path, parent: Path) -> None:
    resolved_parent = parent.resolve()
    resolved = path.resolve()
    if resolved.parent != resolved_parent or resolved.name != "images":
        raise RuntimeError(f"Refusing unsafe cleanup target: {resolved}")
    shutil.rmtree(resolved)


def _extract_illusionbench(
    root: Path,
    config: Mapping[str, Any],
    *,
    overwrite: bool,
) -> int:
    archive = root / str(config["archive_file"])
    if not archive.is_file():
        raise FileNotFoundError(f"IllusionBench archive is missing: {archive}")
    actual_sha = file_sha256(archive)
    if actual_sha != config["archive_sha256"]:
        raise ValueError(
            "IllusionBench archive SHA-256 mismatch: "
            f"expected {config['archive_sha256']}, found {actual_sha}"
        )
    output = root / "images"
    if output.exists() and not overwrite:
        LOGGER.info("Extracted IllusionBench images already exist; validating them")
        return len(
            [
                path
                for path in output.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".png"
                and not path.name.startswith("._")
            ]
        )

    root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(dir=root, prefix=".illusion_extract_"))
    temporary_images = temporary_root / "images"
    temporary_images.mkdir()
    extracted: set[str] = set()
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                normalized = member.filename.replace("\\", "/")
                parts = Path(normalized).parts
                if member.is_dir() or ".." in parts or normalized.startswith("/"):
                    continue
                if normalized.startswith("__MACOSX/"):
                    continue
                name = Path(normalized).name
                if (
                    len(parts) != 2
                    or parts[0] != "IllusionDataset"
                    or not name.lower().endswith(".png")
                    or name.startswith("._")
                ):
                    continue
                if name in extracted:
                    raise ValueError(f"Duplicate real image in IllusionBench ZIP: {name}")
                with bundle.open(member) as source, open(temporary_images / name, "wb") as sink:
                    shutil.copyfileobj(source, sink)
                extracted.add(name)
        expected = int(config["expected_archive_images"])
        if len(extracted) != expected:
            raise ValueError(
                f"IllusionBench ZIP inventory drift: expected {expected}, found {len(extracted)}"
            )
        if output.exists():
            _safe_remove_directory(output, root)
        os.replace(temporary_images, output)
        return len(extracted)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _tree_content_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if not (args.download or args.extract or args.verify or args.dry_run):
        raise SystemExit("Choose at least one of --download, --extract, --verify, or --dry_run")
    data_root = Path(args.data_root).expanduser()
    report_path = Path(args.output_report)
    configs = {name: _load_config(DATASET_CONFIGS[name]) for name in args.datasets}
    plan = {
        name: {
            "repo_id": config["repo_id"],
            "revision": config["revision"],
            "target": str(_target(data_root, name)),
            "allow_patterns": list(config["download_allow_patterns"]),
        }
        for name, config in configs.items()
    }
    if args.dry_run:
        print(json.dumps({"is_valid": True, "plan": plan}, indent=2))
        return

    results: Dict[str, Any] = {}
    for name, config in configs.items():
        target = _target(data_root, name)
        LOGGER.info("Preparing %s at %s", name, target)
        downloaded_path = None
        if args.download:
            downloaded_path = _download(config, target)
        extracted_count = None
        if name == "illusionbench" and args.extract:
            extracted_count = _extract_illusionbench(
                target, config, overwrite=args.overwrite
            )
        if args.verify:
            if name == "prehal":
                loaded = load_prehal_release(target, config)
                image_paths = sorted(
                    path
                    for path in (target / "images").rglob("*")
                    if path.is_file() and not path.name.startswith(".")
                )
            else:
                loaded = load_illusionbench_release(target, config)
                image_dir = Path(loaded.audit["extracted_image_dir"])
                image_paths = [
                    path
                    for path in image_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() == ".png"
                    and not path.name.startswith("._")
                ]
            audit = dict(loaded.audit)
            audit["release_image_content_sha256"] = _tree_content_hash(
                image_paths, target
            )
            audit["release_image_file_count"] = len(image_paths)
            audit["exclusion_count"] = len(loaded.exclusions)
        else:
            audit = {"verification_skipped": True}
        results[name] = {
            "target": str(target),
            "downloaded_snapshot_path": downloaded_path,
            "extracted_image_count": extracted_count,
            "audit": audit,
        }

    report: Dict[str, Any] = {
        "format_version": "heldout_dataset_setup_v1",
        "is_valid": args.verify,
        "network_streaming_used_for_inference": False,
        "results": results,
    }
    report["report_sha256"] = hash_dict(report)
    write_json(
        report,
        report_path,
        overwrite=args.overwrite and report_path.exists(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
