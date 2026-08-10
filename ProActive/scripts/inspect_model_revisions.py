#!/usr/bin/env python3
"""Inspect local model directories for immutable revision provenance.

This script is read-only. It never substitutes a current remote `main` hash for
the revision of local weights; unresolved provenance must be repaired before a
full Week 4 run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set

import yaml


REVISION_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{40})(?![0-9a-fA-F])")


def inspect_path(path: Path) -> Dict[str, object]:
    candidates: Set[str] = set()
    evidence: List[str] = []
    resolved = path.resolve()
    for part in resolved.parts:
        if re.fullmatch(r"[0-9a-fA-F]{40}", part):
            candidates.add(part.lower())
            evidence.append(f"resolved_path_component:{part}")

    if (path / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        revision = result.stdout.strip()
        if result.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            candidates.add(revision.lower())
            evidence.append("git:HEAD")

    metadata_roots = [path / ".cache" / "huggingface" / "download", path]
    metadata_files: List[Path] = []
    if metadata_roots[0].exists():
        metadata_files.extend(metadata_roots[0].glob("*.metadata"))
    for name in ("REVISION", "revision.txt", ".proactive_revision"):
        candidate_file = path / name
        if candidate_file.exists():
            metadata_files.append(candidate_file)
    for metadata_file in metadata_files:
        try:
            text = metadata_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matches = REVISION_RE.findall(text)
        for revision in matches:
            candidates.add(revision.lower())
            evidence.append(f"metadata:{metadata_file}")

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=True
        )
        revision = getattr(config, "_commit_hash", None)
        if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            candidates.add(revision.lower())
            evidence.append("transformers:config._commit_hash")
    except Exception as exc:
        evidence.append(f"transformers_error:{type(exc).__name__}")

    status = "RESOLVED" if len(candidates) == 1 else "AMBIGUOUS" if candidates else "UNRESOLVED"
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "status": status,
        "revision_candidates": sorted(candidates),
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_configs",
        nargs="+",
        required=True,
        help="One or more model YAML configurations",
    )
    args = parser.parse_args()
    report = {}
    failed = False
    for config_name in args.model_configs:
        with open(config_name, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        result = inspect_path(Path(config["model_path"]))
        result["configured_revision"] = config.get("model_revision")
        result["model_id"] = config.get("model_id")
        report[config_name] = result
        failed = failed or result["status"] != "RESOLVED"
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

