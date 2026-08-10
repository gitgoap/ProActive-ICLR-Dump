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
from typing import Dict, List, Optional, Set

import yaml


REVISION_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{40})(?![0-9a-fA-F])")
EXACT_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")


def huggingface_metadata_revision(text: str) -> Optional[str]:
    """Return the repository commit stored in Hugging Face local-dir metadata.

    ``huggingface_hub`` writes download metadata as three lines: repository
    commit, file ETag, and timestamp. The ETag is often also a 40-character
    hexadecimal Git blob ID, but it is not a model revision. Only the first
    non-empty line is therefore valid revision evidence.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not EXACT_REVISION_RE.fullmatch(lines[0]):
        return None
    return lines[0].lower()


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

    huggingface_metadata_dir = path / ".cache" / "huggingface" / "download"
    if huggingface_metadata_dir.exists():
        for metadata_file in sorted(huggingface_metadata_dir.glob("*.metadata")):
            try:
                text = metadata_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            revision = huggingface_metadata_revision(text)
            if revision is None:
                evidence.append(f"huggingface_metadata_invalid:{metadata_file}")
                continue
            candidates.add(revision)
            evidence.append(f"huggingface_commit:{metadata_file}:{revision}")

    explicit_revision_files: List[Path] = []
    for name in ("REVISION", "revision.txt", ".proactive_revision"):
        candidate_file = path / name
        if candidate_file.exists():
            explicit_revision_files.append(candidate_file)
    for revision_file in explicit_revision_files:
        try:
            text = revision_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matches = REVISION_RE.findall(text)
        for revision in matches:
            candidates.add(revision.lower())
            evidence.append(f"explicit_revision_file:{revision_file}:{revision.lower()}")

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
