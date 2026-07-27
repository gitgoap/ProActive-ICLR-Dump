"""
IO utilities: atomic JSONL writes, resumable reads, and file hashing.

All teacher cache and manifest files use JSONL format. Writes are atomic
(write to temp file, then rename) to prevent partial-write corruption
on interruption.  (Plan §27.5)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Lazily iterate records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(
    records: List[Dict[str, Any]],
    path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write records to a JSONL file.

    Writes to a temp file in the same directory, then renames.
    This prevents partial files if the process is interrupted.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}. "
            f"Use overwrite=True or --overwrite flag."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in the same directory, then atomic rename
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem + "_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Atomic rename (same filesystem)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return path


def append_jsonl(
    record: Dict[str, Any],
    path: str | Path,
) -> None:
    """Append a single record to a JSONL file (for resumable generation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_completed_ids(
    path: str | Path,
    id_field: str = "instance_id",
) -> Set[str]:
    """Read completed IDs from an output file for resume support.

    Returns an empty set if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return set()
    ids = set()
    for record in iter_jsonl(path):
        if id_field in record:
            ids.add(record[id_field])
    return ids


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
