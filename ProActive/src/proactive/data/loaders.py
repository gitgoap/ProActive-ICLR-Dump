"""
Dataset-specific loaders.

Each loader reads a dataset from its native format and produces
standardized manifest records. Loaders are registered by name
and dispatched from the config's 'loader' field.

All loaders return List[Dict] with fields matching the manifest schema.
Actual data paths come from config YAML or environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from proactive.data.manifests import make_manifest_record


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

_LOADERS: Dict[str, Callable] = {}


def register_loader(name: str):
    """Decorator to register a dataset loader function."""
    def decorator(fn: Callable):
        _LOADERS[name] = fn
        return fn
    return decorator


def get_loader(name: str) -> Callable:
    """Get a registered loader by name."""
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown loader '{name}'. Available: {sorted(_LOADERS.keys())}"
        )
    return _LOADERS[name]


def resolve_data_path(config: Dict[str, Any]) -> Path:
    """Resolve the data path from config, env var, or default server path."""
    # Try environment variable first
    env_root = os.environ.get("PROACTIVE_DATA_ROOT")
    if env_root:
        dataset_name = config["dataset_name"]
        candidate = Path(env_root) / dataset_name
        if candidate.exists():
            return candidate

    # Try config data_path (with env var expansion)
    data_path = config.get("data_path", "")
    if data_path:
        expanded = os.path.expandvars(data_path)
        if Path(expanded).exists():
            return Path(expanded)

    # Fall back to default server path
    default = config.get("default_server_path", "")
    if default and Path(default).exists():
        return Path(default)

    raise FileNotFoundError(
        f"Cannot find data for {config.get('dataset_name', '?')}. "
        f"Set PROACTIVE_DATA_ROOT env var or update the config."
    )


# ---------------------------------------------------------------------------
# HallusionBench loader
# ---------------------------------------------------------------------------

@register_loader("hallusionbench")
def load_hallusionbench(
    config: Dict[str, Any],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load HallusionBench dataset."""
    data_root = resolve_data_path(config)
    annotation_file = config.get(
        "annotation_file", "hallusion_bench/HallusionBench.json"
    )
    ann_path = data_root / annotation_file

    if not ann_path.exists():
        # Try alternative paths
        for alt in ["HallusionBench.json", "hallusion_bench.json"]:
            alt_path = data_root / alt
            if alt_path.exists():
                ann_path = alt_path
                break

    if not ann_path.exists():
        raise FileNotFoundError(
            f"HallusionBench annotations not found at {ann_path}"
        )

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for i, item in enumerate(data):
        if limit and len(records) >= limit:
            break

        image_path = ""
        if "filename" in item:
            image_path = str(data_root / item["filename"])
        elif "image" in item:
            image_path = str(data_root / item["image"])

        question = item.get("question", "")
        gold = item.get("gt_answer", item.get("answer", ""))

        record = make_manifest_record(
            dataset="hallusionbench",
            image_id=str(item.get("id", i)),
            question_id=str(i),
            image_path=image_path,
            question=question,
            gold_answer=str(gold),
            relation_applicable=False,
            extra={
                "category": item.get("category", ""),
                "subcategory": item.get("subcategory", ""),
            },
        )
        records.append(record)

    cap = config.get("sample_cap")
    if cap and len(records) > cap:
        records = records[:cap]

    return records


# ---------------------------------------------------------------------------
# POPE loader
# ---------------------------------------------------------------------------

@register_loader("pope")
def load_pope(
    config: Dict[str, Any],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load POPE dataset (random/popular/adversarial)."""
    data_root = resolve_data_path(config)

    records = []
    # POPE typically has 3 JSONL files
    for split_type in ["random", "popular", "adversarial"]:
        for pattern in [
            f"coco_pope_{split_type}.json",
            f"pope_{split_type}.json",
            f"*{split_type}*.json",
            f"*{split_type}*.jsonl",
        ]:
            candidates = list(data_root.glob(pattern))
            if candidates:
                ann_path = candidates[0]
                break
        else:
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Try JSONL format
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]

        for i, item in enumerate(data):
            if limit and len(records) >= limit:
                break

            image_path = item.get("image", "")
            question = item.get("text", item.get("question", ""))
            gold = item.get("label", item.get("answer", ""))

            record = make_manifest_record(
                dataset="pope",
                image_id=str(item.get("image_id", f"{split_type}_{i}")),
                question_id=f"{split_type}_{i}",
                image_path=image_path,
                question=question,
                gold_answer=str(gold),
                relation_applicable=False,
                extra={"pope_split": split_type},
            )
            records.append(record)

    cap = config.get("sample_cap")
    if cap and len(records) > cap:
        records = records[:cap]

    return records


# ---------------------------------------------------------------------------
# VizWiz loader
# ---------------------------------------------------------------------------

@register_loader("vizwiz")
def load_vizwiz(
    config: Dict[str, Any],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load VizWiz-VQA dataset."""
    data_root = resolve_data_path(config)
    annotation_file = config.get("annotation_file", "val.json")
    ann_path = data_root / annotation_file

    # Try alternative locations
    if not ann_path.exists():
        for alt in ["Annotations/val.json", "annotations/val.json"]:
            alt_path = data_root / alt
            if alt_path.exists():
                ann_path = alt_path
                break

    if not ann_path.exists():
        raise FileNotFoundError(
            f"VizWiz annotations not found at {ann_path}"
        )

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_dir = config.get("image_dir", "val")

    records = []
    for i, item in enumerate(data):
        if limit and len(records) >= limit:
            break

        image_name = item.get("image", "")
        image_path = str(data_root / image_dir / image_name)
        question = item.get("question", "")

        # VizWiz has multiple annotator answers
        answers = item.get("answers", [])
        if answers:
            # Use majority answer
            answer_texts = [a.get("answer", "") for a in answers]
            gold = max(set(answer_texts), key=answer_texts.count)
        else:
            gold = item.get("answer", "unanswerable")

        record = make_manifest_record(
            dataset="vizwiz",
            image_id=image_name.replace(".jpg", ""),
            question_id=str(i),
            image_path=image_path,
            question=question,
            gold_answer=gold,
            relation_applicable=False,
            extra={
                "answerable": item.get("answerable", None),
            },
        )
        records.append(record)

    cap = config.get("sample_cap")
    if cap and len(records) > cap:
        records = records[:cap]

    return records


# ---------------------------------------------------------------------------
# VSR loader
# ---------------------------------------------------------------------------

@register_loader("vsr")
def load_vsr(
    config: Dict[str, Any],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load VSR (Visual Spatial Reasoning) dataset."""
    data_root = resolve_data_path(config)
    annotation_file = config.get("annotation_file", "test.jsonl")
    ann_path = data_root / annotation_file

    if not ann_path.exists():
        # Try alternatives
        for alt in ["dev.jsonl", "all_vsr_validated_data.jsonl"]:
            alt_path = data_root / alt
            if alt_path.exists():
                ann_path = alt_path
                break

    if not ann_path.exists():
        raise FileNotFoundError(
            f"VSR annotations not found at {ann_path}"
        )

    records = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            image_path = item.get("image", item.get("image_link", ""))
            caption = item.get("caption", "")
            label = item.get("label", "")

            record = make_manifest_record(
                dataset="vsr",
                image_id=str(item.get("image_id", i)),
                question_id=str(i),
                image_path=image_path,
                question=caption,
                gold_answer=str(label).lower(),
                relation_applicable=True,
                extra={
                    "relation": item.get("relation", ""),
                    "subrelation": item.get("subrelation", ""),
                },
            )
            records.append(record)

    cap = config.get("sample_cap")
    if cap and len(records) > cap:
        records = records[:cap]

    return records


# ---------------------------------------------------------------------------
# GQA relation slice loader (placeholder)
# ---------------------------------------------------------------------------

@register_loader("gqa_relation")
def load_gqa_relation(
    config: Dict[str, Any],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load GQA relation slice.

    This dataset must first be CONSTRUCTED from GQA scene graphs
    using scripts/build_gqa_relation.py (Plan §13.4).
    """
    data_root = resolve_data_path(config)

    # Look for pre-built relation slice
    built_path = data_root / "gqa_relation_slice.jsonl"
    if not built_path.exists():
        raise FileNotFoundError(
            f"GQA relation slice not found at {built_path}. "
            f"Run scripts/build_gqa_relation.py first."
        )

    records = []
    with open(built_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(item)

    cap = config.get("sample_cap")
    if cap and len(records) > cap:
        records = records[:cap]

    return records


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------

def load_dataset_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a dataset YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_from_config(
    config_path: str | Path,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load a dataset using its YAML config."""
    config = load_dataset_config(config_path)
    loader_name = config["loader"]
    loader_fn = get_loader(loader_name)
    return loader_fn(config, limit=limit)
