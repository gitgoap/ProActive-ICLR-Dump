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
from proactive.probes.relation_swap import swap_relation



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
    """Resolve the data path from config, env var, or local search paths."""
    dataset_name = config.get("dataset_name", "")

    # Candidate dataset directory names (e.g. POPE, pope, Pope)
    name_variants = list({
        dataset_name,
        dataset_name.upper(),
        dataset_name.lower(),
        dataset_name.capitalize(),
        "POPE" if "pope" in dataset_name.lower() else "",
        "VSR" if "vsr" in dataset_name.lower() else "",
        "VizWiz" if "vizwiz" in dataset_name.lower() else "",
        "HallusionBench" if "hallusion" in dataset_name.lower() else "",
    } - {""})

    # Candidate root directories
    search_roots: List[Path] = []

    env_root = os.environ.get("PROACTIVE_DATA_ROOT")
    if env_root:
        search_roots.append(Path(env_root))

    search_roots.extend([
        Path.cwd() / "data",
        Path.cwd().parent / "data",
        Path.home() / "ProActive" / "data",
        Path.home() / "MMUQ" / "data",
        Path("/home/aman/ProActive/data"),
        Path("/home/aman/MMUQ/data"),
        Path("data"),
    ])

    # Direct config checks first
    data_path = config.get("data_path", "")
    if data_path:
        expanded = os.path.expandvars(data_path)
        if not expanded.startswith("${") and Path(expanded).exists():
            return Path(expanded)

    default = config.get("default_server_path", "")
    if default and Path(default).exists():
        return Path(default)

    # Search through candidate roots and name variants
    for root in search_roots:
        if not root.exists():
            continue
        # Direct check if root itself is the dataset folder
        if root.name.lower() in [v.lower() for v in name_variants]:
            return root
        for v in name_variants:
            cand = root / v
            if cand.exists():
                return cand

    raise FileNotFoundError(
        f"Cannot find data for {config.get('dataset_name', '?')}. "
        f"Searched roots: {[str(r) for r in search_roots if r.exists()]}. "
        f"Please export PROACTIVE_DATA_ROOT or pass --data_root."
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
        "annotation_file", "HallusionBench.json"
    )
    ann_path = data_root / annotation_file

    if not ann_path.exists():
        # Try alternative paths
        for alt in ["HallusionBench.json", "hallusion_bench/HallusionBench.json", "hallusion_bench.json"]:
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

        raw_name = item.get("filename", item.get("image", ""))
        # HallusionBench contains 178 text-only examples in addition to the
        # 951 image-paired examples used by ProActive/HalluPrism.  Filtering
        # must happen before ``limit``/``sample_cap`` so the pilot cannot
        # silently spend part of its quota on records with no image.
        if not raw_name:
            continue
        image_path = ""
        candidates = [
            data_root / raw_name,
            data_root / "hallusion_bench" / raw_name,
            data_root / "hallusion_bench" / "VD" / "illusion" / Path(raw_name).name,
            data_root / "hallusion_bench" / "VS" / Path(raw_name).name,
        ]
        for cand in candidates:
            if cand.exists():
                image_path = str(cand)
                break
        if not image_path:
            image_path = str(data_root / raw_name)

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

    # Resolve COCO image directory
    coco_image_dir = None
    if "coco_image_dir" in config:
        coco_cfg = Path(os.path.expandvars(config["coco_image_dir"]))
        if coco_cfg.exists():
            coco_image_dir = coco_cfg

    if coco_image_dir is None:
        env_root = os.environ.get("PROACTIVE_DATA_ROOT")
        candidates = [
            data_root / "images",
            data_root / "val2014",
            data_root / "coco" / "val2014",
            data_root.parent / "coco" / "val2014",
            data_root.parent / "COCO" / "val2014",
            data_root.parent / "val2014",
        ]
        if env_root:
            candidates.extend([
                Path(env_root) / "POPE" / "images",
                Path(env_root) / "coco" / "val2014",
                Path(env_root) / "COCO" / "val2014",
                Path(env_root) / "val2014",
            ])
        for cand in candidates:
            if cand.exists():
                coco_image_dir = cand
                break

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

            img_name = item.get("image", "")
            if coco_image_dir and (coco_image_dir / img_name).exists():
                image_path = str(coco_image_dir / img_name)
            elif (data_root / "images" / img_name).exists():
                image_path = str(data_root / "images" / img_name)
            elif (data_root / "val2014" / img_name).exists():
                image_path = str(data_root / "val2014" / img_name)
            elif (data_root / img_name).exists():
                image_path = str(data_root / img_name)
            elif coco_image_dir:
                image_path = str(coco_image_dir / img_name)
            else:
                image_path = str(data_root / "images" / img_name)

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
        for alt in [
            "Annotations/val.json",
            "annotations/val.json",
            "Annotations/train.json",
            "val.json",
        ]:
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

    # Locate image directory (supports Images/val, images/val, val, Images/train)
    image_dir = None
    for cand_dir in [
        data_root / "Images" / "val",
        data_root / "images" / "val",
        data_root / "val",
        data_root / "Images" / "train",
        data_root / "images" / "train",
        data_root / "train",
        data_root / "Images",
        data_root / "images",
    ]:
        if cand_dir.exists():
            image_dir = cand_dir
            break
    if image_dir is None:
        image_dir = data_root / config.get("image_dir", "val")

    records = []
    for i, item in enumerate(data):
        if limit and len(records) >= limit:
            break

        image_name = item.get("image", "")
        if (image_dir / image_name).exists():
            image_path = str(image_dir / image_name)
        elif (data_root / "Images" / "val" / image_name).exists():
            image_path = str(data_root / "Images" / "val" / image_name)
        elif (data_root / "Images" / "train" / image_name).exists():
            image_path = str(data_root / "Images" / "train" / image_name)
        elif (data_root / "val" / image_name).exists():
            image_path = str(data_root / "val" / image_name)
        else:
            image_path = str(image_dir / image_name)

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
        for alt in ["dev.jsonl", "train.jsonl", "all_vsr_validated_data.jsonl"]:
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

            raw_img = item.get("image", item.get("image_link", ""))
            img_name = Path(raw_img).name
            if (data_root / raw_img).exists():
                image_path = str(data_root / raw_img)
            elif (data_root / "val2017" / img_name).exists():
                image_path = str(data_root / "val2017" / img_name)
            elif (data_root / "train2017" / img_name).exists():
                image_path = str(data_root / "train2017" / img_name)
            elif (data_root / "images" / img_name).exists():
                image_path = str(data_root / "images" / img_name)
            else:
                image_path = str(data_root / raw_img)

            caption = item.get("caption", "")
            raw_label = item.get("label", "")
            label_str = "true" if raw_label == 1 or str(raw_label).lower() == "true" else "false"

            swap = swap_relation(
                caption,
                annotated_relation=item.get("relation"),
                gold_answer=label_str,
            )

            extra: Dict[str, Any] = {
                "relation": item.get("relation", ""),
                "subrelation": item.get("subrelation", ""),
            }
            if swap.applicable:
                extra["swapped_question"] = swap.swapped_text
                extra["swapped_gold_answer"] = swap.swapped_gold_answer
                extra["relation_found"] = swap.relation_found

            record = make_manifest_record(
                dataset="vsr",
                image_id=str(item.get("image_id", i)),
                question_id=str(i),
                image_path=image_path,
                question=caption,
                gold_answer=label_str,
                relation_applicable=swap.applicable,
                extra=extra,
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
