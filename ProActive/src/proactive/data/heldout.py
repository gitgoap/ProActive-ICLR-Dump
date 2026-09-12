"""Strict loaders for the frozen Week 8 held-out stress datasets.

The official PRE-HAL and IllusionBench releases use different native layouts.
This module converts both releases to the ordinary ProActive manifest contract
while retaining a complete release audit.  In particular, malformed or
image-less IllusionBench rows are never silently promoted to valid examples.

Held-out rows are sampled without consulting any model output.  The sample is
deterministic, proportionally stratified by release metadata, and grouped by
source image for downstream bootstrap resampling.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from proactive.data.manifests import make_manifest_record
from proactive.features.normalization import normalize_true_false
from proactive.utils.hashing import compute_group_id
from proactive.utils.io import file_sha256


PREHAL_COLUMNS = (
    "index",
    "question",
    "A",
    "B",
    "C",
    "D",
    "answer",
    "image",
    "hallucination_type",
)
ILLUSION_QA_FIELDS = ("Question", "Question Type", "Correct Answer")
ROMAN_TO_LETTER = {
    "i": "A",
    "ii": "B",
    "iii": "C",
    "iv": "D",
    "v": "E",
    "vi": "F",
}
_ROMAN_MARKER = re.compile(
    r"(?<![A-Za-z0-9.])(?P<roman>vi|iv|v|iii|ii|i)[.)]\s*",
    flags=re.IGNORECASE,
)
_ANSWER_MARKER = re.compile(
    r"^\s*(?P<roman>vi|iv|v|iii|ii|i)[.)]\s*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class HeldoutLoadResult:
    """Selected records and the release-level audit that produced them."""

    records: List[Dict[str, Any]]
    audit: Dict[str, Any]
    exclusions: List[Dict[str, Any]]


def _clean_release_text(value: Any) -> str:
    """Remove known decode sentinels and normalize whitespace uniformly."""

    text = str(value if value is not None else "").replace("\ufffd", " ")
    return " ".join(text.split())


def _stable_rank(seed: int, namespace: str, identity: str) -> str:
    return hashlib.sha256(
        f"{seed}|{namespace}|{identity}".encode("utf-8")
    ).hexdigest()


def _image_group_record(
    *,
    dataset: str,
    image_id: str,
    question_id: str,
    image_path: Path,
    question: str,
    gold_answer: str,
    extra: Mapping[str, Any],
) -> Dict[str, Any]:
    record = make_manifest_record(
        dataset=dataset,
        image_id=image_id,
        question_id=question_id,
        image_path=str(image_path),
        question=question,
        gold_answer=gold_answer,
        relation_applicable=False,
        extra=dict(extra),
    )
    # Multiple questions can share one held-out image.  Keeping them in one
    # group makes confidence intervals resample the independent visual unit.
    record["group_id"] = compute_group_id(dataset, image_id, "heldout_image")
    record["split"] = "shift"
    return record


def deterministic_stratified_sample(
    records: Sequence[Dict[str, Any]],
    *,
    cap: int,
    seed: int,
    stratum_field: str = "heldout_stratum",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return a proportional deterministic sample with an exact size.

    The quota is Hamilton's largest-remainder apportionment.  Ties and records
    inside each stratum are ordered by SHA-256, never by source-file order.
    """

    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("Held-out sample cap must be a positive integer")
    if not records:
        raise ValueError("Cannot sample an empty held-out release")
    if cap >= len(records):
        selected = list(records)
        return selected, dict(Counter(str(row[stratum_field]) for row in selected))

    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(stratum_field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Held-out record lacks {stratum_field}")
        strata[value].append(record)

    total = len(records)
    exact = {key: cap * len(rows) / total for key, rows in strata.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = cap - sum(quotas.values())
    remainder_order = sorted(
        strata,
        key=lambda key: (
            -(exact[key] - quotas[key]),
            _stable_rank(seed, "stratum", key),
        ),
    )
    for key in remainder_order[:remaining]:
        quotas[key] += 1

    selected: List[Dict[str, Any]] = []
    for key in sorted(strata):
        ranked = sorted(
            strata[key],
            key=lambda row: _stable_rank(seed, key, str(row["instance_id"])),
        )
        selected.extend(ranked[: quotas[key]])
    selected.sort(key=lambda row: str(row["instance_id"]))
    if len(selected) != cap or len({row["instance_id"] for row in selected}) != cap:
        raise ValueError("Deterministic held-out sampling did not produce an exact unique cap")
    return selected, dict(Counter(str(row[stratum_field]) for row in selected))


def _require_file_hash(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = file_sha256(path)
    if expected and actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def _resolve_prehal_image(root: Path, source_value: str) -> Path:
    relative = source_value.strip().replace("\\", "/").lstrip("/")
    if not relative or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe PRE-HAL image path: {source_value!r}")
    candidates = (root / "images" / relative, root / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _prehal_image_identity(source_value: str) -> Tuple[str, str]:
    """Return a collision-safe ID and canonical release-relative image key.

    PRE-HAL contains identical basenames in different source subdirectories.
    Using only ``Path(...).stem`` merges distinct images and corrupts grouped
    sampling/bootstrap units.  The complete normalized relative path is the
    identity; a short hash keeps manifest IDs filesystem- and shell-friendly.
    """

    relative = source_value.strip().replace("\\", "/").lstrip("/")
    path = Path(relative)
    if not relative or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"Unsafe PRE-HAL image path: {source_value!r}")
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
    return f"{path.stem}_{digest}", relative


def load_prehal_release(
    root: str | Path,
    config: Mapping[str, Any],
    *,
    limit: int | None = None,
) -> HeldoutLoadResult:
    """Load the pinned PRE-HAL CSV/images release."""

    root = Path(root)
    csv_path = root / str(config.get("annotation_file", "dataset.csv"))
    source_sha = _require_file_hash(
        csv_path,
        str(config.get("annotation_sha256") or "") or None,
        "PRE-HAL annotation CSV",
    )
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREHAL_COLUMNS:
            raise ValueError(
                "PRE-HAL CSV schema drift: expected "
                f"{list(PREHAL_COLUMNS)}, found {reader.fieldnames}"
            )
        source_rows = list(reader)

    expected_rows = config.get("expected_source_rows")
    if expected_rows is not None and len(source_rows) != int(expected_rows):
        raise ValueError(
            f"PRE-HAL source-row drift: expected {expected_rows}, found {len(source_rows)}"
        )

    records: List[Dict[str, Any]] = []
    missing_images: List[str] = []
    type_counts: Counter[str] = Counter()
    for row_number, row in enumerate(source_rows, start=2):
        source_index = _clean_release_text(row["index"])
        question = _clean_release_text(row["question"])
        gold = _clean_release_text(row["answer"]).upper()
        options = {
            key: _clean_release_text(row[key])
            for key in ("A", "B", "C", "D")
            if _clean_release_text(row[key])
        }
        if not source_index or not question:
            raise ValueError(f"PRE-HAL row {row_number} has an empty identity/question")
        if gold not in options:
            raise ValueError(
                f"PRE-HAL row {row_number} gold {gold!r} is absent from its options"
            )
        image_id, source_image_key = _prehal_image_identity(row["image"])
        image_path = _resolve_prehal_image(root, source_image_key)
        if not image_path.is_file():
            missing_images.append(str(row["image"]))
            continue
        hall_type = _clean_release_text(row["hallucination_type"]).lower()
        if not hall_type:
            raise ValueError(f"PRE-HAL row {row_number} has no hallucination type")
        type_counts[hall_type] += 1
        prompt = "\n".join(
            [question, *[f"{key}. {value}" for key, value in options.items()]]
        )
        records.append(
            _image_group_record(
                dataset="prehal",
                image_id=image_id,
                question_id=source_index,
                image_path=image_path,
                question=prompt,
                gold_answer=gold,
                extra={
                    "answer_type": "multiple_choice",
                    "normalizer_type": "multiple_choice",
                    "answer_choices": options,
                    "answer_match_mode": "choice_exact",
                    "benchmark_gold_answer": gold,
                    "source_question": question,
                    "source_image": str(row["image"]),
                    "source_image_key": source_image_key,
                    "hallucination_type": hall_type,
                    "category": hall_type,
                    "heldout_stratum": hall_type,
                    "heldout_release_revision": str(config["revision"]),
                },
            )
        )

    if missing_images:
        raise FileNotFoundError(
            "PRE-HAL snapshot is incomplete: "
            f"{len(missing_images)} referenced images are missing; first={missing_images[:5]}"
        )
    expected_unique_images = config.get("expected_unique_images")
    observed_unique_images = len({row["source_image_key"] for row in records})
    if expected_unique_images is not None and observed_unique_images != int(expected_unique_images):
        raise ValueError(
            "PRE-HAL unique-image inventory drift: "
            f"expected {expected_unique_images}, found {observed_unique_images}"
        )
    cap = min(int(config.get("sample_cap", 600)), len(records))
    selected, selected_strata = deterministic_stratified_sample(
        records, cap=cap, seed=int(config.get("split_seed", 42))
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = sorted(
            selected,
            key=lambda row: _stable_rank(
                int(config.get("split_seed", 42)), "limit", row["instance_id"]
            ),
        )[:limit]

    audit = {
        "dataset": "prehal",
        "format_version": "heldout_release_audit_v1",
        "official_url": str(config["official_url"]),
        "revision": str(config["revision"]),
        "license": config.get("license"),
        "source_annotation_path": str(csv_path),
        "source_annotation_sha256": source_sha,
        "source_rows": len(source_rows),
        "eligible_rows": len(records),
        "selected_rows": len(selected),
        "source_unique_images": observed_unique_images,
        "selected_unique_images": len({row["image_id"] for row in selected}),
        "normalized_hallucination_type_counts": dict(sorted(type_counts.items())),
        "selected_stratum_counts": dict(sorted(selected_strata.items())),
        "selection_rule": "proportional_strata_sha256_rank_v1",
        "selection_uses_model_outputs": False,
        "split": "shift",
    }
    return HeldoutLoadResult(selected, audit, [])


def _find_illusion_image_dir(root: Path) -> Path:
    candidates = (
        root / "images",
        root / "IllusionDataset",
        root / "IllusionDataset" / "IllusionDataset",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(
            path.is_file() and path.suffix.lower() == ".png" and not path.name.startswith("._")
            for path in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError(
        f"IllusionBench extracted images not found under {root}; run setup_heldout_datasets.py"
    )


def _option_sequence(matches: Sequence[re.Match[str]]) -> List[re.Match[str]]:
    """Choose the longest well-formed i, ii, iii, ... option sequence."""

    best: List[re.Match[str]] = []
    for start, match in enumerate(matches):
        if match.group("roman").lower() != "i":
            continue
        candidate = [match]
        expected = 1
        for later in matches[start + 1 :]:
            expected += 1
            if expected > len(ROMAN_TO_LETTER):
                break
            roman = later.group("roman").lower()
            if ROMAN_TO_LETTER.get(roman) != chr(64 + expected):
                break
            candidate.append(later)
        if len(candidate) > len(best) or (
            len(candidate) == len(best) and candidate and candidate[0].start() > best[0].start()
        ):
            best = candidate
    return best if len(best) >= 2 else []


def parse_illusion_multiple_choice(
    question: str,
    correct_answer: str,
) -> Tuple[str, Dict[str, str], str]:
    """Convert a well-formed roman-numeral IllusionBench item to A–F.

    The official annotations contain malformed and conflicting rows.  This
    parser accepts only consecutive option markers and requires the supplied
    answer text to match the option named by its marker (or uniquely match one
    option when the marker is absent).
    """

    question = _clean_release_text(question)
    answer = _clean_release_text(correct_answer)
    matches = _option_sequence(list(_ROMAN_MARKER.finditer(question)))
    if not matches:
        raise ValueError("malformed_option_sequence")
    stem = question[: matches[0].start()].strip()
    stem = re.sub(r"^\s*(?:vi|iv|v|iii|ii|i)[.)]\s*", "", stem, flags=re.I)
    if not stem:
        raise ValueError("empty_question_stem")
    options: Dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        option = question[match.end() : end].strip()
        letter = chr(65 + index)
        if not option:
            raise ValueError("empty_option")
        options[letter] = option

    answer_match = _ANSWER_MARKER.match(answer)
    if answer_match:
        letter = ROMAN_TO_LETTER[answer_match.group("roman").lower()]
        answer_text = answer[answer_match.end() :].strip()
        if letter not in options:
            raise ValueError("answer_marker_outside_options")
        if answer_text and _comparison_text(answer_text) != _comparison_text(options[letter]):
            raise ValueError("answer_text_option_mismatch")
    else:
        candidates = [
            letter
            for letter, option in options.items()
            if _comparison_text(answer) == _comparison_text(option)
        ]
        if len(candidates) != 1:
            raise ValueError("answer_has_no_unique_option")
        letter = candidates[0]
    prompt = "\n".join([stem, *[f"{key}. {value}" for key, value in options.items()]])
    return prompt, options, letter


def _comparison_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def load_illusionbench_release(
    root: str | Path,
    config: Mapping[str, Any],
    *,
    limit: int | None = None,
) -> HeldoutLoadResult:
    """Load the pinned IllusionBench JSON/extracted-image release."""

    root = Path(root)
    annotation_path = root / str(config.get("annotation_file", "Image_properties.json"))
    annotation_sha = _require_file_hash(
        annotation_path,
        str(config.get("annotation_sha256") or "") or None,
        "IllusionBench annotation JSON",
    )
    with open(annotation_path, "r", encoding="utf-8") as handle:
        source = json.load(handle)
    if not isinstance(source, list):
        raise ValueError("IllusionBench annotation root must be a list")
    expected_images = config.get("expected_annotation_images")
    if expected_images is not None and len(source) != int(expected_images):
        raise ValueError(
            "IllusionBench annotation-image count drift: "
            f"expected {expected_images}, found {len(source)}"
        )

    image_dir = _find_illusion_image_dir(root)
    archive_images = {
        path.name: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png" and not path.name.startswith("._")
    }
    expected_archive = config.get("expected_archive_images")
    if expected_archive is not None and len(archive_images) != int(expected_archive):
        raise ValueError(
            "IllusionBench archive-image inventory drift: "
            f"expected {expected_archive}, found {len(archive_images)}"
        )
    annotation_names: set[str] = set()
    exclusions: List[Dict[str, Any]] = []
    eligible: List[Dict[str, Any]] = []
    source_qa_count = 0
    eligible_type_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()

    for image_index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise ValueError(f"IllusionBench image record {image_index} is not an object")
        prop = item.get("image_property")
        qa_data = item.get("qa_data")
        if not isinstance(prop, Mapping) or not isinstance(qa_data, list):
            raise ValueError(f"IllusionBench image record {image_index} schema mismatch")
        image_name = _clean_release_text(prop.get("image_name"))
        if not image_name or Path(image_name).name != image_name:
            raise ValueError(f"Unsafe IllusionBench image name: {image_name!r}")
        if image_name in annotation_names:
            raise ValueError(f"Duplicate IllusionBench image annotation: {image_name}")
        annotation_names.add(image_name)
        image_path = archive_images.get(image_name)
        category = _clean_release_text(prop.get("Category"))
        difficulty = prop.get("Difficult Level")
        description = _clean_release_text(prop.get("Description"))
        category_counts[category] += 1
        difficulty_counts[str(difficulty)] += 1
        for qa_index, qa in enumerate(qa_data):
            source_qa_count += 1
            identity = f"{image_name}#{qa_index}"
            if image_path is None:
                exclusions.append(
                    {"source_id": identity, "image_name": image_name, "reason": "missing_image"}
                )
                continue
            if not isinstance(qa, Mapping) or any(field not in qa for field in ILLUSION_QA_FIELDS):
                exclusions.append(
                    {"source_id": identity, "image_name": image_name, "reason": "qa_schema_mismatch"}
                )
                continue
            raw_question = _clean_release_text(qa["Question"])
            raw_type = _clean_release_text(qa["Question Type"]).lower()
            raw_answer = _clean_release_text(qa["Correct Answer"])
            if not raw_question or not raw_answer:
                exclusions.append(
                    {"source_id": identity, "image_name": image_name, "reason": "empty_question_or_answer"}
                )
                continue
            try:
                if raw_type == "tf":
                    gold = normalize_true_false(raw_answer)
                    if gold == "unknown":
                        raise ValueError("invalid_true_false_gold")
                    question = raw_question
                    answer_type = "binary"
                    normalizer_type = "true_false"
                    choices = None
                    match_mode = "binary_exact"
                elif raw_type == "select":
                    question, choices, gold = parse_illusion_multiple_choice(
                        raw_question, raw_answer
                    )
                    answer_type = "multiple_choice"
                    normalizer_type = "multiple_choice"
                    match_mode = "choice_exact"
                else:
                    raise ValueError("unsupported_question_type")
            except ValueError as exc:
                exclusions.append(
                    {
                        "source_id": identity,
                        "image_name": image_name,
                        "reason": str(exc),
                        "question_type": raw_type,
                    }
                )
                continue
            stratum = f"category_{category}|{raw_type}|difficulty_{difficulty}"
            extra: Dict[str, Any] = {
                "answer_type": answer_type,
                "normalizer_type": normalizer_type,
                "answer_match_mode": match_mode,
                "benchmark_gold_answer": raw_answer,
                "source_question": raw_question,
                "source_question_type": raw_type,
                "source_category": category,
                "category": f"category_{category}",
                "difficulty_level": difficulty,
                "image_description": description,
                "heldout_stratum": stratum,
                "heldout_release_revision": str(config["revision"]),
            }
            if choices is not None:
                extra["answer_choices"] = choices
            eligible.append(
                _image_group_record(
                    dataset="illusionbench",
                    image_id=Path(image_name).stem,
                    question_id=str(qa_index),
                    image_path=image_path,
                    question=question,
                    gold_answer=gold,
                    extra=extra,
                )
            )
            eligible_type_counts[raw_type] += 1

    missing_names = sorted(annotation_names - set(archive_images))
    extra_names = sorted(set(archive_images) - annotation_names)
    expected_missing = config.get("expected_missing_annotation_images")
    expected_extra = config.get("expected_unreferenced_archive_images")
    if expected_missing is not None and len(missing_names) != int(expected_missing):
        raise ValueError(
            "IllusionBench missing-image inventory drift: "
            f"expected {expected_missing}, found {len(missing_names)}"
        )
    if expected_extra is not None and len(extra_names) != int(expected_extra):
        raise ValueError(
            "IllusionBench extra-image inventory drift: "
            f"expected {expected_extra}, found {len(extra_names)}"
        )
    expected_qa = config.get("expected_source_qa_rows")
    if expected_qa is not None and source_qa_count != int(expected_qa):
        raise ValueError(
            f"IllusionBench QA-count drift: expected {expected_qa}, found {source_qa_count}"
        )
    if len(eligible) < int(config.get("sample_cap", 600)):
        raise ValueError(
            f"IllusionBench has only {len(eligible)} eligible rows, below the requested cap"
        )
    selected, selected_strata = deterministic_stratified_sample(
        eligible,
        cap=int(config.get("sample_cap", 600)),
        seed=int(config.get("split_seed", 42)),
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = sorted(
            selected,
            key=lambda row: _stable_rank(
                int(config.get("split_seed", 42)), "limit", row["instance_id"]
            ),
        )[:limit]

    exclusion_counts = Counter(str(row["reason"]) for row in exclusions)
    audit = {
        "dataset": "illusionbench",
        "format_version": "heldout_release_audit_v1",
        "official_url": str(config["official_url"]),
        "revision": str(config["revision"]),
        "license": config.get("license"),
        "license_status": config.get("license_status"),
        "source_annotation_path": str(annotation_path),
        "source_annotation_sha256": annotation_sha,
        "extracted_image_dir": str(image_dir),
        "annotation_image_count": len(annotation_names),
        "archive_image_count": len(archive_images),
        "missing_annotation_image_count": len(missing_names),
        "missing_annotation_images": missing_names,
        "unreferenced_archive_image_count": len(extra_names),
        "unreferenced_archive_images": extra_names,
        "source_qa_rows": source_qa_count,
        "eligible_rows": len(eligible),
        "excluded_rows": len(exclusions),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "selected_rows": len(selected),
        "eligible_question_type_counts": dict(sorted(eligible_type_counts.items())),
        "annotation_category_counts": dict(sorted(category_counts.items())),
        "annotation_difficulty_counts": dict(sorted(difficulty_counts.items())),
        "selected_stratum_counts": dict(sorted(selected_strata.items())),
        "selection_rule": "proportional_strata_sha256_rank_v1",
        "selection_uses_model_outputs": False,
        "filter_rule": "image_present_and_strict_supported_qa_contract_v1",
        "split": "shift",
    }
    return HeldoutLoadResult(selected, audit, exclusions)
