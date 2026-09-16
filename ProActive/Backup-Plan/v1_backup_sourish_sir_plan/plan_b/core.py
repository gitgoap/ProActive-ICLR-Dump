"""Fail-closed data and decision contracts for ProActive Plan B.

The module intentionally consumes only cached normalized answers.  It never
loads an image or calls a multimodal model.  Gold answers are available only
to audit/training/evaluation helpers; candidate feature construction accepts
no gold field.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence


ABSTAIN = "__ABSTAIN__"
SOURCE_ORDER = ("clean", "grounding", "blur", "crop", "brightness", "noise", "blank")
PROBE_ORDER = SOURCE_ORDER[1:]
FEATURE_NAMES = tuple(f"source_{name}" for name in SOURCE_ORDER) + (
    "vote_share",
    "distinct_answer_fraction",
    "acquired_probe_fraction",
)
UNUSABLE_NORMALIZED = frozenset({"", "unknown", "invalid", "uncertain"})


class PlanBError(RuntimeError):
    """Raised when a Plan B scientific or provenance contract fails."""


@dataclass(frozen=True)
class Observation:
    source: str
    norm_answer: str
    raw_answer: str
    usable: bool
    attempted: bool


@dataclass(frozen=True)
class Candidate:
    norm_answer: str
    sources: tuple[str, ...]
    raw_answer: str
    vote_count: int
    earliest_rank: int


@dataclass(frozen=True)
class Selection:
    source: str
    norm_answer: str
    raw_answer: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def with_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("report_sha256", None)
    result["report_sha256"] = canonical_json_sha256(result)
    return result


def verify_self_hash(payload: Mapping[str, Any]) -> None:
    expected = payload.get("report_sha256")
    if not isinstance(expected, str):
        raise PlanBError("Signed JSON is missing report_sha256")
    unsigned = dict(payload)
    unsigned.pop("report_sha256", None)
    actual = canonical_json_sha256(unsigned)
    if actual != expected:
        raise PlanBError(f"Signed JSON self-hash mismatch: expected={expected} actual={actual}")


def _atomic_path(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temporary = _atomic_path(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    fd, temporary = _atomic_path(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise PlanBError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise PlanBError(f"Blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanBError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise PlanBError(f"Non-object JSONL row at {path}:{line_number}")
            yield row


def normalize_gold(raw: Any, normalizer_type: str) -> str:
    text = "" if raw is None else str(raw).strip()
    cleaned = text.lower().rstrip(".")
    first = cleaned.split()[0] if cleaned.split() else ""
    if normalizer_type == "yes_no":
        yes = {"yes", "yeah", "yep", "yup", "y", "correct", "right", "true", "affirmative", "indeed", "absolutely", "sure", "positive"}
        no = {"no", "nope", "nah", "n", "incorrect", "wrong", "false", "negative", "not"}
        if first in yes or cleaned in yes or cleaned.startswith("yes"):
            return "yes"
        if first in no or cleaned in no or cleaned.startswith("no"):
            return "no"
        return "unknown"
    if normalizer_type == "hallusion_binary":
        if cleaned == "0":
            return "no"
        if cleaned == "1":
            return "yes"
        if cleaned in {"2", "uncertain", "unsure", "unknown", "cannot determine", "can't determine", "cannot be determined", "indeterminate"}:
            return "uncertain"
        return normalize_gold(text, "yes_no")
    if normalizer_type == "true_false":
        if first in {"true", "correct", "yes", "right"}:
            return "true"
        if first in {"false", "incorrect", "no", "wrong"}:
            return "false"
        return "unknown"
    if normalizer_type == "multiple_choice":
        stripped = text.replace("**", "").replace("`", "")
        patterns = (
            r"^\s*([A-Fa-f])\s*$",
            r"^\s*([A-Fa-f])\s*[.):,-]",
            r"^\s*\(([A-Fa-f])\)\s*[.:-]?\s*$",
            r"^\s*(?:(?:final\s+)?answer\s*(?:is|:)\s*)([A-Fa-f])(?:\s|[.):,-]|$)",
            r"^\s*(?:option|choice)\s+([A-Fa-f])(?:\s|[.):,-]|$)",
        )
        for pattern in patterns:
            match = re.match(pattern, stripped, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return "unknown"
    if normalizer_type == "freeform":
        unanswerable = {
            "unanswerable", "not answerable", "cannot be answered", "can't be answered",
            "cannot answer", "not sure", "i don't know", "i do not know", "n/a", "na",
            "unsuitable", "unsuitable image", "no answer", "there is no answer",
            "cannot be determined", "can't be determined", "indeterminate",
        }
        if cleaned in unanswerable:
            return "unanswerable"
        words = re.sub(r"[^\w\s]", " ", cleaned).split()
        result = " ".join(word for word in words if word not in {"a", "an", "the"})
        result = re.sub(r"\s+", " ", result).strip()
        return "unanswerable" if not result or result in unanswerable else result
    raise PlanBError(f"Unsupported normalizer_type={normalizer_type!r}")


def record_normalizer_type(row: Mapping[str, Any]) -> str:
    """Resolve the answer contract used by the historical teacher caches.

    Week 8 caches persist ``normalizer_type`` explicitly.  The older Week 4
    caches predate that field, so their contract is reconstructed only from a
    closed, dataset-specific mapping that is also enforced by the original
    manifests.  Unknown datasets fail closed.
    """

    normalizer = row.get("normalizer_type")
    if isinstance(normalizer, str) and normalizer:
        return normalizer
    dataset = str(row.get("dataset", "")).lower()
    if dataset == "pope":
        return "yes_no"
    if dataset == "vsr":
        return "true_false"
    if dataset == "vizwiz":
        return "freeform"
    if dataset == "hallusionbench":
        return "freeform" if row.get("answer_type") == "open_ended" else "yes_no"
    raise PlanBError(
        f"Missing normalizer_type for unsupported historical dataset "
        f"{dataset!r} ({row.get('instance_id')})"
    )


def record_gold_norm(row: Mapping[str, Any]) -> str:
    return normalize_gold(row.get("gold_answer"), record_normalizer_type(row))


def recompute_clean_correctness(row: Mapping[str, Any]) -> int:
    """Reproduce the cache's declared clean-answer scoring contract.

    The only non-single-gold case in the source caches is open-ended
    HallusionBench, whose frozen contract is exact normalized equality against
    a non-empty annotation alias set. Semantic fallback is forbidden there.
    """

    clean = row.get("clean")
    if not isinstance(clean, Mapping):
        raise PlanBError(f"Missing clean payload for {row.get('instance_id')}")
    normalizer = record_normalizer_type(row)
    recomputed_norm = normalize_gold(clean.get("raw_answer"), normalizer)
    saved_norm = clean.get("norm_answer")
    if recomputed_norm != saved_norm:
        raise PlanBError(
            f"Clean normalization drift for {row.get('instance_id')}: "
            f"saved={saved_norm!r} recomputed={recomputed_norm!r}"
        )
    if row.get("answer_type") == "open_ended":
        if row.get("answer_match_mode") != "exact_alias":
            raise PlanBError(
                f"Unsupported open-answer match mode for {row.get('instance_id')}: "
                f"{row.get('answer_match_mode')!r}"
            )
        references = row.get("reference_answers")
        if (
            not isinstance(references, Sequence)
            or isinstance(references, (str, bytes))
            or not references
            or any(not isinstance(reference, str) or not reference.strip() for reference in references)
        ):
            raise PlanBError(f"Invalid reference_answers for {row.get('instance_id')}")
        normalized_references = {normalize_gold(reference, normalizer) for reference in references}
        return int(recomputed_norm in normalized_references)
    return int(recomputed_norm == record_gold_norm(row))


def is_usable_normalized(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in UNUSABLE_NORMALIZED


def image_key(row: Mapping[str, Any]) -> str:
    dataset = str(row.get("dataset", "")).lower()
    raw_path = str(row.get("image_path", "")).replace("\\", "/")
    name = PurePosixPath(raw_path).name
    match = re.search(r"(?:COCO_(?:train|val)\d+_)?(\d{12})\.(?:jpg|png|jpeg)$", name, re.IGNORECASE)
    if match and (dataset in {"pope", "vsr"} or "coco" in raw_path.lower()):
        return f"coco:{match.group(1)}"
    if dataset == "hallusionbench":
        marker = "hallusion_bench/"
        lower = raw_path.lower()
        suffix = raw_path[lower.index(marker) + len(marker):] if marker in lower else name
        return f"hallusionbench:{suffix}"
    if dataset == "prehal":
        marker = "/images/"
        lower = raw_path.lower()
        suffix = raw_path[lower.index(marker) + len(marker):] if marker in lower else name
        return f"prehal:{suffix}"
    return f"{dataset}:{name}"


def is_closed_answer(row: Mapping[str, Any]) -> bool:
    return str(row.get("dataset", "")).lower() != "vizwiz" and row.get("answer_type") != "open_ended"


def observation_from_row(row: Mapping[str, Any], source: str) -> Observation:
    if source == "clean":
        payload = row.get("clean")
        if not isinstance(payload, Mapping):
            raise PlanBError(f"Missing clean payload for {row.get('instance_id')}")
        attempted = True
        valid = payload.get("valid", True) is True
        applicable = True
    else:
        probes = row.get("probes")
        if not isinstance(probes, Mapping):
            raise PlanBError(f"Missing probes for {row.get('instance_id')}")
        payload = probes.get(source)
        if payload is None:
            return Observation(source, "", "", False, False)
        if not isinstance(payload, Mapping):
            raise PlanBError(f"Malformed {source} payload for {row.get('instance_id')}")
        applicable = payload.get("applicable", False) is True
        attempted = applicable
        valid = payload.get("valid", False) is True
    norm = payload.get("norm_answer")
    raw = payload.get("raw_answer")
    norm_text = norm.strip() if isinstance(norm, str) else ""
    raw_text = raw if isinstance(raw, str) else ""
    usable = applicable and valid and is_usable_normalized(norm_text)
    return Observation(source, norm_text, raw_text, usable, attempted)


def acquired_observations(row: Mapping[str, Any], budget: int, acquisition_order: Sequence[str]) -> tuple[list[Observation], int]:
    if budget < 0 or budget > len(acquisition_order):
        raise PlanBError(f"Invalid budget {budget}")
    observations = [observation_from_row(row, "clean")]
    cost = 0
    for source in acquisition_order[:budget]:
        observation = observation_from_row(row, source)
        observations.append(observation)
        if observation.attempted:
            cost += 1
    return observations, cost


def build_candidates(observations: Sequence[Observation], acquired_cost: int) -> list[Candidate]:
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.usable:
            grouped[observation.norm_answer].append(observation)
    candidates: list[Candidate] = []
    for norm_answer, matches in grouped.items():
        sources = tuple(obs.source for obs in matches)
        earliest = min(SOURCE_ORDER.index(source) for source in sources)
        provenance = min(matches, key=lambda obs: SOURCE_ORDER.index(obs.source))
        candidates.append(Candidate(norm_answer, sources, provenance.raw_answer, len(matches), earliest))
    candidates.sort(key=lambda candidate: (candidate.earliest_rank, candidate.norm_answer))
    return candidates


def candidate_features(candidate: Candidate, candidates: Sequence[Candidate], acquired_cost: int) -> tuple[float, ...]:
    usable_votes = sum(item.vote_count for item in candidates)
    if usable_votes <= 0:
        raise PlanBError("Cannot construct candidate features without usable observations")
    values = [float(source in candidate.sources) for source in SOURCE_ORDER]
    values.extend(
        [
            candidate.vote_count / usable_votes,
            len(candidates) / 7.0,
            acquired_cost / 6.0,
        ]
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise PlanBError("Invalid candidate feature vector")
    return tuple(values)


def select_keep(observations: Sequence[Observation]) -> Selection:
    clean = observations[0]
    return Selection("clean" if clean.usable else "abstain", clean.norm_answer if clean.usable else ABSTAIN, clean.raw_answer if clean.usable else "")


def select_ground(observations: Sequence[Observation]) -> Selection:
    by_source = {observation.source: observation for observation in observations}
    ground = by_source.get("grounding")
    if ground is not None and ground.usable:
        return Selection("grounding", ground.norm_answer, ground.raw_answer)
    return select_keep(observations)


def select_majority(observations: Sequence[Observation]) -> Selection:
    usable = [observation for observation in observations if observation.usable]
    if not usable:
        return Selection("abstain", ABSTAIN, "")
    counts = Counter(observation.norm_answer for observation in usable)
    maximum = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == maximum}
    clean = observations[0]
    if clean.usable and clean.norm_answer in tied:
        winner = clean
    else:
        winner = min(
            (observation for observation in usable if observation.norm_answer in tied),
            key=lambda observation: SOURCE_ORDER.index(observation.source),
        )
    return Selection(winner.source, winner.norm_answer, winner.raw_answer)


def select_supported_ground(observations: Sequence[Observation]) -> Selection:
    by_source = {observation.source: observation for observation in observations}
    ground = by_source.get("grounding")
    if ground is not None and ground.usable:
        support = any(
            observation.usable
            and observation.source not in {"clean", "grounding"}
            and observation.norm_answer == ground.norm_answer
            for observation in observations
        )
        if support:
            return Selection("grounding", ground.norm_answer, ground.raw_answer)
    return select_keep(observations)


def select_oracle(observations: Sequence[Observation], gold_norm: str) -> Selection:
    clean = observations[0]
    if clean.usable and clean.norm_answer == gold_norm:
        return Selection("clean", clean.norm_answer, clean.raw_answer)
    for observation in observations[1:]:
        if observation.usable and observation.norm_answer == gold_norm:
            return Selection(observation.source, observation.norm_answer, observation.raw_answer)
    return select_keep(observations)


def select_with_scores(
    candidates: Sequence[Candidate],
    scores: Sequence[float],
    margin: float,
    probability_threshold: float,
    tie_tolerance: float,
) -> Selection:
    if len(candidates) != len(scores):
        raise PlanBError("Candidate/score length mismatch")
    if not candidates:
        return Selection("abstain", ABSTAIN, "")
    if not all(math.isfinite(float(score)) and 0.0 <= float(score) <= 1.0 for score in scores):
        raise PlanBError("Selector emitted an invalid probability")
    clean_indices = [index for index, candidate in enumerate(candidates) if "clean" in candidate.sources]
    clean_index = clean_indices[0] if clean_indices else None
    alternative_indices = [index for index in range(len(candidates)) if index != clean_index]
    if clean_index is None:
        best = max(alternative_indices, key=lambda index: (scores[index], -candidates[index].earliest_rank))
    elif not alternative_indices:
        best = clean_index
    else:
        best_alt = max(alternative_indices, key=lambda index: (scores[index], -candidates[index].earliest_rank))
        alt_score = float(scores[best_alt])
        clean_score = float(scores[clean_index])
        switch = (
            alt_score + tie_tolerance >= probability_threshold
            and alt_score - clean_score > margin + tie_tolerance
        )
        best = best_alt if switch else clean_index
    candidate = candidates[best]
    return Selection(candidate.sources[0], candidate.norm_answer, candidate.raw_answer)


def correctness(selection: Selection, gold_norm: str) -> int:
    return int(selection.norm_answer != ABSTAIN and selection.norm_answer == gold_norm)


def operational_cost(method: str, observations: Sequence[Observation], matched_cost: int) -> int:
    if method == "keep_original":
        return 0
    if method == "always_ground":
        return int(any(obs.source == "grounding" and obs.attempted for obs in observations))
    if method == "supported_ground":
        ground_seen = False
        for obs in observations[1:]:
            if obs.source == "grounding" and obs.attempted:
                ground_seen = True
            elif ground_seen and obs.attempted:
                return 2
        return int(ground_seen)
    return matched_cost


def load_and_verify_inputs(repo_root: Path, config: Mapping[str, Any], limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    remaining = limit
    for specification in config["input_files"]:
        path = (repo_root / specification["path"]).resolve()
        if not path.is_file():
            raise PlanBError(f"Missing pinned input: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != specification["sha256"]:
            raise PlanBError(f"Input hash mismatch for {specification['path']}: {actual_hash}")
        actual_bytes = path.stat().st_size
        if actual_bytes != int(specification["bytes"]):
            raise PlanBError(
                f"Input byte-size mismatch for {specification['path']}: {actual_bytes}"
            )
        rows = list(iter_jsonl(path))
        if len(rows) != int(specification["rows"]):
            raise PlanBError(f"Input row-count mismatch for {specification['path']}: {len(rows)}")
        inventory.append({"path": specification["path"], "sha256": actual_hash, "rows": len(rows), "bytes": actual_bytes})
        if remaining is not None:
            rows = rows[: max(0, remaining)]
            remaining -= len(rows)
        for row in rows:
            key = (str(row.get("instance_id")), str(row.get("model_id")))
            if not all(key):
                raise PlanBError(f"Missing record identity in {specification['path']}")
            if key in seen:
                raise PlanBError(f"Duplicate model-example record: {key}")
            seen.add(key)
            records.append(row)
        if remaining is not None and remaining <= 0:
            break
    return records, inventory


def build_cohort_index(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_by_image: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split_by_image[image_key(row)].add(str(row.get("split")))
    cohort: list[dict[str, Any]] = []
    exclusions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        split = str(row.get("split"))
        key = image_key(row)
        other = split_by_image[key] - {split}
        excluded = (split == "train" and bool(other & {"val", "cal", "test", "shift"})) or (
            split == "val" and bool(other & {"cal", "test", "shift"})
        )
        closed = is_closed_answer(row)
        fit = split in {"train", "val"} and closed and not excluded
        entry = {
            "instance_id": row["instance_id"],
            "group_id": row["group_id"],
            "image_key": key,
            "dataset": row["dataset"],
            "split": split,
            "model_id": row["model_id"],
            "answer_type": row.get("answer_type"),
            "closed_answer_eligible": int(closed),
            "image_excluded_from_fit": int(excluded),
            "proposed_fit_eligible": int(fit),
        }
        cohort.append(entry)
        if excluded:
            exclusions[(key, split)] = {
                "image_key": key,
                "excluded_split": split,
                "other_splits": "|".join(sorted(other)),
                "reason": "train_image_seen_later" if split == "train" else "validation_image_seen_in_cal_test_or_shift",
            }
    return cohort, sorted(exclusions.values(), key=lambda row: (row["image_key"], row["excluded_split"]))


def metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise PlanBError("Cannot summarize an empty evaluation slice")
    original_correct = sum(int(row["original_correct"]) for row in rows)
    final_correct = sum(int(row["correct"]) for row in rows)
    repairs = sum(int(row["repair"]) for row in rows)
    damage = sum(int(row["damage"]) for row in rows)
    originally_wrong = n - original_correct
    if final_correct != original_correct + repairs - damage:
        raise PlanBError("Correctness identity failed")
    return {
        "n": n,
        "original_accuracy": original_correct / n,
        "final_accuracy": final_correct / n,
        "repairs": repairs,
        "damage": damage,
        "fix_rate": repairs / originally_wrong if originally_wrong else None,
        "break_rate": damage / original_correct if original_correct else None,
        "abstentions": sum(int(row["abstain"]) for row in rows),
        "candidate_coverage": sum(int(row["candidate_count"] > 0) for row in rows) / n,
        "mean_matched_cost": sum(float(row["matched_cost"]) for row in rows) / n,
        "mean_operational_cost": sum(float(row["operational_cost"]) for row in rows) / n,
    }
