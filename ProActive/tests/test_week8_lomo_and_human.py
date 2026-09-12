import json
import math
from pathlib import Path

import pytest

from scripts.analyze_human_audit import _fleiss_kappa
from scripts.build_lomo_fold import _filter_jsonl, _keep, _row_split_and_model


def test_lomo_split_boundary_excludes_heldout_model_before_test() -> None:
    heldout = "heldout-model"
    source = "source-model"
    for split in ("train", "val", "cal"):
        assert _keep(split, source, heldout) is True
        assert _keep(split, heldout, heldout) is False
    assert _keep("test", heldout, heldout) is True
    assert _keep("test", source, heldout) is False
    assert _keep("shift", heldout, heldout) is False


def test_lomo_reads_partial_state_identity_from_metadata() -> None:
    row = {
        "record_type": "partial_state_v1",
        "metadata": {"split": "train", "model_id": "source-model"},
    }
    assert _row_split_and_model(row, Path("states.jsonl")) == (
        "train",
        "source-model",
    )


def test_lomo_reads_teacher_identity_from_top_level() -> None:
    row = {
        "record_type": "teacher_cache",
        "split": "test",
        "model_id": "heldout-model",
    }
    assert _row_split_and_model(row, Path("teacher.jsonl")) == (
        "test",
        "heldout-model",
    )


def test_lomo_rejects_partial_state_without_metadata() -> None:
    with pytest.raises(SystemExit, match="Missing partial-state metadata"):
        _row_split_and_model(
            {"record_type": "partial_state_v1"}, Path("states.jsonl")
        )


def test_lomo_filter_keeps_source_development_and_heldout_test_states(
    tmp_path: Path,
) -> None:
    source = tmp_path / "states_source.jsonl"
    destination = tmp_path / "states_lomo.jsonl"
    rows = [
        {"record_type": "partial_state_v1", "metadata": {"split": "train", "model_id": "source"}},
        {"record_type": "partial_state_v1", "metadata": {"split": "train", "model_id": "heldout"}},
        {"record_type": "partial_state_v1", "metadata": {"split": "test", "model_id": "heldout"}},
        {"record_type": "partial_state_v1", "metadata": {"split": "test", "model_id": "source"}},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = _filter_jsonl(
        [source],
        destination,
        heldout_model_id="heldout",
        overwrite=False,
    )

    kept = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert report["row_count"] == 2
    assert report["split_counts"] == {"test": 1, "train": 1}
    assert [(row["metadata"]["split"], row["metadata"]["model_id"]) for row in kept] == [
        ("train", "source"),
        ("test", "heldout"),
    ]


def test_fleiss_kappa_is_one_for_perfect_multicategory_agreement() -> None:
    rows = [
        ["visual", "visual", "visual"],
        ["language-prior", "language-prior", "language-prior"],
        ["mixed", "mixed", "mixed"],
    ]
    assert math.isclose(_fleiss_kappa(rows), 1.0)


def test_fleiss_kappa_fails_closed_when_undefined() -> None:
    with pytest.raises(ValueError, match="undefined"):
        _fleiss_kappa([["visual", "visual", "visual"]] * 3)
