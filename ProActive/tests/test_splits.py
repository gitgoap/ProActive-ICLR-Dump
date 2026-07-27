"""
Tests for grouped split construction.

Verifies:
- No group_id appears in more than one split
- All records with the same group_id land in the same split
- Split ratios are approximately correct
- Deterministic: same seed → same assignment
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.data.splits import (
    assign_split,
    build_grouped_splits,
    compute_group_id,
    get_split_stats,
    validate_no_group_overlap,
    SPLIT_NAMES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_records(n_groups: int = 200, records_per_group: int = 3):
    """Create synthetic records with realistic group structure."""
    records = []
    for g in range(n_groups):
        gid = compute_group_id("test_dataset", f"img_{g}", f"q_{g}")
        for r in range(records_per_group):
            records.append({
                "instance_id": f"test_{g}_{r}",
                "group_id": gid,
                "dataset": "test_dataset",
                "model_id": f"model_{r}",
            })
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGroupIdComputation:

    def test_deterministic(self):
        """Same inputs always produce the same group_id."""
        gid1 = compute_group_id("vsr", "img_42", "q_0")
        gid2 = compute_group_id("vsr", "img_42", "q_0")
        assert gid1 == gid2

    def test_different_datasets_different_ids(self):
        """Different datasets produce different group_ids."""
        gid1 = compute_group_id("vsr", "img_1", "q_1")
        gid2 = compute_group_id("pope", "img_1", "q_1")
        assert gid1 != gid2

    def test_starts_with_sha256(self):
        gid = compute_group_id("test", "img", "q")
        assert gid.startswith("sha256:")


class TestSplitAssignment:

    def test_deterministic(self):
        """Same group_id + seed always gets the same split."""
        s1 = assign_split("sha256:abc123", seed=42)
        s2 = assign_split("sha256:abc123", seed=42)
        assert s1 == s2

    def test_valid_split_name(self):
        """Assigned split is always a valid name."""
        for i in range(100):
            gid = f"sha256:test_{i}"
            split = assign_split(gid, seed=42)
            assert split in SPLIT_NAMES

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different assignments."""
        results_42 = set()
        results_99 = set()
        for i in range(50):
            gid = f"sha256:group_{i}"
            results_42.add(assign_split(gid, seed=42))
            results_99.add(assign_split(gid, seed=99))
        # With 50 groups and 4 splits, both should hit multiple splits
        assert len(results_42) > 1
        assert len(results_99) > 1


class TestBuildGroupedSplits:

    def test_no_group_overlap(self):
        """No group_id appears in multiple splits."""
        records = _make_records(n_groups=200, records_per_group=3)
        build_grouped_splits(records, seed=42)
        valid, violations = validate_no_group_overlap(records)
        assert valid, f"Violations found: {violations}"

    def test_all_records_same_group_same_split(self):
        """All records with the same group_id get the same split."""
        records = _make_records(n_groups=100, records_per_group=5)
        build_grouped_splits(records, seed=42)

        from collections import defaultdict
        group_splits = defaultdict(set)
        for r in records:
            group_splits[r["group_id"]].add(r["split"])

        for gid, splits in group_splits.items():
            assert len(splits) == 1, (
                f"Group {gid} has records in multiple splits: {splits}"
            )

    def test_approximate_ratios(self):
        """Split ratios are approximately correct with enough groups."""
        records = _make_records(n_groups=1000, records_per_group=1)
        build_grouped_splits(records, seed=42)
        stats = get_split_stats(records)

        total = sum(s["records"] for s in stats.values())
        train_ratio = stats["train"]["records"] / total
        val_ratio = stats["val"]["records"] / total
        cal_ratio = stats["cal"]["records"] / total
        test_ratio = stats["test"]["records"] / total

        # Allow ±5% tolerance
        assert abs(train_ratio - 0.70) < 0.05, f"Train ratio: {train_ratio}"
        assert abs(val_ratio - 0.10) < 0.05, f"Val ratio: {val_ratio}"
        assert abs(cal_ratio - 0.10) < 0.05, f"Cal ratio: {cal_ratio}"
        assert abs(test_ratio - 0.10) < 0.05, f"Test ratio: {test_ratio}"

    def test_all_splits_populated(self):
        """With enough groups, all four splits have records."""
        records = _make_records(n_groups=500, records_per_group=1)
        build_grouped_splits(records, seed=42)
        stats = get_split_stats(records)

        for split_name in SPLIT_NAMES:
            assert stats[split_name]["records"] > 0, (
                f"Split '{split_name}' has no records"
            )
