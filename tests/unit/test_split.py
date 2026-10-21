"""core.split — seeded held-out split (design §5.2).

Binding property: the split is **stable per student + task across dataset
versions**. v1 -> v2 comparisons must run on the same held-out episodes, and
re-uploads must not reshuffle the holdout. Membership of an episode therefore
depends only on (student, task, episode_id) — never on the other episodes in
the set (per-episode hash bucketing, not shuffle-then-slice).
"""

import random

import pytest

from robovla.core.errors import RobovlaError
from robovla.core.split import (
    CROSSCHECK_N,
    HELD_OUT_FRACTION,
    crosscheck_subset,
    held_out_set,
    split,
)


def _ids(n, prefix="ep"):
    return [f"{prefix}_{i:04d}" for i in range(n)]


class TestStability:
    def test_default_fraction_is_15_percent(self):
        assert HELD_OUT_FRACTION == pytest.approx(0.15)

    def test_deterministic(self):
        ids = _ids(50)
        a = held_out_set(ids, "s07", "stop-at-target")
        b = held_out_set(ids, "s07", "stop-at-target")
        assert a == b

    def test_order_insensitive(self):
        ids = _ids(50)
        base = held_out_set(ids, "s07", "stop-at-target")
        shuffled = ids[:]
        random.Random(1).shuffle(shuffled)
        assert held_out_set(shuffled, "s07", "stop-at-target") == base

    def test_stable_across_versions(self):
        # v1 episodes keep their membership when v2 adds more episodes.
        v1 = _ids(100)
        v2 = v1 + [f"new_{i:04d}" for i in range(10)]
        h1 = held_out_set(v1, "s01", "task-a")
        h2 = held_out_set(v2, "s01", "task-a")
        for ep in v1:
            assert (ep in h1) == (ep in h2), "v1 membership must not change in v2"

    def test_stable_when_episodes_removed(self):
        v1 = _ids(100)
        v2 = v1[:-5]
        h1 = held_out_set(v1, "s02", "task-b")
        h2 = held_out_set(v2, "s02", "task-b")
        for ep in v2:
            assert (ep in h1) == (ep in h2)

    def test_seed_is_student_and_task(self):
        ids = _ids(50)
        h_student = held_out_set(ids, "s01", "task")
        h_student2 = held_out_set(ids, "s02", "task")
        h_task = held_out_set(ids, "s01", "other-task")
        # with 50 ids the probability all 50 memberships coincidentally match
        # is 2**-50; a collision here means the seed lost its inputs.
        assert any((a in h_student) != (a in h_student2) for a in ids)
        assert any((a in h_student) != (a in h_task) for a in ids)

    def test_duplicate_ids_are_deduplicated(self):
        ids = ["ep_0001", "ep_0001", "ep_0002"]
        assert len(held_out_set(ids, "s01", "t")) <= 2


class TestFraction:
    def test_large_set_is_near_15_percent(self):
        ids = _ids(1000)
        h = held_out_set(ids, "s03", "task-c")
        # 150 expected, sigma ~ 11.3 -> [100, 200] is ~+-4.4 sigma.
        assert 100 <= len(h) <= 200

    def test_fraction_bounds_respected_for_small_set(self):
        ids = _ids(10)
        h = held_out_set(ids, "s04", "task-d")
        assert 0 <= len(h) <= 10

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_fraction_must_be_open_interval(self, fraction):
        with pytest.raises(RobovlaError) as e:
            held_out_set(_ids(5), "s01", "t", fraction=fraction)
        assert e.value.code == "validation"

    def test_empty_episodes_give_empty_split(self):
        assert held_out_set([], "s01", "t") == frozenset()

    def test_empty_student_or_task_rejected(self):
        with pytest.raises(RobovlaError):
            held_out_set(_ids(5), "", "t")
        with pytest.raises(RobovlaError):
            held_out_set(_ids(5), "s01", "")


class TestSplitResult:
    def test_buckets_partition_input_preserving_order(self):
        ids = _ids(60)
        result = split(ids, "s05", "task-e")
        # partition of the input, each bucket preserving input order
        assert set(result.train) | set(result.held_out) == set(ids)
        assert not (set(result.train) & set(result.held_out))
        assert len(result.train) + len(result.held_out) == len(ids)
        assert list(result.train) == [e for e in ids if e not in result.held_out_set]
        assert list(result.held_out) == [e for e in ids if e in result.held_out_set]

    def test_result_matches_held_out_set(self):
        ids = _ids(40)
        result = split(ids, "s06", "task-f")
        assert result.held_out_set == held_out_set(ids, "s06", "task-f")


class TestCrosscheckSubset:
    """crosscheck_subset (§6): the deterministic 50-episode subset the
    crosscheck job runs the official engine on — same episodes every re-run,
    seeded per student + dataset version (per-episode bucketing, so other
    episodes present never reshuffle the choice)."""

    def test_default_n_is_50(self):
        assert CROSSCHECK_N == 50

    def test_deterministic(self):
        ids = _ids(100)
        assert crosscheck_subset(ids, "s01", "s01/task-a/v1") == \
            crosscheck_subset(ids, "s01", "s01/task-a/v1")

    def test_size_and_membership(self):
        ids = _ids(100)
        sub = crosscheck_subset(ids, "s01", "s01/task-a/v1")
        assert len(sub) == 50
        assert set(sub) <= set(ids)
        assert len(set(sub)) == 50  # no duplicates
        # input order preserved (a subsequence of the input)
        it = iter(ids)
        assert all(e in it for e in sub)

    def test_seed_by_student(self):
        ids = _ids(200)
        a = set(crosscheck_subset(ids, "s01", "s01/task-a/v1"))
        b = set(crosscheck_subset(ids, "s02", "s01/task-a/v1"))
        assert a != b  # different student -> different subset
        # but not fully disjoint either (50 of 200 each)
        assert len(a & b) > 0

    def test_seed_by_dataset_version(self):
        ids = _ids(200)
        v1 = set(crosscheck_subset(ids, "s01", "s01/task-a/v1"))
        v2 = set(crosscheck_subset(ids, "s01", "s01/task-a/v2"))
        assert v1 != v2

    def test_stable_under_membership_change(self):
        ids = _ids(100)
        full = crosscheck_subset(ids, "s01", "s01/task-a/v1")
        # keep all 50 chosen + exactly 40 others (input order): unchanged
        non_sub = [e for e in ids if e not in full][:40]
        kept = [e for e in ids if e in full or e in set(non_sub)]
        assert len(kept) == 90
        assert crosscheck_subset(kept, "s01", "s01/task-a/v1") == full

    def test_fewer_than_n_returns_all(self):
        ids = _ids(30)
        assert crosscheck_subset(ids, "s01", "s01/task-a/v1") == tuple(ids)

    def test_invalid(self):
        ids = _ids(5)
        for args in ((ids, "", "s01/t/v1"), (ids, "s01", ""),
                     (ids, "s01", "s01/t/v1", 0), (ids, "s01", "s01/t/v1", -3)):
            with pytest.raises(RobovlaError):
                crosscheck_subset(*args)
