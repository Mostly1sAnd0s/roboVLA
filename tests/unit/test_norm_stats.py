"""core.norm_stats — 7-dim min/max stats + un-normalization (design §5.2, §9.5).

OpenVLA emits *normalized* 256-bin actions; the robot must un-normalize with
the dataset's per-dimension min/max (model-card-review.md: the #1 silent
killer). The action convention for this course is (vx, vy, wz, 0, 0, 0, 0),
so dims 4-7 are degenerate (min == max == 0) and must round-trip to 0.
"""

import math

import pytest

from robovla.core.errors import RobovlaError
from robovla.core.norm_stats import (
    ACTION_DIMS,
    NormStats,
    compute_norm_stats,
    denormalize,
    normalize,
)


def _stats(actions):
    return compute_norm_stats(actions)


class TestCompute:
    def test_exact_min_max_per_dim(self):
        stats = _stats([[1.0, 0.0, -1.0, 0, 0, 0, 0], [2.0, -0.5, 0.5, 0, 0, 0, 0]])
        assert stats.min == (1.0, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0)
        assert stats.max == (2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0)

    def test_seven_dims_enforced(self):
        assert ACTION_DIMS == 7
        with pytest.raises(RobovlaError) as e:
            _stats([[0.1, 0.2, 0.3]])
        assert e.value.code == "validation"

    def test_empty_dataset_rejected(self):
        with pytest.raises(RobovlaError) as e:
            _stats([])
        assert e.value.code == "validation"

    def test_non_finite_rejected(self):
        with pytest.raises(RobovlaError):
            _stats([[float("nan"), 0, 0, 0, 0, 0, 0]])
        with pytest.raises(RobovlaError):
            _stats([[float("inf"), 0, 0, 0, 0, 0, 0]])

    def test_seven_dim_course_convention(self):
        # (vx, vy, wz, 0, 0, 0, 0) — dims 4-7 degenerate (setup doc §6).
        stats = _stats([[0.3, -0.1, 0.9, 0.0, 0.0, 0.0, 0.0]] * 5)
        assert stats.min[3:] == (0.0, 0.0, 0.0, 0.0)
        assert stats.max[3:] == (0.0, 0.0, 0.0, 0.0)


class TestNormalizeDenormalize:
    def test_round_trip(self):
        actions = [[0.0, 0.0, 0.0, 0, 0, 0, 0], [0.5, -0.5, 1.0, 0, 0, 0, 0]]
        stats = _stats(actions)
        for a in actions:
            assert denormalize(normalize(a, stats), stats) == pytest.approx(a)

    def test_normalized_within_unit_interval(self):
        stats = _stats([[1.0, 2.0, -3.0, 0, 0, 0, 0], [3.0, 1.0, -1.0, 0, 0, 0, 0]])
        for n in (normalize([2.0, 1.5, -2.0, 0, 0, 0, 0], stats),):
            assert all(0.0 <= v <= 1.0 for v in n)

    def test_endpoints(self):
        stats = _stats([[0.0, 0.0, 0.0, 0, 0, 0, 0], [2.0, -2.0, 4.0, 0, 0, 0, 0]])
        assert normalize([0.0, -2.0, 0.0, 0, 0, 0, 0], stats) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert normalize([2.0, 0.0, 4.0, 0, 0, 0, 0], stats) == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    def test_degenerate_dim_round_trips_to_its_value(self):
        stats = _stats([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]] * 3)
        norm = normalize([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], stats)
        assert norm[3:] == (0.0, 0.0, 0.0, 0.0)
        assert denormalize(norm, stats) == pytest.approx([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])

    def test_wrong_dim_count_rejected(self):
        stats = NormStats((0.0,) * 7, (1.0,) * 7)
        with pytest.raises(RobovlaError) as e:
            normalize([0.5, 0.5], stats)
        assert e.value.code == "validation"

    def test_non_finite_rejected(self):
        stats = NormStats((0.0,) * 7, (1.0,) * 7)
        with pytest.raises(RobovlaError):
            denormalize([0.5] * 6 + [math.nan], stats)
