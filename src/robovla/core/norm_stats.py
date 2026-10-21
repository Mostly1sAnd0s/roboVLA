"""Per-dimension min/max normalization stats for 7-dim OpenVLA actions (design §5.2).

OpenVLA outputs *normalized* 256-bin actions; the robot must un-normalize with
the dataset's per-dimension min/max stats before execution (model-card-review:
without this the robot receives meaningless numbers). This module is the
single place that math lives — the console's prep job, the agent's
``/predict`` response, and the capture app (vendored) all use these functions
so capture-side and console-side stats can never drift (design §11).

Course action convention (setup-and-materials §6): ``(vx, vy, wz, 0, 0, 0, 0)``
— dims 4-7 are zeros, so their min == max == 0 (degenerate); the round-trip
holds there by definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .errors import RobovlaError

#: OpenVLA action space: 7-dim delta pose, 256 bins per dim.
ACTION_DIMS = 7


@dataclass(frozen=True)
class NormStats:
    """Per-dimension min/max over a dataset (the ``norm_stats.json`` content)."""

    min: tuple[float, ...]
    max: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.min) != ACTION_DIMS or len(self.max) != ACTION_DIMS:
            raise RobovlaError("validation", f"norm stats must have {ACTION_DIMS} dims")
        for m, M in zip(self.min, self.max):
            if not math.isfinite(m) or not math.isfinite(M):
                raise RobovlaError("validation", "norm stats must be finite")
            if m > M:
                raise RobovlaError("validation", "norm stats min must be <= max")

    def to_dict(self) -> dict:
        """JSON shape for ``norm_stats.json`` / API responses."""
        return {"dims": ACTION_DIMS, "min": list(self.min), "max": list(self.max)}


def _check_action(action: Sequence[float], what: str = "action") -> tuple[float, ...]:
    if len(action) != ACTION_DIMS:
        raise RobovlaError("validation", f"{what} must have {ACTION_DIMS} dims, got {len(action)}")
    out = []
    for v in action:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise RobovlaError("validation", f"{what} values must be finite numbers")
        out.append(float(v))
    return tuple(out)


def compute_norm_stats(actions: Sequence[Sequence[float]]) -> NormStats:
    """Per-dim min/max over a non-empty set of 7-dim actions."""
    if not actions:
        raise RobovlaError("validation", "cannot compute norm stats over zero actions")
    first = _check_action(actions[0], "first action")
    mins = list(first)
    maxs = list(first)
    for action in actions[1:]:
        a = _check_action(action, "action")
        for i, v in enumerate(a):
            if v < mins[i]:
                mins[i] = v
            if v > maxs[i]:
                maxs[i] = v
    return NormStats(tuple(mins), tuple(maxs))


def normalize(action: Sequence[float], stats: NormStats) -> tuple[float, ...]:
    """Map a raw action to [0, 1] per dim.

    Degenerate dims (min == max) map to 0.0. Values outside the observed
    [min, max] are not clipped: stats are computed from the same dataset the
    action came from, so out-of-range values are a caller bug, not a clamp
    case (inference bins are always inside [0, 1] by construction).
    """
    a = _check_action(action)
    out = []
    for v, lo, hi in zip(a, stats.min, stats.max):
        span = hi - lo
        out.append(0.0 if span == 0.0 else (v - lo) / span)
    return tuple(out)


def denormalize(norm_action: Sequence[float], stats: NormStats) -> tuple[float, ...]:
    """Map normalized actions back to robot units: ``min + n * (max - min)``.

    This is the Pi-side step the model card calls out (``unnorm_key``); the
    console's ``/api/robot/predict`` returns both ``action_norm`` and
    ``action_robot`` (design §9.5).
    """
    n = _check_action(norm_action, "normalized action")
    out = []
    for v, lo, hi in zip(n, stats.min, stats.max):
        span = hi - lo
        out.append(lo if span == 0.0 else lo + v * span)
    return tuple(out)
