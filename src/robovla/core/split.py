"""Seeded held-out split, stable per student + task across dataset versions (design §5.2).

Binding property: an episode's held-out membership depends **only** on
``(student, task, episode_id)`` — never on the other episodes present. So:

* v1 -> v2 comparisons run on the same held-out episodes (adding or removing
  other episodes cannot reshuffle the holdout);
* re-uploads of the same content give the same split;
* the split is deterministic with no RNG state (no wall clock, no seed
  parameter — the seed *is* the student + task, per §5.2).

Implementation: per-episode hash bucketing (sha256 of ``student|task|episode``
mapped to a uniform [0, 1) float; held out iff below ``fraction``). This gives
an expected ``fraction`` of episodes (15% for 100 episodes ~ 15, sigma ~ 3.6)
with exact stability, which a shuffle-then-slice scheme cannot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .errors import RobovlaError

#: Default held-out fraction (design §5.2: 15%).
HELD_OUT_FRACTION = 0.15


def _check_inputs(student: str, task: str, fraction: float) -> None:
    if not isinstance(student, str) or not student:
        raise RobovlaError("validation", "student must be a non-empty string")
    if not isinstance(task, str) or not task:
        raise RobovlaError("validation", "task must be a non-empty string")
    if not isinstance(fraction, (int, float)) or not (0.0 < fraction < 1.0):
        raise RobovlaError("validation", "fraction must be in the open interval (0, 1)")


def _bucket(student: str, task: str, episode_id: str) -> float:
    """Uniform [0, 1) value for one (student, task, episode) triple."""
    key = f"{student}|{task}|{episode_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    n = int.from_bytes(digest[:8], "big")
    return n / float(1 << 64)


def held_out_set(
    episodes: Sequence[str],
    student: str,
    task: str,
    fraction: float = HELD_OUT_FRACTION,
) -> frozenset[str]:
    """The set of episode ids held out for evaluation.

    Order- and membership-insensitive for unchanged ids (see module docstring).
    Duplicate ids are deduplicated (an episode is either held out or not).
    """
    _check_inputs(student, task, fraction)
    return frozenset(
        ep for ep in dict.fromkeys(episodes) if _bucket(student, task, ep) < fraction
    )


@dataclass(frozen=True)
class Split:
    """Held-out split with input order preserved inside each bucket."""

    train: tuple[str, ...]
    held_out: tuple[str, ...]
    held_out_set: frozenset[str]


def split(
    episodes: Sequence[str],
    student: str,
    task: str,
    fraction: float = HELD_OUT_FRACTION,
) -> Split:
    """Partition ``episodes`` into (train, held_out), preserving input order.

    ``held_out_set`` is exposed so callers (QA report, JobSpec, crosscheck
    subset) can check membership without re-scanning the bucket.
    """
    h = held_out_set(episodes, student, task, fraction)
    train = []
    held = []
    for ep in episodes:
        (held if ep in h else train).append(ep)
    return Split(tuple(train), tuple(held), h)
