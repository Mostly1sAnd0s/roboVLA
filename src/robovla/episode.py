"""The shared single-episode validator (design §11: one episode validator).

Vendored verbatim into the standalone capture app and the runner agent so
capture-side, console-side, and agent-side QA can never drift. Pure: operates
on parsed episode data — reading ``frames/`` / ``actions.json`` /
``episode.json`` is the seam's job (fs.py / the capture app).

Episode format (setup-and-materials §6):
  * ``actions.json``: ``[{t, vx, vy, wz}, ...]`` at 10 Hz, velocities
    normalized to [-1, 1] against the autonomy caps;
  * ``frames/*.jpg`` at 10 Hz (counted by the seam, passed as ``frame_count``);
  * ``episode.json``: ``{id, task, prompt, start_pose, success, pilot, notes, created}``.

Severity: ``fail`` rules block the dataset from "ready" (the QA gate, §5.2);
``warn`` rules flag data the model is known to struggle with (official
OpenVLA guidance: avoid idle/pause actions, the model can get stuck).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .core.errors import RobovlaError

# --- thresholds (official guidance: 10 Hz, no pauses, consistent motion) ---

DT_NOMINAL = 0.1  # s between actions (10 Hz)
DT_TOL = 0.02  # s — acceptance band around the nominal dt
DT_TOLERANCE = 0.10  # max fraction of intervals outside the band before fail
IDLE_EPS = 0.05  # "near-zero" velocity on all dims
IDLE_MAX_S = 2.0  # s — idle longer than this warns (official: avoid pauses)
DEADZONE = 0.1  # sign flips only count when both sides exceed this
MAX_SIGN_FLIPS = 3  # hard reversals per episode before warning
JUMP_TH = 0.8  # per-dim jump between consecutive actions
MAX_JUMPS = 3  # jumps per episode before warning
RANGE = 1.0  # actions are normalized to [-1, 1]

EP_ID_RE = re.compile(r"^ep_\d+$")  # folder name, e.g. ep_0042

_DRIVING_DIMS = ("vx", "vy", "wz")


# --- data model ----------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One logged action: t in seconds, velocities normalized to [-1, 1]."""

    t: float
    vx: float
    vy: float
    wz: float


@dataclass(frozen=True)
class Episode:
    """A parsed episode folder (frame reading happens at the seam)."""

    id: str
    frame_count: int
    actions: tuple[Action, ...]
    task: str = ""
    prompt: str = ""
    success: bool | None = None
    pilot: str = ""
    notes: str = ""
    created: str | None = None
    start_pose: Any = None


@dataclass(frozen=True)
class Issue:
    """One QA finding. ``rule`` is a stable id; ``severity`` is fail|warn."""

    rule: str
    severity: str
    detail: str


@dataclass(frozen=True)
class EpisodeReport:
    """Result of validating one episode: status ok|warn|fail + issues."""

    episode_id: str
    status: str
    issues: tuple[Issue, ...] = field(default=())


def to_action_7d(a: Action) -> tuple[float, ...]:
    """Course action convention (setup §6): (vx, vy, wz) -> 7-dim, zeros for
    the unused dims — feeds norm-stats computation and the training set."""
    return (a.vx, a.vy, a.wz, 0.0, 0.0, 0.0, 0.0)


# --- validation ------------------------------------------------------------------


def _check_finite(actions: tuple[Action, ...]) -> None:
    for i, a in enumerate(actions):
        for name in ("t",) + _DRIVING_DIMS:
            v = getattr(a, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise RobovlaError(
                    "validation", f"action #{i} field {name!r} must be a finite number"
                )


def _check_format(ep: Episode, issues: list[Issue]) -> None:
    if not EP_ID_RE.fullmatch(ep.id):
        issues.append(Issue("format", "fail", f"episode id {ep.id!r} must look like 'ep_0042'"))
    if not ep.actions:
        issues.append(Issue("empty", "fail", "episode has no actions"))
        return
    if ep.frame_count != len(ep.actions):
        issues.append(
            Issue(
                "frame_action_mismatch",
                "fail",
                f"{len(ep.actions)} actions vs {ep.frame_count} frames (1:1 required)",
            )
        )


def _check_range(ep: Episode, issues: list[Issue]) -> None:
    bad = [
        i
        for i, a in enumerate(ep.actions)
        if max(abs(a.vx), abs(a.vy), abs(a.wz)) > RANGE
    ]
    if bad:
        a = ep.actions[bad[0]]
        issues.append(
            Issue(
                "action_range",
                "fail",
                f"{len(bad)} action(s) outside [-1, 1] (normalized), e.g. #{bad[0]} vx={a.vx!r}",
            )
        )


def _check_timing(ep: Episode, issues: list[Issue]) -> None:
    acts = ep.actions
    dts = [acts[i + 1].t - acts[i].t for i in range(len(acts) - 1)]
    if not dts:
        return
    non_mono = sum(1 for dt in dts if dt <= 0)
    if non_mono:
        issues.append(Issue("timing_monotonic", "fail", f"{non_mono} non-positive interval(s)"))
    lo, hi = DT_NOMINAL - DT_TOL, DT_NOMINAL + DT_TOL
    off = sum(1 for dt in dts if 0 < dt and not (lo <= dt <= hi))
    if off / len(dts) > DT_TOLERANCE:
        issues.append(
            Issue(
                "timing_10hz",
                "fail",
                f"{off} of {len(dts)} intervals outside [{lo:.2f}, {hi:.2f}] s (expected 10 Hz)",
            )
        )


def _check_idle(ep: Episode, issues: list[Issue]) -> None:
    """Idle = near-zero on all driving dims; warn when sustained > IDLE_MAX_S.

    Official OpenVLA guidance: avoid pauses/idle actions — the model can get
    stuck on near-zero actions at inference (model-card-review §1).
    """
    acts = ep.actions
    worst = 0.0
    run_start = None
    for i, a in enumerate(acts + (None,)):  # sentinel closes the final run
        idle = a is not None and all(abs(getattr(a, d)) < IDLE_EPS for d in _DRIVING_DIMS)
        if idle and run_start is None:
            run_start = i
        elif not idle and run_start is not None:
            worst = max(worst, acts[i - 1].t - acts[run_start].t)
            run_start = None
    if worst > IDLE_MAX_S:
        issues.append(
            Issue("idle_pause", "warn", f"robot idle {worst:.1f} s (> {IDLE_MAX_S:g} s) — model can get stuck")
        )


def _check_sign_flips(ep: Episode, issues: list[Issue]) -> None:
    """Hard reversals: sign change with both sides above the deadzone.

    Flips through tiny velocities (< DEADZONE) are not counted — a student
    easing through a stop is normal; repeated full-speed reversals are not a
    consistent driving strategy (official rule 4).
    """
    acts = ep.actions
    flips = 0
    for i in range(len(acts) - 1):
        for d in _DRIVING_DIMS:
            a, b = getattr(acts[i], d), getattr(acts[i + 1], d)
            if a * b < 0 and min(abs(a), abs(b)) >= DEADZONE:
                flips += 1
    if flips > MAX_SIGN_FLIPS:
        issues.append(
            Issue(
                "sign_flip",
                "warn",
                f"{flips} hard direction reversals (> {MAX_SIGN_FLIPS}) — inconsistent strategy?",
            )
        )


def _check_jumps(ep: Episode, issues: list[Issue]) -> None:
    acts = ep.actions
    jumps = 0
    for i in range(len(acts) - 1):
        for d in _DRIVING_DIMS:
            if abs(getattr(acts[i + 1], d) - getattr(acts[i], d)) > JUMP_TH:
                jumps += 1
    if jumps > MAX_JUMPS:
        issues.append(
            Issue("jump_anomaly", "warn", f"{jumps} per-dim jumps > {JUMP_TH} (> {MAX_JUMPS}) — sensor/log glitch?")
        )


def validate_episode(ep: Episode) -> EpisodeReport:
    """Validate one parsed episode -> :class:`EpisodeReport`.

    Raises :class:`RobovlaError` (``validation``) on structurally unusable
    input (non-finite fields); every QA finding is an Issue, never an error.
    """
    if not isinstance(ep, Episode):
        raise RobovlaError("validation", "validate_episode expects an Episode")
    _check_finite(ep.actions)
    issues: list[Issue] = []
    _check_format(ep, issues)
    if ep.actions:
        _check_range(ep, issues)
        _check_timing(ep, issues)
        _check_idle(ep, issues)
        _check_sign_flips(ep, issues)
        _check_jumps(ep, issues)
    status = "fail" if any(i.severity == "fail" for i in issues) else (
        "warn" if issues else "ok"
    )
    return EpisodeReport(episode_id=ep.id, status=status, issues=tuple(issues))
