"""Dataset-level QA rules (design §5.2): the gate that stops bad data from
eating queue time.

Composes per-episode :class:`~robovla.episode.EpisodeReport` results (from
the shared validator) with dataset-level rules. This is what backs
``GET /api/datasets/{id}/qa`` — the QA report page students see before any
queue time is spent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .errors import RobovlaError
from ..episode import EpisodeReport

#: Official floor: OpenVLA typically needs ~100 demos (model-card-review).
#: Below this the dataset is still *ready* — the warning is the teacher's call.
MIN_EPISODES = 100

_SEV_ORDER = {"fail": 0, "warn": 1}


@dataclass(frozen=True)
class QAIssue:
    """One finding attributed to an episode (``episode_id``) or the dataset
    (empty string). ``severity`` is fail|warn."""

    episode_id: str
    rule: str
    severity: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "rule": self.rule,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class QASummary:
    episode_count: int
    ok: int
    warned: int
    failed: int


@dataclass(frozen=True)
class DatasetQA:
    """Dataset QA verdict: ``status`` is ready|failed (warnings never block)."""

    status: str
    summary: QASummary
    dataset_issues: tuple[QAIssue, ...]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "summary": {
                "episode_count": self.summary.episode_count,
                "ok": self.summary.ok,
                "warned": self.summary.warned,
                "failed": self.summary.failed,
            },
            "issues": [i.to_dict() for i in self.dataset_issues],
        }


def qa_dataset(
    reports: Sequence[EpisodeReport],
    total_bytes: int,
    max_bytes: int,
) -> DatasetQA:
    """Run the dataset-level QA gate over per-episode reports.

    * any episode fail  -> dataset ``failed``
    * bytes over cap    -> ``failed`` (size_cap_exceeded)
    * zero episodes     -> ``failed`` (no_episodes)
    * < MIN_EPISODES    -> warning only (too_few_episodes)
    """
    if not isinstance(total_bytes, int) or isinstance(total_bytes, bool) or total_bytes < 0:
        raise RobovlaError("validation", "total_bytes must be an int >= 0")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise RobovlaError("validation", "max_bytes must be an int >= 1")

    issues = [
        QAIssue(r.episode_id, i.rule, i.severity, i.detail)
        for r in reports
        for i in r.issues
    ]
    if not reports:
        issues.append(QAIssue("", "no_episodes", "fail", "dataset has no episodes"))
    if total_bytes > max_bytes:
        issues.append(
            QAIssue(
                "",
                "size_cap_exceeded",
                "fail",
                f"{total_bytes} bytes exceeds the {max_bytes} byte cap",
            )
        )
    if reports and len(reports) < MIN_EPISODES:
        issues.append(
            QAIssue(
                "",
                "too_few_episodes",
                "warn",
                f"{len(reports)} episodes < {MIN_EPISODES}; expect underfit "
                "(official ~100-demo guidance)",
            )
        )

    summary = QASummary(
        episode_count=len(reports),
        ok=sum(1 for r in reports if r.status == "ok"),
        warned=sum(1 for r in reports if r.status == "warn"),
        failed=sum(1 for r in reports if r.status == "fail"),
    )
    status = "failed" if any(i.severity == "fail" for i in issues) else "ready"
    return DatasetQA(status=status, summary=summary, dataset_issues=tuple(issues))
