"""core.qa — dataset-level QA rules (design §5.2): the gate that stops bad
data from eating queue time. Composes per-episode reports (episode.py) with
dataset-level rules; emits the QA summary behind ``GET /api/datasets/{id}/qa``.
"""

import pytest

from robovla.core.errors import RobovlaError
from robovla.core.qa import MIN_EPISODES, DatasetQA, qa_dataset
from robovla.episode import Action, Episode, EpisodeReport, Issue


def _report(ep="ep_0001", status="ok", issues=()):
    return EpisodeReport(episode_id=ep, status=status, issues=tuple(issues))


def _episode_reports(n=100):
    return [_report(f"ep_{i:04d}") for i in range(n)]


def _episode(n=100):
    return Episode(
        id="ep_0001",
        frame_count=n,
        actions=tuple(Action(t=i * 0.1, vx=0.2, vy=0.0, wz=0.0) for i in range(n)),
        task="stop at the target",
    )


class TestStatus:
    def test_all_ok_is_ready(self):
        qa = qa_dataset(_episode_reports(), total_bytes=500_000_000, max_bytes=2**31)
        assert qa.status == "ready"
        assert qa.summary.ok == 100
        assert qa.summary.failed == 0
        assert qa.summary.warned == 0
        assert qa.dataset_issues == ()

    def test_one_failed_episode_fails_dataset(self):
        reports = _episode_reports()
        reports[7] = _report(
            "ep_0008",
            "fail",
            (Issue("frame_action_mismatch", "fail", "100 frames vs 99 actions"),),
        )
        qa = qa_dataset(reports, total_bytes=500_000_000, max_bytes=2**31)
        assert qa.status == "failed"
        fail = next(f for f in qa.dataset_issues if f.severity == "fail")
        assert fail.episode_id == "ep_0008"
        assert fail.rule == "frame_action_mismatch"

    def test_warnings_do_not_block_ready(self):
        reports = _episode_reports()
        reports[3] = _report("ep_0004", "warn", (Issue("idle_pause", "warn", "2.4 s idle"),))
        qa = qa_dataset(reports, total_bytes=500_000_000, max_bytes=2**31)
        assert qa.status == "ready"
        assert qa.summary.warned == 1

    def test_empty_dataset_fails(self):
        qa = qa_dataset([], total_bytes=0, max_bytes=2**31)
        assert qa.status == "failed"
        assert any(i.rule == "no_episodes" and i.severity == "fail" for i in qa.dataset_issues)

    def test_size_cap(self):
        over = qa_dataset(_episode_reports(), total_bytes=2**31 + 1, max_bytes=2**31)
        assert over.status == "failed"
        assert any(i.rule == "size_cap_exceeded" for i in over.dataset_issues)
        at_cap = qa_dataset(_episode_reports(), total_bytes=2**31, max_bytes=2**31)
        assert at_cap.status == "ready"

    def test_min_episode_floor_is_official_guidance(self):
        assert MIN_EPISODES == 100  # model-card-review: ~100 demos floor

    def test_below_floor_warns_but_stays_ready(self):
        qa = qa_dataset(_episode_reports(50), total_bytes=200_000_000, max_bytes=2**31)
        assert qa.status == "ready"
        warn = next(i for i in qa.dataset_issues if i.rule == "too_few_episodes")
        assert warn.severity == "warn"
        assert "50" in warn.detail

    def test_at_floor_no_warning(self):
        qa = qa_dataset(_episode_reports(100), total_bytes=200_000_000, max_bytes=2**31)
        assert not any(i.rule == "too_few_episodes" for i in qa.dataset_issues)

    def test_invalid_args_rejected(self):
        with pytest.raises(RobovlaError) as e:
            qa_dataset(_episode_reports(), total_bytes=-1, max_bytes=2**31)
        assert e.value.code == "validation"
        with pytest.raises(RobovlaError):
            qa_dataset(_episode_reports(), total_bytes=1, max_bytes=0)


class TestSummary:
    def test_summary_counts(self):
        reports = _episode_reports(4)
        reports[1] = _report("ep_0002", "warn", (Issue("sign_flip", "warn", "4 flips"),))
        reports[2] = _report("ep_0003", "fail", (Issue("timing_10hz", "fail", "40% off"),))
        qa = qa_dataset(reports, total_bytes=10, max_bytes=100)
        assert (
            qa.summary.episode_count,
            qa.summary.ok,
            qa.summary.warned,
            qa.summary.failed,
        ) == (4, 2, 1, 1)  # 4 episodes: 2 ok, 1 warn, 1 fail

    def test_qa_is_json_friendly(self):
        import json

        qa = qa_dataset(_episode_reports(3), total_bytes=10, max_bytes=100)
        json.dumps(qa.to_dict())
