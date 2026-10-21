"""episode — the shared single-episode validator (design §11, §5.2).

Vendored verbatim into the capture app and the runner agent so capture-side,
console-side, and agent-side QA can never drift. Pure: operates on parsed
episode data (the file reading is the seam's job).

Episode format (setup-and-materials §6):
  * actions.json: ``[{t, vx, vy, wz}, ...]`` at 10 Hz, velocities normalized
    to [-1, 1] against the autonomy caps;
  * frames/*.jpg at 10 Hz (counted by the seam, passed as ``frame_count``);
  * episode.json: {id, task, prompt, start_pose, success, pilot, notes, ...}.
"""

import pytest

from robovla.core.errors import RobovlaError
from robovla.episode import (
    IDLE_EPS,
    IDLE_MAX_S,
    MAX_JUMPS,
    MAX_SIGN_FLIPS,
    DT_NOMINAL,
    DT_TOL,
    JUMP_TH,
    DEADZONE,
    Action,
    Episode,
    EpisodeReport,
    to_action_7d,
    validate_episode,
)


def _actions(n, dt=0.1, value=0.2):
    return tuple(Action(t=i * dt, vx=value, vy=0.0, wz=0.0) for i in range(n))


def _episode(n=100, **kw):
    kw.setdefault("actions", _actions(n))
    kw.setdefault("frame_count", n)
    kw.setdefault("id", "ep_0042")
    kw.setdefault("task", "stop at the target")
    return Episode(**kw)


class TestClean:
    def test_clean_episode_is_ok(self):
        r = validate_episode(_episode())
        assert r.status == "ok"
        assert r.issues == ()
        assert r.episode_id == "ep_0042"

    def test_bounds_inclusive(self):
        acts = tuple(Action(t=i * 0.1, vx=1.0 if i % 2 else -1.0, vy=0.0, wz=0.0) for i in range(20))
        # full-scale velocities are legal (normalized to [-1, 1])
        r = validate_episode(_episode(actions=acts, frame_count=20))
        assert not any(i.rule == "action_range" for i in r.issues)

    def test_constants_match_spec(self):
        assert DT_NOMINAL == pytest.approx(0.1)
        assert DT_TOL == pytest.approx(0.02)
        assert IDLE_MAX_S == pytest.approx(2.0)
        assert IDLE_EPS == pytest.approx(0.05)
        assert DEADZONE == pytest.approx(0.1)
        assert JUMP_TH == pytest.approx(0.8)
        assert MAX_SIGN_FLIPS == 3
        assert MAX_JUMPS == 3


class TestFormatRules:
    def test_empty_actions_fail(self):
        r = validate_episode(_episode(actions=(), frame_count=0))
        assert r.status == "fail"
        assert any(i.rule == "empty" and i.severity == "fail" for i in r.issues)

    def test_bad_id_fails(self):
        r = validate_episode(_episode(id="episode_1"))
        assert r.status == "fail"
        assert any(i.rule == "format" for i in r.issues)

    def test_frame_action_mismatch_fails(self):
        r = validate_episode(_episode(frame_count=99))
        assert r.status == "fail"
        issue = next(i for i in r.issues if i.rule == "frame_action_mismatch")
        assert issue.severity == "fail"
        assert "100" in issue.detail and "99" in issue.detail

    def test_non_finite_or_missing_action_fields_rejected(self):
        with pytest.raises(RobovlaError) as e:
            validate_episode(_episode(actions=(Action(t=float("nan"), vx=0, vy=0, wz=0),), frame_count=1))
        assert e.value.code == "validation"


class TestTimingRules:
    def test_5hz_timing_fails(self):
        acts = tuple(Action(t=i * 0.2, vx=0.2, vy=0, wz=0) for i in range(50))
        r = validate_episode(_episode(actions=acts, frame_count=50))
        assert any(i.rule == "timing_10hz" and i.severity == "fail" for i in r.issues)

    def test_tolerance_band_accepted(self):
        acts = tuple(Action(t=i * 0.08 if i % 2 else i * 0.12, vx=0.2, vy=0, wz=0) for i in range(40))
        # 0.08/0.12 alternate -> dt oscillates 0.02..0.04? no: t sequence is
        # 0, .12, .16, .28, ...; build explicit within-band dts instead.
        t = 0.0
        out = []
        for i in range(40):
            out.append(Action(t=t, vx=0.2, vy=0, wz=0))
            t += 0.11 if i % 2 else 0.09  # every dt in [0.08, 0.12]
        r = validate_episode(_episode(actions=tuple(out), frame_count=40))
        assert not any(i.rule == "timing_10hz" for i in r.issues)

    def test_minor_timing_noise_within_10_percent_is_ok(self):
        acts = list(_actions(100))
        acts[5] = Action(t=acts[5].t + 0.05, vx=0.2, vy=0, wz=0)  # 1 bad interval of 2 (5%)
        r = validate_episode(_episode(actions=tuple(acts), frame_count=100))
        assert not any(i.rule == "timing_10hz" for i in r.issues)

    def test_widespread_timing_drift_fails(self):
        acts = list(_actions(100))
        for i in range(20):  # 20 of 99 intervals (20%) off the band
            acts[i] = Action(t=i * 0.05, vx=0.2, vy=0, wz=0)
        r = validate_episode(_episode(actions=tuple(acts), frame_count=100))
        assert any(i.rule == "timing_10hz" and i.severity == "fail" for i in r.issues)

    def test_non_monotonic_timestamps_fail(self):
        acts = list(_actions(10))
        acts[5] = Action(t=acts[4].t, vx=0.2, vy=0, wz=0)  # duplicate t
        r = validate_episode(_episode(actions=tuple(acts), frame_count=10))
        assert any(i.rule == "timing_monotonic" and i.severity == "fail" for i in r.issues)


class TestIdlePause:
    def test_idle_over_2s_warns(self):
        acts = list(_actions(50))
        for i in range(20, 45):  # 25 samples (2.5 s) of near-zero
            acts[i] = Action(t=i * 0.1, vx=0.01, vy=-0.01, wz=0.0)
        r = validate_episode(_episode(actions=tuple(acts), frame_count=50))
        assert r.status == "warn"
        assert any(i.rule == "idle_pause" and i.severity == "warn" for i in r.issues)

    def test_short_pause_below_threshold_ok(self):
        acts = list(_actions(50))
        for i in range(20, 39):  # 19 samples (1.9 s)
            acts[i] = Action(t=i * 0.1, vx=0.0, vy=0.0, wz=0.0)
        r = validate_episode(_episode(actions=tuple(acts), frame_count=50))
        assert not any(i.rule == "idle_pause" for i in r.issues)

    def test_idle_uses_deadzone_not_zero(self):
        # "near-zero" means below IDLE_EPS on all three dims
        acts = list(_actions(50))
        for i in range(20, 45):
            acts[i] = Action(t=i * 0.1, vx=IDLE_EPS * 0.5, vy=0.0, wz=0.0)
        r = validate_episode(_episode(actions=tuple(acts), frame_count=50))
        assert any(i.rule == "idle_pause" for i in r.issues)


class TestSignAndJump:
    def _flipping(self, flips, mag=0.5):
        acts = []
        t = 0.0
        sign = 1
        for i in range(60):
            if i in flips:
                sign = -sign
            acts.append(Action(t=t, vx=sign * mag, vy=0.0, wz=0.0))
            t += 0.1
        return tuple(acts)

    def test_many_hard_sign_flips_warn(self):
        r = validate_episode(_episode(actions=self._flipping({10, 20, 30, 40}), frame_count=60))
        assert r.status == "warn"
        assert any(i.rule == "sign_flip" and i.severity == "warn" for i in r.issues)

    def test_few_sign_flips_ok(self):
        r = validate_episode(_episode(actions=self._flipping({10, 20}), frame_count=60))
        assert not any(i.rule == "sign_flip" for i in r.issues)

    def test_flips_through_deadzone_not_counted(self):
        # sign changes while one side is below DEADZONE are not reversals
        acts = []
        t = 0.0
        for i in range(60):
            if i % 2:
                acts.append(Action(t=t, vx=0.05, vy=0, wz=0))
            else:
                acts.append(Action(t=t, vx=-0.05, vy=0, wz=0))
            t += 0.1
        r = validate_episode(_episode(actions=tuple(acts), frame_count=60))
        assert not any(i.rule == "sign_flip" for i in r.issues)

    def test_jumps_warn(self):
        acts = list(_actions(60))
        for i in (10, 20, 30):
            acts[i] = Action(t=i * 0.1, vx=-0.7, vy=0, wz=0)  # 0.9 jump from +0.2
        r = validate_episode(_episode(actions=tuple(acts), frame_count=60))
        assert any(i.rule == "jump_anomaly" and i.severity == "warn" for i in r.issues)

    def test_small_deltas_ok(self):
        acts = tuple(Action(t=i * 0.1, vx=0.2 + 0.001 * (i % 3), vy=0, wz=0) for i in range(60))
        r = validate_episode(_episode(actions=acts, frame_count=60))
        assert not any(i.rule == "jump_anomaly" for i in r.issues)


class TestStatusAndReport:
    def test_status_is_fail_when_any_fail(self):
        r = validate_episode(_episode(frame_count=99, id="bad id"))
        assert r.status == "fail"

    def test_status_warn_beats_ok(self):
        acts = list(_actions(50))
        for i in range(20, 45):
            acts[i] = Action(t=i * 0.1, vx=0.0, vy=0.0, wz=0.0)
        r = validate_episode(_episode(actions=tuple(acts), frame_count=50))
        assert r.status == "warn"

    def test_report_is_frozen(self):
        r = validate_episode(_episode())
        with pytest.raises(Exception):
            r.episode_id = "x"

    def test_issue_carries_rule_severity_detail(self):
        r = validate_episode(_episode(frame_count=99))
        issue = r.issues[0]
        assert issue.rule and issue.severity in ("fail", "warn") and issue.detail


class TestAction7d:
    def test_maps_to_seven_dim_convention(self):
        a = Action(t=0.0, vx=0.1, vy=-0.2, wz=0.3)
        assert to_action_7d(a) == (0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0)

    def test_episode_actions_feed_norm_stats(self):
        # cross-module: the (vx, vy, wz, 0, 0, 0, 0) convention (setup §6)
        # gives degenerate dims 4-7 in the dataset norm stats.
        from robovla.core.norm_stats import compute_norm_stats, denormalize, normalize

        ep = _episode(50)
        stats = compute_norm_stats([to_action_7d(a) for a in ep.actions])
        assert stats.min[3:] == (0.0, 0.0, 0.0, 0.0)
        a7 = to_action_7d(ep.actions[0])
        assert denormalize(normalize(a7, stats), stats) == pytest.approx(a7)
