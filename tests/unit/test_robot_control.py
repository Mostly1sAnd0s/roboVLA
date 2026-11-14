"""The robot command ladder and robot config (core.robot_control).

The ladder is the safety story: stop beats teleop beats model, and an expired
command is the same as no command. Pure functions, so the rules are testable
without a robot, a socket, or a clock.
"""
import pytest

from robovla.core.robot_control import (DEFAULT_CONFIG, TELEOP_LIMITS,
                                        clamp_cmd, control_for, merge_config)

NOW = "2026-10-21T09:00:00Z"


def _later(seconds: float) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)
            + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class TestMergeConfig:
    def test_empty_stored_gives_defaults(self):
        assert merge_config(None) == DEFAULT_CONFIG

    def test_out_of_range_values_are_clamped_not_trusted(self):
        cfg = merge_config({"ros_domain_id": 9999, "rate_hz": 500,
                            "telemetry_rate_hz": 0.01, "frame_size": 4000})
        assert cfg["ros_domain_id"] == 232
        assert cfg["rate_hz"] == 10.0
        assert cfg["telemetry_rate_hz"] == 0.5
        assert cfg["frame_size"] == 640

    def test_unknown_keys_are_dropped(self):
        cfg = merge_config({"rm_rf": "now", "rate_hz": 2.0})
        assert "rm_rf" not in cfg
        assert cfg["rate_hz"] == 2.0

    def test_junk_types_fall_back(self):
        cfg = merge_config({"rate_hz": "fast", "ros_domain_id": None})
        assert cfg["rate_hz"] == DEFAULT_CONFIG["rate_hz"]
        assert cfg["ros_domain_id"] == DEFAULT_CONFIG["ros_domain_id"]

    def test_prompt_is_length_capped(self):
        assert len(merge_config({"prompt": "x" * 500})["prompt"]) == 200

    def test_nested_blocks_survive_partial_patches(self):
        cfg = merge_config({"teleop": {"enabled": True}})
        assert cfg["teleop"]["enabled"] is True
        assert cfg["teleop"]["deadman_s"] == DEFAULT_CONFIG["teleop"]["deadman_s"]
        assert cfg["record"]["enabled"] is True


class TestClampCmd:
    def test_clamps_to_classroom_limits(self):
        assert clamp_cmd(99, -99, 99) == [TELEOP_LIMITS["vx"][1],
                                          TELEOP_LIMITS["vy"][0],
                                          TELEOP_LIMITS["wz"][1]]

    def test_junk_becomes_zero(self):
        assert clamp_cmd("x", None, [1]) == [0.0, 0.0, 0.0]

    def test_passes_a_normal_command_through(self):
        assert clamp_cmd(0.2, 0.0, -0.5) == [0.2, 0.0, -0.5]


class TestControlFor:
    def test_nothing_stored_means_model(self):
        assert control_for(None, NOW)["mode"] == "model"
        assert control_for({}, NOW)["cmd_vel"] is None

    def test_live_teleop_wins_over_model(self):
        ctl = {"teleop": {"vx": 0.1, "vy": 0.0, "wz": 0.0,
                          "expires_at": _later(0.5)}}
        out = control_for(ctl, NOW)
        assert out["mode"] == "teleop" and out["cmd_vel"] == [0.1, 0.0, 0.0]
        assert out["expires_in_s"] == pytest.approx(0.5)

    def test_expired_teleop_is_no_command(self):
        ctl = {"teleop": {"vx": 0.3, "vy": 0.0, "wz": 0.0,
                          "expires_at": _later(-0.1)}}
        assert control_for(ctl, NOW)["mode"] == "model"

    def test_stop_outranks_a_live_teleop(self):
        ctl = {"teleop": {"vx": 0.3, "vy": 0.0, "wz": 0.0, "expires_at": _later(5)},
               "stop": {"by": "teacher", "at": NOW, "reason": "walkway"}}
        out = control_for(ctl, NOW)
        assert out["mode"] == "stop"
        assert out["cmd_vel"] == [0.0, 0.0, 0.0]      # zero, not "unchanged"
        assert out["stop"]["by"] == "teacher"

    def test_stop_reports_who_and_why(self):
        out = control_for({"stop": {"by": "s01", "at": NOW}}, NOW)
        assert out["stop"]["reason"] == ""
        assert out["stop"]["by"] == "s01"

    def test_garbage_expiry_is_treated_as_expired(self):
        assert control_for({"teleop": {"vx": 1, "expires_at": "not a date"}},
                           NOW)["mode"] == "model"

    def test_record_command_rides_whichever_mode(self):
        rec = {"seq": 3, "action": "start", "task": "stop-at-target"}
        assert control_for({"record": rec}, NOW)["record"] == rec
        assert control_for({"record": rec, "stop": {"by": "t"}}, NOW)["record"] == rec

    def test_malformed_rows_never_raise_on_the_hot_path(self):
        # the robot polls this every tick; a bad row must degrade, not 500
        assert control_for({"teleop": {}}, NOW)["mode"] == "model"
        assert control_for({"stop": None, "record": None}, NOW)["mode"] == "model"
        assert control_for({"teleop": {"expires_at": None}}, NOW)["mode"] == "model"
