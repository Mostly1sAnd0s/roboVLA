"""Browser robot operation over the telemetry channel: config, teleop, stop,
record. One round trip both ways — the browser writes state, the robot reads it
on its next poll, and the stop ladder outranks both teleop and the model.
"""

from __future__ import annotations

import base64


def _token(env, uid):
    """A Pi token for this rig, cached on the fixture: login is rate-limited
    to 5/min per host, so a test that drives two robots logs in twice."""
    cache = env.__dict__.setdefault("_pi_tokens", {})
    if uid not in cache:
        env.as_student(uid)
        cache[uid] = {"Authorization": "Bearer " + env.client.post(
            "/api/auth/token", json={"action": "generate"}).json()["token"]}
    return cache[uid]


def _open_teleop(env, uid="s01", **extra):
    """Turn on the teacher's teleop switch for this robot; returns the config."""
    env.as_student(uid)
    cfg = env.client.get("/api/robot/config").json()["config"]
    cfg["teleop"]["enabled"] = True
    cfg.update(extra)
    r = env.client.post("/api/robot/config", json={"config": cfg})
    assert r.status_code == 200, r.text
    return r.json()["config"]


def _tele(env, auth, **over):
    body = {"image": base64.b64encode(b"\xff\xd8frame").decode(),
            "action_robot": [0.1, 0, 0, 0, 0, 0, 0], "latency_ms": 9.0,
            "prompt": "p", **over}
    return env.client.post("/api/robot/telemetry", headers=auth, json=body)


class TestRobotConfig:
    def test_defaults_then_round_trip(self, env):
        env.as_student("s01")
        got = env.client.get("/api/robot/config").json()["config"]
        assert got["ros_domain_id"] == 99 and got["teleop"]["enabled"] is False
        got["rate_hz"], got["telemetry_rate_hz"] = 2.0, 8.0
        saved = env.client.post("/api/robot/config",
                                json={"config": got}).json()["config"]
        assert (saved["rate_hz"], saved["telemetry_rate_hz"]) == (2.0, 8.0)

    def test_values_are_clamped_on_the_way_in(self, env):
        env.as_student("s01")
        cfg = env.client.post("/api/robot/config",
                              json={"config": {"ros_domain_id": 9999}}).json()["config"]
        assert cfg["ros_domain_id"] == 232

    def test_a_peer_cannot_edit_another_robot(self, env):
        env.as_student("s07")
        env.client.post("/api/robot/config", json={"config": {"rate_hz": 3.0}})
        env.as_student("s01")
        # refused out loud: silently editing one's own robot instead would let a
        # mis-targeted request look like it worked
        assert env.client.post("/api/robot/config",
                               json={"student": "s07",
                                     "config": {"rate_hz": 9.0}}).status_code == 403
        assert env.client.get("/api/robot/config?student=s07").status_code == 403
        env.as_student("s07")
        assert env.client.get("/api/robot/config").json()["config"]["rate_hz"] == 3.0

    def test_config_reaches_the_robot_on_its_next_poll(self, env):
        cfg = _open_teleop(env, "s01", telemetry_rate_hz=7.5)
        auth = _token(env, "s01")
        reply = _tele(env, auth).json()
        assert reply["config"]["telemetry_rate_hz"] == 7.5
        assert reply["config"]["deadman_s"] == cfg["teleop"]["deadman_s"]


class TestTeleop:
    def test_needs_the_teacher_switch_first(self, env):
        env.as_student("s01")
        r = env.client.post("/api/robot/teleop", json={"vx": 0.1})
        assert r.status_code == 403 and "teleop" in r.text

    def test_command_rides_the_next_telemetry_reply(self, env):
        _open_teleop(env)
        auth = _token(env, "s01")
        out = env.client.post("/api/robot/teleop",
                              json={"vx": 0.2, "vy": 0.0, "wz": -0.4}).json()
        assert out["mode"] == "teleop" and out["cmd_vel"] == [0.2, 0.0, -0.4]
        ctl = _tele(env, auth).json()["control"]
        assert ctl["mode"] == "teleop" and ctl["cmd_vel"] == [0.2, 0.0, -0.4]
        assert ctl["expires_in_s"] > 0

    def test_commands_are_clamped_not_passed_through(self, env):
        _open_teleop(env)
        out = env.client.post("/api/robot/teleop",
                              json={"vx": 50, "vy": -50, "wz": 50}).json()
        assert out["cmd_vel"] == [0.3, -0.1, 0.9]

    def test_expiry_falls_back_to_model(self, env):
        _open_teleop(env)
        auth = _token(env, "s01")
        env.client.post("/api/robot/teleop", json={"vx": 0.3, "hold_s": 0.5})
        env.svc.clock = lambda: "2026-10-21T09:00:30Z"
        assert _tele(env, auth).json()["control"]["mode"] == "model"

    def test_works_with_no_gpu_slot_claimed(self, env):
        """A human driving needs no model: teleop opens without an infer slot,
        and predict still refuses one."""
        _open_teleop(env)
        auth = _token(env, "s01")
        assert _tele(env, auth).status_code == 200
        r = env.client.post("/api/robot/predict", headers=auth,
                            json={"image": "x", "prompt": "p"})
        assert r.status_code == 403

    def test_locked_when_the_switch_is_off(self, env):
        env.as_student("s01")
        auth = _token(env, "s01")
        assert _tele(env, auth).status_code == 403      # telemetry needs a slot


class TestStop:
    def test_stop_outranks_a_live_teleop_command(self, env):
        _open_teleop(env)
        auth = _token(env, "s01")
        env.client.post("/api/robot/teleop", json={"vx": 0.3, "hold_s": 2.0})
        r = env.client.post("/api/robot/stop", json={"reason": "somebody in the way"})
        assert r.status_code == 200 and r.json()["stopped"] == ["s01"]
        ctl = _tele(env, auth).json()["control"]
        assert ctl["mode"] == "stop" and ctl["cmd_vel"] == [0.0, 0.0, 0.0]
        assert ctl["stop"]["reason"] == "somebody in the way"

    def test_stop_stays_latched_until_cleared(self, env):
        _open_teleop(env)
        auth = _token(env, "s01")
        env.client.post("/api/robot/stop", json={})
        for _ in range(3):
            assert _tele(env, auth).json()["control"]["mode"] == "stop"
        env.client.post("/api/robot/stop", json={"clear": True})
        assert _tele(env, auth).json()["control"]["mode"] == "model"

    def test_a_peer_cannot_stop_or_clear_another_robot(self, env):
        _open_teleop(env, "s07")
        env.as_student("s01")
        assert env.client.post("/api/robot/stop",
                               json={"student": "s07"}).status_code == 403
        assert env.svc.storage.get_kv("robot_ctl:s07") is None

    def test_teacher_broadcast_stops_every_robot_that_is_driving(self, env):
        for uid in ("s01", "s07"):
            _open_teleop(env, uid)
            _tele(env, _token(env, uid))
        env.as_teacher()
        r = env.client.post("/api/robot/stop", json={"broadcast": True})
        assert r.json()["stopped"] == ["s01", "s07"]
        for uid in ("s01", "s07"):
            auth = _token(env, uid)
            assert _tele(env, auth).json()["control"]["mode"] == "stop"

    def test_only_a_teacher_may_broadcast(self, env):
        env.as_student("s01")
        assert env.client.post("/api/robot/stop",
                              json={"broadcast": True}).status_code == 403

    def test_broadcast_with_nothing_driving_is_not_an_error(self, env):
        env.as_teacher()
        r = env.client.post("/api/robot/stop", json={"broadcast": True})
        assert r.status_code == 200 and r.json()["stopped"] == []


class TestRecord:
    def test_start_is_repeated_until_the_robot_acks_it(self, env):
        _open_teleop(env)
        auth = _token(env, "s01")
        seq = env.client.post("/api/robot/record",
                              json={"action": "start",
                                    "task": "stop-at-target"}).json()["seq"]
        assert seq == 1
        ctl = _tele(env, auth).json()["control"]
        assert ctl["record"]["seq"] == 1 and ctl["record"]["action"] == "start"
        # the same reply arrives again after the node acted on it: retired
        reply = _tele(env, auth, record_ack=seq).json()["control"]
        assert reply["record"] is None
        st = env.client.get("/api/robot/status").json()
        assert st["record_acked"] == 1

    def test_stop_record_command_and_task_validation(self, env):
        _open_teleop(env)
        env.client.post("/api/robot/record", json={"action": "start", "task": "a"})
        r = env.client.post("/api/robot/record", json={"action": "stop"})
        assert r.json()["seq"] == 2 and r.json()["action"] == "stop"
        assert env.client.post("/api/robot/record",
                              json={"action": "pause"}).status_code == 400

    def test_recording_can_be_disabled_per_robot(self, env):
        cfg = _open_teleop(env)
        cfg["record"]["enabled"] = False
        env.client.post("/api/robot/config", json={"config": cfg})
        assert env.client.post("/api/robot/record",
                              json={"action": "start"}).status_code == 403


class TestStatus:
    def test_status_bundle_for_the_drive_page(self, env):
        _open_teleop(env)
        _tele(env, _token(env, "s01"), state={"battery": 0.81, "odom": [1, 2, 3]})
        st = env.client.get("/api/robot/status").json()
        assert st["student"] == "s01" and st["age_s"] is not None
        assert st["view"]["state"]["battery"] == 0.81
        assert st["control"]["mode"] in ("model", "teleop", "stop")
        assert st["config"]["teleop"]["enabled"] is True

    def test_view_reports_the_latched_stop(self, env):
        _open_teleop(env)
        auth = _token(env, "s01")
        _tele(env, auth)
        env.client.post("/api/robot/stop", json={"reason": "walkway"})
        v = env.client.get("/api/infer/robot-view").json()["views"]["s01"]
        assert v["control_mode"] == "stop" and v["stop"]["reason"] == "walkway"
