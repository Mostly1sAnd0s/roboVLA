"""§9.10 robot view + live prompt: telemetry gate, ring view, RBAC,
prompt override round trip (telemetry reply carries the override)."""

from __future__ import annotations

import base64

import test_api_matrix as m  # reuse _smoke/_drive_to_done on this rig


def _armed(env, uid="s01"):
    """student with a done run, a claimed slot, and a fresh pi token."""
    env.as_student(uid)
    ds = env.upload_ready(uid)
    jid = m._smoke(env, uid, ds)
    m._drive_to_done(env, jid)
    slot = env.client.post("/api/infer/slots/claim",
                           json={"run_id": jid}).json()
    tok = env.client.post("/api/auth/token",
                          json={"action": "generate"}).json()["token"]
    return slot, {"Authorization": f"Bearer {tok}"}


def _tele(env, auth, **over):
    body = {"image": base64.b64encode(b"\xff\xd8frame").decode(),
            "action_robot": [0.1, -0.02, 0, 0, 0, 0, 0.05],
            "latency_ms": 12.5, "prompt": "drive to the red target", **over}
    return env.client.post("/api/robot/telemetry", headers=auth, json=body)


def test_telemetry_round_trip_and_view(env):
    slot, auth = _armed(env)
    r = _tele(env, auth)
    assert r.status_code == 200 and r.json()["ok"], r.text
    assert r.json()["prompt"] == ""          # no override set yet

    v = env.client.get("/api/infer/robot-view").json()["views"]["s01"]
    assert base64.b64decode(v["frame_b64"]) == b"\xff\xd8frame"
    assert v["action"][0] == 0.1 and v["latency_ms"] == 12.5
    assert v["slot_job"] == slot["job_id"] and v["history"]

    # telemetry refreshes the lease (a driving robot keeps its slot)
    row = env.svc.storage.get_job(slot["job_id"])
    assert row["state"] == "running" and row["last_used_at"]


def test_telemetry_requires_active_slot(env):
    env.as_student("s01")
    tok = env.client.post("/api/auth/token",
                          json={"action": "generate"}).json()["token"]
    r = _tele(env, {"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403
    r = env.client.post("/api/robot/telemetry", json={})  # no bearer
    assert r.status_code == 401


def test_prompt_override_reaches_the_robot(env):
    slot, auth = _armed(env)
    # teacher sets the prompt on s01's slot
    env.as_teacher()
    r = env.client.post(f"/api/infer/slots/{slot['job_id']}/prompt",
                        json={"prompt": "go to the red circle"})
    assert r.status_code == 200 and r.json()["prompt"] == "go to the red circle"

    # the robot's next telemetry reply carries it (one round trip, §9.10)
    env.as_student("s01")
    r = _tele(env, auth)
    assert r.json()["prompt"] == "go to the red circle"

    # an empty override means "keep your own"
    env.as_teacher()
    env.client.post(f"/api/infer/slots/{slot['job_id']}/prompt",
                    json={"prompt": ""})
    env.as_student("s01")
    r = env.client.post("/api/robot/telemetry", headers=auth, json={
        "image": base64.b64encode(b"x").decode(), "action_robot": [0] * 7,
        "latency_ms": 1, "prompt": "p"})
    assert r.json()["prompt"] == ""


def test_view_scoping_students_and_teacher(env):
    slot, auth = _armed(env)
    _tele(env, auth)
    # s07 cannot see s01's robot
    env.as_student("s07")
    assert env.client.get("/api/infer/robot-view").json()["views"] == {}
    # and cannot set s01's prompt
    r = env.client.post(f"/api/infer/slots/{slot['job_id']}/prompt",
                        json={"prompt": "hijack"})
    assert r.status_code == 403
    # teacher sees everything
    env.as_teacher()
    views = env.client.get("/api/infer/robot-view").json()["views"]
    assert "s01" in views


def test_oversized_frame_rejected(env):
    slot, auth = _armed(env)
    big = base64.b64encode(b"0" * (3 * 1024 * 1024)).decode()
    r = _tele(env, auth, image=big)
    assert r.status_code == 400 and "too large" in r.text
