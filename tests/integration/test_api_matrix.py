"""§11 test-matrix rows, part 2: the HTTP surface against fakes.

Covers: robot predict (token auth, lease expiry, ~5 req/s), slot claim
202-queue when infer slots are full, agent bundle pull (the one
non-cookie route), RBAC 403 for students on /api/admin, an audit entry
per admin action, rate limits (login/submit/admin), maintenance mode
read-only, node probe + agent-scoped VRAM purge + orphan alert, SSE
Last-Event-ID resume, and VRAM budget admission on a 2-slot node
(two full-batch no / two batch-1 yes on 32 GB).
"""

from __future__ import annotations

import asyncio
import base64
import time

import conftest as ic  # tests/integration/conftest.py (Env, FLEET, NODE)
from robovla.core.config import parse_fleet
from robovla.http.auth import hash_password, token_lookup_key


def _drive_to_done(env, jid: str, steps: int = 2) -> None:
    env.runner.script(jid, steps=steps,
                      artifacts=[{"kind": "adapter",
                                  "path": f"artifacts/{jid}/adapter.safetensors",
                                  "nbytes": 9}])
    row = None
    for _ in range(10):
        env.tick()
        row = env.svc.storage.get_job(jid)
        if row["state"] in ("done", "failed", "killed"):
            break
    assert row is not None and row["state"] == "done", row


def _smoke(env, uid: str, ds: str, batch: int = 1) -> str:
    r = env.client.post("/api/runs", json={"dataset_id": ds, "job_type": "smoke",
                                           "config": {"batch": batch}})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def _train(env, uid: str, ds: str, ticket: str, batch: int) -> dict:
    r = env.client.post("/api/runs", json={"dataset_id": ds, "ticket_id": ticket,
                                           "job_type": "train",
                                           "config": {"batch": batch}})
    assert r.status_code == 200, r.text
    return r.json()


# -- robot path (§9.5): Bearer token, lease, rate -------------------------------------


def test_robot_predict_token_auth_lease_and_rate(env):
    env.as_student("s01")
    ds = env.upload_ready("s01")
    jid = _smoke(env, "s01", ds)
    _drive_to_done(env, jid)
    r = env.client.post("/api/infer/slots/claim", json={"run_id": jid})
    assert r.status_code == 200, r.text
    slot = r.json()
    assert slot["slot"].startswith("slot-") and slot["lease_s"] == 300

    # no token / bad token -> auth (401 shape)
    r = env.client.post("/api/robot/predict", json={"image": "aGk=", "prompt": "p"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "auth"
    r = env.client.post("/api/robot/predict",
                        headers={"Authorization": "Bearer pi_wrong"},
                        json={"image": "aGk=", "prompt": "p"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "auth"

    # own token (shown once) -> 7-dim un-normalized action
    r = env.client.post("/api/auth/token", json={"action": "generate"})
    assert r.status_code == 200 and r.json()["shown_once"] is True
    tok = r.json()["token"]
    auth = {"Authorization": f"Bearer {tok}"}
    r = env.client.post("/api/robot/predict", headers=auth,
                        json={"image": base64.b64encode(b"\x89PNG..").decode(),
                              "prompt": "stop at target"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["action_norm"]) == 7 and len(body["action_robot"]) == 7
    assert body["action_robot"] != body["action_norm"]  # denormalized by stats
    assert body["latency_ms"] == 180.0

    # lease: idle > 300 s releases the slot (conflict), then no slot (forbidden)
    env.svc.storage.set_job(slot["job_id"], last_used_at="2026-10-20T09:00:00Z")
    r = env.client.post("/api/robot/predict", headers=auth,
                        json={"image": "aGk=", "prompt": "p"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    assert env.svc.storage.get_job(slot["job_id"])["state"] == "killed"
    r = env.client.post("/api/robot/predict", headers=auth,
                        json={"image": "aGk=", "prompt": "p"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"

    # ~5 req/s per student: re-claim a slot, let the 1 s window clear, burst
    time.sleep(1.05)
    r = env.client.post("/api/infer/slots/claim", json={"run_id": jid})
    assert r.status_code == 200 and r.json().get("slot"), r.text
    codes = [env.client.post("/api/robot/predict", headers=auth,
                             json={"image": "aGk=", "prompt": "p"}).status_code
             for _ in range(6)]
    assert codes == [200, 200, 200, 200, 200, 429]
    assert env.client.post("/api/robot/predict", headers=auth,
                           json={"image": "aGk=", "prompt": "p"}).json()["error"]["code"] == "rate_limited"


def test_slot_claim_queues_when_infer_slots_full(env):
    """2 infer slots on testnode: the 3rd claim is 202-queued; releasing
    one slot admits the queued claim on the next tick."""
    env.as_student("s01")
    ds1 = env.upload_ready("s01")
    j1 = _smoke(env, "s01", ds1)
    _drive_to_done(env, j1)
    r1 = env.client.post("/api/infer/slots/claim", json={"run_id": j1}).json()

    env.as_student("s07")
    ds2 = env.upload_ready("s07")
    j2 = _smoke(env, "s07", ds2)
    _drive_to_done(env, j2)
    r2 = env.client.post("/api/infer/slots/claim", json={"run_id": j2}).json()
    assert r1["slot"] and r2["slot"]  # both served immediately

    # a third student (seeded directly; login needs a known hash)
    env.svc.storage.create_user("s09", "student", hash_password("stu9123"),
                                display_name="S9", active=True, at=env.svc.clock())
    r = env.client.post("/api/auth/login", json={"username": "s09", "password": "stu9123"})
    assert r.status_code == 200, r.text
    ds3 = env.upload_ready("s09")
    j3 = _smoke(env, "s09", ds3)
    _drive_to_done(env, j3)
    r3 = env.client.post("/api/infer/slots/claim", json={"run_id": j3})
    assert r3.status_code == 200, r.text
    q = r3.json()
    assert q["queued"] is True and q["position"] == 1

    board = env.client.get("/api/infer/slots").json()["slots"]
    assert {s["state"] for s in board} == {"running", "queued"}

    # release s01's slot -> the queued claim is served on the next tick
    env.as_student("s01")
    r = env.client.post(f"/api/infer/slots/{r1['job_id']}/release")
    assert r.status_code == 200 and r.json()["state"] == "killed"
    env.tick()
    assert env.svc.storage.get_job(q["job_id"])["state"] == "running"


# -- agent bundle pull (§9.9: the one non-cookie route) -------------------------------


def test_bundle_pull_requires_agent_token(env):
    env.as_student("s01")
    ds = env.upload_ready("s01")
    env.svc.storage.set_agent_token(
        ic.NODE, hash_password("agent-secret"), token_lookup_key("agent-secret"),
        at=env.svc.clock())
    # wrong / missing token -> auth
    r = env.client.get(f"/api/agent/datasets/{ds}/bundle",
                       headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "auth"
    r = env.client.get(f"/api/agent/datasets/{ds}/bundle")
    assert r.status_code == 401

    r = env.client.get(f"/api/agent/datasets/{ds}/bundle",
                       headers={"Authorization": "Bearer agent-secret"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/gzip"
    assert r.headers["x-dataset-sha256"] and len(r.headers["x-dataset-sha256"]) == 64
    assert r.content[:2] == b"\x1f\x8b"
    pulls = [a for a in env.svc.storage.audit_entries() if a["action"] == "dataset.pull"]
    assert pulls and pulls[-1]["target"] == ds

    # not-ready dataset -> dataset_not_ready (409)
    env.svc.storage.create_dataset("s01/pending/v1", "s01", "uploading", at=env.svc.clock())
    r = env.client.get("/api/agent/datasets/s01/pending/v1/bundle",
                       headers={"Authorization": "Bearer agent-secret"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "dataset_not_ready"


# -- RBAC + audit (§9.7: one RBAC dependency; every admin action audited) ------------


def test_students_forbidden_on_admin_routes(env):
    env.as_student("s01")
    paths = ["/api/admin/nodes", "/api/admin/jobs", "/api/admin/audit",
             "/api/admin/config", "/api/admin/users", "/api/admin/backups"]
    for p in paths:
        r = env.client.get(p)
        assert r.status_code == 403, (p, r.status_code)
        assert r.json()["error"]["code"] == "forbidden"
    r = env.client.post("/api/admin/maintenance", json={"on": True})
    assert r.status_code == 403
    env.as_teacher()
    assert env.client.get("/api/admin/nodes").status_code == 200


def test_admin_actions_are_audited(env):
    env.as_student("s01")
    ds = env.upload_ready("s01")
    jid = _smoke(env, "s01", ds)
    env.as_teacher()
    assert env.client.post("/api/admin/maintenance", json={"on": True}).status_code == 200
    assert env.client.post(f"/api/admin/datasets/{ds}/reject",
                           json={"reason": "bad capture"}).status_code == 200
    assert env.client.post(f"/api/admin/jobs/{jid}/note",
                           json={"note": "manual check"}).status_code == 200
    r = env.client.post("/api/admin/users/s07/reset-password")
    assert r.status_code == 200 and r.json()["shown_once"] is True
    assert env.client.post(f"/api/admin/nodes/{ic.NODE}/probe").status_code == 200
    assert env.client.post("/api/admin/maintenance", json={"on": False}).status_code == 200

    entries = env.client.get("/api/admin/audit").json()["entries"]
    actions = {e["action"] for e in entries}
    assert {"system.maintenance", "dataset.reject", "job.note",
            "user.reset_password", "node.probe"} <= actions
    assert all(e["actor"] in ("teacher", "s01") for e in entries)
    # since= filter works (RFC 3339)
    old = env.client.get("/api/admin/audit",
                         params={"since": "2999-01-01T00:00:00Z"}).json()["entries"]
    assert old == []


# -- rate limits (§9.8): login 5/min/IP, submit 10/min/student, admin 60/min ---------


def test_rate_limits_login_submit_admin(env):
    # submit: 10/min per student (the rate check precedes the queue caps, so
    # calls 3-10 are 409 limit_exceeded and the 11th is 429)
    env.as_student("s01")  # login #1 of the 5/min window
    ds = env.upload_ready("s01")
    codes = []
    for _ in range(11):
        r = env.client.post("/api/runs", json={"dataset_id": ds, "job_type": "smoke"})
        codes.append(r.status_code)
    assert codes[:2] == [200, 200]
    assert set(codes[2:10]) == {409}
    assert codes[10] == 429

    # login: 5/min per client IP — 4 more allowed (1 used above), then 429
    for _ in range(4):
        r = env.client.post("/api/auth/login",
                            json={"username": "s01", "password": "stu1123"})
        assert r.status_code == 200, r.text
    r = env.client.post("/api/auth/login", json={"username": "s01", "password": "stu1123"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"


def test_rate_limit_admin_60_per_min(env):
    env.as_teacher()
    for i in range(60):
        r = env.client.get("/api/admin/nodes")
        assert r.status_code == 200, (i, r.text)
    r = env.client.get("/api/admin/nodes")
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"


# -- maintenance mode (§9.7): console read-only for students ---------------------------


def test_maintenance_mode_read_only(env):
    env.as_teacher()
    assert env.client.post("/api/admin/maintenance", json={"on": True}).json() == {"maintenance": True}
    env.as_student("s01")
    # writes are refused with conflict; reads still work
    r = env.client.post("/api/datasets/upload",
                        data={"dataset_id": "s01/x/v1", "total_chunks": "1",
                              "chunk_index": "0", "sha256": "0" * 64},
                        files={"file": ("c0.bin", b"abc", "application/octet-stream")})
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    r = env.client.post("/api/runs", json={"dataset_id": "s01/x/v1", "job_type": "smoke"})
    assert r.status_code == 409
    r = env.client.post("/api/infer/slots/claim", json={"run_id": "j_x"})
    assert r.status_code == 409
    assert env.client.get("/api/datasets").status_code == 200
    assert env.client.get("/api/runs").status_code == 200
    # admin is not blocked by its own maintenance switch
    env.as_teacher()
    assert env.client.get("/api/admin/nodes").status_code == 200


# -- node ops (§8.2): join probe, agent-scoped purge, orphan alert ----------------------


def test_node_probe_and_agent_scoped_purge(env):
    env.runner.set_vram(30.0, procs=[
        {"pid": 4242, "name": "python", "vram_gb": 8.0, "agent_spawned": True},
        {"pid": 11, "name": "ninfer-serve", "vram_gb": 10.0, "agent_spawned": False},
    ])
    asyncio.run(env.svc.refresh_nodes())
    env.as_teacher()
    # probe: agent healthy, image matches the fleet entry
    r = env.client.post(f"/api/admin/nodes/{ic.NODE}/probe")
    assert r.status_code == 200 and r.json()["ok"] is True, r.text

    # agent-scoped purge only; host procs are forbidden
    r = env.client.post(f"/api/admin/nodes/{ic.NODE}/purge-vram", json={"pids": [4242]})
    assert r.status_code == 200 and r.json()["purged"] == [4242]
    assert env.runner.purged_pids == [4242]
    r = env.client.post(f"/api/admin/nodes/{ic.NODE}/purge-vram", json={"pids": [11]})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    r = env.client.post(f"/api/admin/nodes/{ic.NODE}/purge-vram", json={"pids": [777]})
    assert r.status_code == 403

    # the node view flags the agent-spawned proc as an orphan (no running job)
    nodes = env.client.get("/api/admin/nodes").json()["nodes"]
    assert nodes[0]["orphan_alert"] is True and 4242 in nodes[0]["orphan_pids"]
    assert env.client.get(f"/api/admin/nodes/{ic.NODE}/logs").status_code == 200


# -- SSE resume (§9.4): Last-Event-ID replays only what was missed ----------------------


def test_sse_resume_via_last_event_id(env):
    env.as_student("s01")
    ds = env.upload_ready("s01")
    jid = _smoke(env, "s01", ds)
    _drive_to_done(env, jid, steps=3)
    hist = env.svc.bus.replay(jid, 0)
    assert hist and hist[-1][1] == "state" and hist[-1][2]["state"] == "done"
    last = hist[-1][0]

    seen: list[int] = []
    with env.client.stream("GET", f"/stream/{jid}",
                           headers={"Last-Event-ID": str(last - 2)}) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("id: "):
                seen.append(int(line[4:]))
            if len(seen) == 2:
                break
    # exactly the two missed events — nothing older, nothing extra
    assert seen == [last - 1, last]


# -- VRAM budget admission (§3): 2 slots on 32 GB ---------------------------------------


FLEET_2SLOT = ic.FLEET.replace("train_slots: 1", "train_slots: 2")


def _two_train_slots(env):
    env.svc.fleet = parse_fleet(FLEET_2SLOT)
    env.svc._scheduler = env.svc._build_scheduler()
    asyncio.run(env.svc.refresh_nodes())


def test_vram_budget_admission_two_slots(env):
    """batch-8 train costs the full 18 GB: two don't fit (30 GB free);
    batch-1 scales to ~11 GB: two do. Slot count is not the constraint.
    The budget guard works on the MEASURED probe, so the fake reports the
    VRAM the running job occupies (as a real agent probe would)."""
    _two_train_slots(env)
    env.as_student("s01")
    ds1 = env.upload_ready("s01")
    t1 = env.make_ticket("s01")
    r1 = _train(env, "s01", ds1, t1, batch=8)
    j1 = r1["job_id"]
    env.tick()
    assert env.svc.storage.get_job(j1)["state"] == "running"
    # the probe now reports 30 - 18 GB free
    env.runner.set_vram(12.0)
    asyncio.run(env.svc.refresh_nodes())

    env.as_student("s07")
    ds2 = env.upload_ready("s07")
    t2 = env.make_ticket("s07")
    r2 = _train(env, "s07", ds2, t2, batch=8)
    j2 = r2["job_id"]
    assert r2.get("waiting") == "slot"
    assert {"node": ic.NODE, "reason": "budget"} in r2["blockers"]
    env.tick()
    assert env.svc.storage.get_job(j2)["state"] == "queued"

    # free both, then two batch-1 jobs (each ~11 GB) are both admissible
    env.as_teacher()
    for j in (j1, j2):
        r = env.client.post(f"/api/admin/jobs/{j}/kill")
        assert r.status_code == 200, r.text
    env.runner.set_vram(30.0)
    asyncio.run(env.svc.refresh_nodes())
    env.as_student("s01")
    t3 = env.make_ticket("s01")
    j3 = _train(env, "s01", ds1, t3, batch=1)["job_id"]
    env.as_student("s07")
    t4 = env.make_ticket("s07")
    j4 = _train(env, "s07", ds2, t4, batch=1)["job_id"]
    for _ in range(4):
        env.tick()
    assert env.svc.storage.get_job(j3)["state"] == "running"
    assert env.svc.storage.get_job(j4)["state"] == "running"
