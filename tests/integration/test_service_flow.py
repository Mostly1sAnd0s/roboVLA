"""Integration: the full CPU service flow (design §11 test matrix).

upload -> prep -> QA gate -> ticket -> submit -> queue admission ->
SSE state/metrics/val -> artifacts -> ticket link released at terminal.
Plus: idempotency dedup, run-now ordering, and the error-shape contract.
"""

from __future__ import annotations

import time


def test_health_public(env):
    r = env.client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"]["up"] is True
    node = body["nodes"][0]
    assert node["id"] == "testnode" and node["enabled"] is True


def test_login_and_me(env):
    r = env.client.post("/api/auth/login",
                        json={"username": "s01", "password": "wrong"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "auth"
    env.as_student("s01")
    r = env.client.get("/api/auth/me")
    assert r.status_code == 200
    me = r.json()
    assert me["user_id"] == "s01" and me["role"] == "student"
    assert me["limits"]["max_queued"] == 2


def test_upload_prep_qa_gate_and_norm_stats(env):
    env.as_student("s01")
    ds = env.upload_ready("s01", ep_count=101)
    r = env.client.get(f"/api/datasets/{ds}")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["qa"]["summary"]["episode_count"] == 101
    r = env.client.get(f"/api/datasets/{ds}/qa")
    assert r.json()["qa"]["status"] == "ready"
    r = env.client.get(f"/api/datasets/{ds}/norm-stats")
    ns = r.json()["norm_stats"]
    assert ns["dims"] == 7 and len(ns["min"]) == 7 and len(ns["max"]) == 7
    # every dim has min<=max
    assert all(m <= M for m, M in zip(ns["min"], ns["max"]))


def test_train_requires_ticket_400_and_409(env):
    env.as_student("s01")
    ds = env.upload_ready("s01")
    # 400 without ticket (the test-matrix row)
    r = env.client.post("/api/runs", json={
        "dataset_id": ds, "job_type": "train", "node_id": "auto",
        "config": {"batch": 4}})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "validation"
    # create + link a ticket, then a second train with the same ticket -> 409
    tid = env.make_ticket("s01")
    r = env.client.post("/api/runs", json={
        "dataset_id": ds, "job_type": "train", "ticket_id": tid,
        "node_id": "auto", "config": {"batch": 4}})
    assert r.status_code == 200, r.text
    jid1 = r.json()["job_id"]
    r = env.client.post("/api/runs", json={
        "dataset_id": ds, "job_type": "train", "ticket_id": tid,
        "node_id": "auto", "config": {"batch": 4}})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "conflict"
    # run-now admission: the node is free -> admitted (position 0) or waiting
    assert jid1.startswith("j_")


def test_submit_run_and_drive_to_done_via_sse(env):
    env.as_student("s01")
    ds = env.upload_ready("s01")
    tid = env.make_ticket("s01")
    r = env.client.post("/api/runs", json={
        "dataset_id": ds, "job_type": "train", "ticket_id": tid,
        "node_id": "auto", "config": {"batch": 4, "epochs": 1}})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    # script the agent: a few metric steps, a val, then done with an adapter
    env.runner.script(jid, steps=3, val_accuracy=0.72,
                      artifacts=[{"kind": "adapter", "path": f"artifacts/{jid}/adapter.safetensors",
                                  "nbytes": 12345}])
    events: list[tuple[str, dict]] = []

    def consume():
        # read the SSE stream until the terminal state event arrives;
        # a close-time abort (teardown) may raise — the thread must die clean
        try:
            with env.client.stream("GET", f"/stream/{jid}") as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                etype = None
                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        etype = line[7:]
                    elif line.startswith("data: ") and etype:
                        events.append((etype, __import__("json").loads(line[6:])))
                        etype = None
                        if any(t == "state" and d.get("state") == "done"
                                for t, d in events):
                            break
        except Exception:  # noqa: BLE001 - stream aborted while the thread read it
            pass

    import threading
    t = threading.Thread(target=consume)
    t.start()
    # drive the loop: admit + poll until terminal
    for _ in range(20):
        env.tick()
        row = env.svc.storage.get_job(jid)
        if row["state"] in ("done", "failed", "killed"):
            break
        time.sleep(0.01)
    t.join(timeout=5)
    row = env.svc.storage.get_job(jid)
    assert row["state"] == "done", row
    kinds = {t for t, _ in events}
    assert "state" in kinds and "metrics" in kinds
    # the ticket link released at terminal (re-linkable / record kept)
    tk = env.svc.storage.get_ticket(tid)
    assert tk["state"] == "released"
    assert tk["linked_job"] is None
    # artifacts persisted
    arts = env.svc.storage.artifacts_for(jid)
    assert any(a["kind"] == "adapter" for a in arts)
    # GET run reflects terminal + config
    r = env.client.get(f"/api/runs/{jid}")
    assert r.status_code == 200 and r.json()["state"] == "done"


def test_idempotency_dedup(env):
    env.as_student("s01")
    ds = env.upload_ready("s01")
    tid = env.make_ticket("s01")
    payload = {"dataset_id": ds, "job_type": "train", "ticket_id": tid,
               "node_id": "auto", "config": {"batch": 4}}
    h = {"Idempotency-Key": "11111111-1111-4111-8111-111111111111"}
    r1 = env.client.post("/api/runs", json=payload, headers=h)
    r2 = env.client.post("/api/runs", json=payload, headers=h)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert r2.json().get("deduped") is True


def test_run_now_orders_ahead_of_fifo(env):
    env.as_student("s01")
    ds1 = env.upload_ready("s01")
    r = env.client.post("/api/runs", json={
        "dataset_id": ds1, "job_type": "smoke", "node_id": "auto",
        "config": {"batch": 4}})
    assert r.status_code == 200
    head = r.json()["job_id"]
    env.runner.script(head, steps=2)
    env.tick()  # head admitted + running (1 train slot occupied)
    # s07: a normal job lands in line; a run-now job jumps ahead of it (§3)
    env.client.cookies.clear()
    env.as_student("s07")
    ds7 = env.upload_ready("s07")
    r = env.client.post("/api/runs", json={
        "dataset_id": ds7, "job_type": "smoke", "node_id": "auto",
        "config": {"batch": 4}})
    assert r.status_code == 200, r.text
    pos_normal = r.json()["position"]
    r = env.client.post("/api/runs", json={
        "dataset_id": ds7, "job_type": "smoke", "node_id": "auto",
        "run_now": True, "config": {"batch": 4}})
    assert r.status_code == 200, r.text
    jid_runnow = r.json()["job_id"]
    # live queue positions: run-now sits at the front (1), the earlier
    # normal job drops to 2 — "front of the line after the current
    # completion" (no preemption, §3)
    queue = {j["job_id"]: j for j in env.client.get("/api/runs/queue").json()["queue"]}
    queued = [j for j in queue.values() if j["state"] == "queued"]
    assert queue[jid_runnow]["position"] == 1
    assert sorted(j["position"] for j in queued) == [1, 2]


def test_student_cannot_see_another_students_run(env):
    env.as_student("s01")
    ds1 = env.upload_ready("s01")
    tid = env.make_ticket("s01")
    r = env.client.post("/api/runs", json={
        "dataset_id": ds1, "job_type": "train", "ticket_id": tid,
        "node_id": "auto", "config": {"batch": 4}})
    jid = r.json()["job_id"]
    # switch to s07
    env.client.cookies.clear()
    env.as_student("s07")
    r = env.client.get(f"/api/runs/{jid}")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"
    r = env.client.post(f"/api/runs/{jid}/kill")
    assert r.status_code == 403
    # queue (class board) is still visible to all
    r = env.client.get("/api/runs/queue")
    assert r.status_code == 200
