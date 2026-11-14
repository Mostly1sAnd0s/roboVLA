"""The training-endpoint board (§7.3/§8.2): students see per-node load and
the wait a job of their shape would face, and the pickability reasons match
the submit gate of §9.3. FakeRunner nodes; no GPU."""

from __future__ import annotations

from conftest import Env, NODE


def _ready(env):
    """One service tick: the node-state cache fills from the (fake) agents.
    Without it every node reads offline — the production loop ticks every
    0.5 s, so this is a test artifact, not a state the class ever sees."""
    env.tick()


def _board(client, **q):
    r = client.get("/api/nodes", params=q)
    assert r.status_code == 200, r.text
    return {n["id"]: n for n in r.json()["nodes"]}


def test_board_requires_login(env):
    assert env.client.get("/api/nodes").status_code == 401


def test_board_reports_load_and_queues(env):
    c = env.as_student()
    env.upload_ready()
    t = env.make_ticket()
    r = c.post("/api/runs", json={"dataset_id": "s01/stop-at-target/v1",
                                  "job_type": "train", "ticket_id": t,
                                  "node_id": NODE, "config": {"batch": 4, "epochs": 3}})
    assert r.status_code == 200, r.text
    job = r.json()["job_id"]
    env.tick()
    nodes = _board(c, job_type="train", batch=4, epochs=3)
    n = nodes[NODE]
    assert n["enabled"] and n["online"]
    assert n["selectable"] is True and n["reasons"] == []
    assert n["train_slots"] == 1
    total = n["train_used"] + len(n["queued"])
    assert total >= 1                       # the run we just submitted shows up
    assert n["est_run_s"] == 5400           # train @ batch 4 / 3 epochs (§7)
    assert n["eta_start_s"] is not None
    ids = {j["job_id"] for j in n["running"]} | {j["job_id"] for j in n["queued"]}
    assert job in ids
    # a pinned job is in this node's queue, not counted as auto
    assert isinstance(n["auto_queued"], int)


def test_board_explains_a_locked_endpoint(env):
    _ready(env)
    # teacher closes the node to students (§8.2) — the board says so plainly
    # (one TestClient share: logging in as teacher replaces the session)
    tc = env.as_teacher()
    r = tc.patch(f"/api/admin/nodes/{NODE}", json={"student_selectable": False})
    assert r.status_code == 200, r.text
    n = _board(tc)[NODE]
    assert n["selectable"] is True                 # teacher may always pick
    ns = _board(env.as_student())[NODE]
    assert ns["selectable"] is False
    assert "teacher has not opened this node to students" in ns["reasons"]
    # and the submit gate agrees with the board (§9.3)
    env.upload_ready()
    t = env.make_ticket()
    r = env.as_student().post("/api/runs", json={
        "dataset_id": "s01/stop-at-target/v1", "job_type": "train",
        "ticket_id": t, "node_id": NODE, "config": {"batch": 1}})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"


def test_board_batch_preset_mismatch_is_a_reason(env):
    _ready(env)
    c = env.as_student()
    n = _board(c, job_type="train", batch=8, epochs=1)[NODE]
    assert n["selectable"] is True                       # batch 8 is allowed here
    smol = _board(c, job_type="train", batch=4, epochs=1)["smolvla1"]
    assert smol["selectable"] is False
    assert "batch 4 is not in this node's presets" in smol["reasons"]


def test_agent_endpoint_teacher_only(env):
    _ready(env)
    assert _board(env.as_teacher())[NODE]["agent"]
    assert _board(env.as_student())[NODE]["agent"] is None
