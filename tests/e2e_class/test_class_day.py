"""The scripted class day (design §11 test matrix: "Simulated class E2E").

Ordered phases over ONE module-scoped ClassEnv (conftest) — 12 students
from the shipped roster.csv, console + 3 nodes, FakeRunner everywhere:

    p1  launch config (pai1 only): 12 uploads -> 12 tickets -> 12 submits
    p2  extended config (+node2/node3): student-selectable gating,
        first-fit by priority + the VRAM budget guard, teacher pin
    p3  the queue drains in the expected order — FIFO + run-now bump +
        kill + a requeue pin, all mid-drain
    p4  demo-day 2-slot inference relay (back to launch shape)
    p5  backup trigger + restore round-trip (the SPOF procedure, §10)

Job ids are sequential per day (the frozen clock): j_…_0001–0012 are the
launch submissions (s01–s12), 0013/0014 the extended-config ones.

Expected drain — admission is slot + VRAM-budget gated (presets are NOT
checked at admission); the interesting part is the stagger: dispatch
runs BEFORE the poll phase in a tick, so a job's terminal frees its node
only at the NEXT tick's dispatch. The run-now job (j10) starts one tick
behind the j4/j5 wave, and with equal job lengths that one-tick stagger
persists all the way down: pai1 is always the first node to free, so it
takes the queue head of every wave (j10, j7, j11, j14); when node2+node3
free together, the priority node (node3, p1) first-fits the head (j5,
j8, j12) and node2 the next (j4 pinned, j9, j13):

    pai1  (p0): j1, j10 (run-now), j7, j11, j14
    node3 (p1): j2, j5, j8, j12
    node2 (p0): j3, j4 (pinned), j9, j13
    j6 killed (queued, never ran); every ticket released at terminal.
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from robovla.backup import restore_backup
from robovla.http.app import create_app
from robovla.runner import FakeRunner
from robovla.service import ConsoleService
from robovla.storage import Storage

# Keep in sync with conftest (no __init__.py package style in this repo —
# the scenario constants are duplicated, not imported).
STUDENTS = tuple(f"s{i:02d}" for i in range(1, 13))
CLOCK = "2026-10-21T09:00:00Z"

J = {uid: f"j_20261021_{i:04d}" for i, uid in enumerate(STUDENTS, start=1)}
J13, J14 = "j_20261021_0013", "j_20261021_0014"  # s02 batch-1, s03 batch-4
PAI1_ORDER = [J["s01"], J["s10"], J["s07"], J["s11"], J14]
NODE3_ORDER = [J["s02"], J["s05"], J["s08"], J["s12"]]
NODE2_ORDER = [J["s03"], J["s04"], J["s09"], J13]
KILLED = J["s06"]


def _drain(cls, until_done: int = 14) -> dict[str, list[str]]:
    """Tick to full drain, observing the admission order per node
    (a job's node is recorded the first tick it is seen running — jobs
    already running when the drain starts are seeded first)."""
    order: dict[str, list[str]] = {}
    seen: set[str] = set()
    rows = cls.run_states()
    for jid, st in rows.items():  # the seeds (one per busy node)
        if st == "running":
            seen.add(jid)
            order.setdefault(cls.svc.storage.get_job(jid)["node_id"], []).append(jid)
    for _ in range(400):  # bounded: 13 jobs x 3 steps + admissions
        cls.tick(1)
        rows = cls.run_states()
        for jid, st in rows.items():
            if st == "running" and jid not in seen:
                seen.add(jid)
                node = cls.svc.storage.get_job(jid)["node_id"]
                order.setdefault(node or "?", []).append(jid)
        active = [j for j, s in rows.items()
                  if s in ("queued", "running", "finishing")]
        if len(active) == 0:
            break
    assert len(seen) + 1 == until_done, f"drain stuck: {cls.run_states()}"
    return order


def test_p1_launch_12_uploads_tickets_submits(cls):
    """Launch shape (pai1 only): every student uploads, tickets, submits."""
    tickets = {}
    for uid in STUDENTS:
        ds = cls.upload(uid)
        assert ds == f"{uid}/stop-at-target/v1"
        tickets[uid] = cls.ticket(uid)
        r = cls.submit_train(uid, ds, tickets[uid], batch=4)
        assert r["job_id"] == J[uid], r
    assert cls.run_states()  # 12 queued jobs exist

    # first tick: only pai1 is enabled -> j1 admitted there, FIFO behind it
    cls.tick(1)
    assert cls.state(J["s01"]) == "running"
    assert cls.svc.storage.get_job(J["s01"])["node_id"] == "pai1"
    queue = {q["job_id"]: q for q in cls.queue()}
    queued = [q for q in queue.values() if q["state"] == "queued"]
    assert [q["job_id"] for q in sorted(queued, key=lambda q: q["position"])] \
        == [J[uid] for uid in STUDENTS[1:]]


def test_p2_extended_config_gating_first_fit_pinning(cls):
    """+node2/node3: students may not name non-selectable nodes (403);
    first-fit walks priority (node3 p1) under the budget guard; the
    teacher pins a queued job (requeue) and clears a pin."""
    cls.patch_node("node2", enabled=True)
    cls.patch_node("node3", enabled=True)

    # student-selectable gating (§8.2): 403 on both non-selectable nodes —
    # the ticket must be fresh (the launch one is linked) and survives the
    # failed submits (they fail before linking)
    cls.as_student("s02")
    t2 = cls.ticket("s02")
    r = cls.client.post("/api/runs", json={
        "dataset_id": "s02/stop-at-target/v1", "job_type": "train",
        "ticket_id": t2, "node_id": "node2", "config": {"batch": 1}})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    r = cls.client.post("/api/runs", json={
        "dataset_id": "s02/stop-at-target/v1", "job_type": "train",
        "ticket_id": t2, "node_id": "node3", "config": {"batch": 1}})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"

    # the same tickets, node auto: s02 batch-1 + s03 batch-4 admitted now
    r = cls.submit_train("s02", "s02/stop-at-target/v1", t2, batch=1)
    assert r["job_id"] == J13
    t3 = cls.ticket("s03")
    r = cls.submit_train("s03", "s03/stop-at-target/v1", t3, batch=4)
    assert r["job_id"] == J14

    # first-fit by priority under the budget guard: node3 (p1) takes the
    # auto queue HEAD (j2, the oldest queued job — not the newest); node2
    # takes the next head (j3). Pai1's slot is still j1's (no preemption).
    cls.tick(1)
    states = {jid: cls.svc.storage.get_job(jid) for jid in (J13, J14,
                                                            J["s02"], J["s03"])}
    assert states[J["s02"]]["state"] == "running"
    assert states[J["s02"]]["node_id"] == "node3"
    assert states[J["s03"]]["state"] == "running"
    assert states[J["s03"]]["node_id"] == "node2"
    assert states[J13]["state"] == "queued"  # behind the launch queue
    assert states[J14]["state"] == "queued"
    assert cls.state(J["s01"]) == "running"  # no preemption

    # teacher pin (requeue) on a NON-selectable node + unpin: durable pin
    # with the budget re-derived for the target node
    cls.requeue(J["s04"], "node2")
    row = cls.svc.storage.get_job(J["s04"])
    assert row["node_id"] == "node2"
    assert row["spec"]["limits"]["budget_gb"] == 14.0  # node2 @ batch 4
    cls.requeue(J["s05"], None)
    assert cls.svc.storage.get_job(J["s05"])["node_id"] is None


def test_p3_drain_fifo_kill_bump_requeue(cls):
    """Mid-drain: a queued job is killed (ticket released), a queued job
    is bumped to run-now, the pinned job lands on its pinned node, and
    the whole queue drains in the expected order."""
    cls.kill(KILLED)
    assert cls.state(KILLED) == "killed"
    t6 = cls.svc.storage.get_job(KILLED)["ticket_id"]
    assert cls.svc.storage.get_ticket(t6)["state"] == "released"

    cls.bump(J["s10"])
    # run-now is the scheduler's queue state (the durable pin is node_id)
    assert cls.svc.scheduler.job(J["s10"]).run_now is True

    order = _drain(cls)
    assert order.get("pai1", []) == PAI1_ORDER
    assert order.get("node3", []) == NODE3_ORDER
    assert order.get("node2", []) == NODE2_ORDER
    assert KILLED not in {j for js in order.values() for j in js}
    states = cls.run_states()
    assert all(s in ("done", "killed") for s in states.values())
    assert sum(1 for s in states.values() if s == "done") == 13

    # every experiment ticket released at its run's terminal state
    for uid in STUDENTS:
        tid = cls.svc.storage.get_job(J[uid])["ticket_id"]
        assert cls.svc.storage.get_ticket(tid)["state"] == "released"


def test_p4_demo_day_2_slot_relay(cls):
    """Demo day (§7.7) at launch shape: 2 infer slots on pai1 — students
    claim, the class relays (release -> the queued claim is served), and
    the slot board tracks it. An active slot counts as running (the
    max_running=1 guard stays respected)."""
    cls.patch_node("node2", enabled=False)
    cls.patch_node("node3", enabled=False)

    def claim(uid: str, run_id: str) -> dict:
        cls.as_student(uid)
        r = cls.client.post("/api/infer/slots/claim", json={"run_id": run_id})
        assert r.status_code == 200, r.text
        return r.json()

    def board() -> dict[str, str]:
        cls.as_teacher()
        return {s["student"]: s["state"]
                for s in cls.client.get("/api/infer/slots").json()["slots"]}

    c1 = claim("s01", J["s01"])          # slot 1
    c2 = claim("s11", J["s11"])          # slot 2
    assert c1.get("slot") and c2.get("slot")  # both served immediately
    assert board() == {"s01": "running", "s11": "running"}

    c3 = claim("s12", J["s12"])          # 2 slots full -> queued
    assert c3["queued"] is True and c3["position"] == 1

    # the relay: s01 presents (releases) -> s12's robot takes the slot
    cls.as_student("s01")
    r = cls.client.post(f"/api/infer/slots/{c1['job_id']}/release")
    assert r.status_code == 200 and r.json()["state"] == "killed"
    cls.tick(1)
    assert board() == {"s11": "running", "s12": "running"}

    # second cycle: s01 re-claims (queued behind s11+s12), s11 releases
    c4 = claim("s01", J["s01"])
    assert c4["queued"] is True
    cls.as_student("s11")
    r = cls.client.post(f"/api/infer/slots/{c2['job_id']}/release")
    assert r.status_code == 200
    cls.tick(1)
    assert board() == {"s01": "running", "s12": "running"}
    # s01's released slot is terminal (a released slot never lingers)
    assert cls.state(c1["job_id"]) == "killed"
    assert cls.state(c2["job_id"]) == "killed"


def test_p5_backup_restore_round_trip(cls, tmp_path: pathlib.Path):
    """The SPOF procedure (§10): admin backup trigger -> bundle on disk ->
    restore into a fresh data dir -> a fresh console over the restore sees
    the same class: users, datasets, jobs (states), tickets, audit."""
    cls.as_teacher()
    r = cls.client.post("/api/admin/backups/trigger")
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["status"] == "ok" and pathlib.Path(info["path"]).is_file()
    r = cls.client.get("/api/admin/backups")
    assert len(r.json()["backups"]) == 1

    # restore: refuses a non-empty data dir without force (§9.6 conflict)
    from robovla.core.errors import RobovlaError

    fresh = tmp_path / "restored" / "data"
    Storage(fresh / "state.db").close()
    with pytest.raises(RobovlaError) as e:
        restore_backup(info["path"], fresh)
    assert e.value.code == "conflict"
    out = restore_backup(info["path"], fresh, force=True)
    assert out["db"] is True and out["files"] > 0

    # a fresh console over the restored data dir (new seams, same fleet —
    # no lifespan loop: the restored state is asserted, not re-driven)
    storage = Storage(fresh / "state.db")
    runners = {"pai1": FakeRunner(), "node2": FakeRunner(),
               "node3": FakeRunner(free_gb=15.0, total_gb=16.0)}
    svc = ConsoleService(
        fresh, cls.fleet, storage, runners,
        clock=lambda: CLOCK, monotonic=lambda: 5000.0, tick_s=0.01)
    tc = TestClient(create_app(svc))
    r = tc.post("/api/auth/login", json={"username": "teacher",
                                         "password": "teach123"})
    assert r.status_code == 200, r.text

    users = {u["user_id"]: u for u in
             tc.get("/api/admin/users").json()["users"]}
    assert set(users) == set(STUDENTS) | {"teacher"}
    assert all(users[uid]["active"] for uid in STUDENTS)

    for uid in STUDENTS:
        ds = storage.get_dataset(f"{uid}/stop-at-target/v1")
        assert ds and ds["status"] == "ready"
        assert ds["qa"]["summary"]["episode_count"] == 101

    jobs = {r_["job_id"]: r_ for r_ in storage.list_jobs(state_in=(
        "queued", "running", "finishing", "done", "failed", "killed"))}
    assert len(jobs) == 18  # 14 train + 4 infer
    for uid in STUDENTS:
        if uid == "s06":
            continue  # the mid-drain kill (KILLED below)
        assert jobs[J[uid]]["state"] == "done"
    assert jobs[KILLED]["state"] == "killed"
    assert {j["state"] for j in jobs.values() if j["spec"]["type"] == "infer"} \
        == {"running", "killed"}
    for uid in STUDENTS:
        tid = jobs[J[uid]]["ticket_id"]
        assert storage.get_ticket(tid)["state"] == "released"
    assert len(storage.audit_entries()) >= 30  # the class day is audited
    storage.close()
