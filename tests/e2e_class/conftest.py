"""E2E class scenario (design §11: scripted 12-student scenario — nightly).

One console + three nodes (pai1 launch shape; node2/node3 offline until
the extended-config phase), the 12 students from the shipped roster.csv,
FakeRunner at every node, real SQLite + real HTTP routes — the whole
class day on CPU. The scenario phases are ORDERED module-scoped tests
(test_class_day.py):

    p1 launch    12 uploads -> 12 tickets -> 12 submissions (pai1 only)
    p2 extended  +node2/node3: first-fit by priority, presets,
                 student-selectable gating, teacher pin (requeue)
    p3 drain     FIFO + run-now bump + kill + requeue pin, mid-drain
    p4 demo-day  2-slot inference relay (launch shape)
    p5 backup    backup trigger + restore round-trip (SPOF procedure §10)

Seams: the RateLimiter's sliding window is driven by the injected
``monotonic`` — 13 users share one TestClient IP (login 5/min/IP, §9.8),
so the env advances the window deterministically (61 s every 5 logins).
The clock is frozen, so job ids land on one day. FakeRunners are scripted
per job on EVERY node (3 metric steps, then done + an adapter artifact):
a job's admission node is only known at dispatch, so every runner must
be able to serve every job.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import pathlib
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from robovla.core.config import parse_fleet
from robovla.http.app import create_app
from robovla.runner import FakeRunner
from robovla.service import ConsoleService
from robovla.storage import Storage

REPO = pathlib.Path(__file__).resolve().parents[2]
ROSTER = REPO / "roster.csv"
STUDENTS = tuple(f"s{i:02d}" for i in range(1, 13))
NODES = ("pai1", "node2", "node3")
CLOCK = "2026-10-21T09:00:00Z"
STEPS = 3  # per-job script length (metric polls before the terminal one)


def _fleet_yaml(backup_dest: str) -> str:
    """The launch fleet: pai1 selectable + enabled; node2/node3 join later
    (admin node patches in phase 2). node3 carries the higher priority —
    first-fit walks priority first, ties keep file order."""
    return f"""
version: 1
console:
  bind: 127.0.0.1
  port: 8501
  tls: false
  data_dir: /tmp/x
  session_ttl_h: 12
  limits:
    student:
      max_running: 1
      max_queued: 2
      max_dataset_bytes: 2147483648
      max_run_min: 180
      run_now: true
    admin_rate: "60/min"
  retention:
    run_logs_per_student: 20
    merged_export_days: 14
    backup: {{enabled: true, at: "02:30", dest: {backup_dest}}}
nodes:
  - id: pai1
    label: "pai1 (RTX 5090, primary)"
    image: robovla-runner-cuda:2026.10
    workdir: /tmp/w
    agent: {{lan_ip: "10.0.0.1", port: 8601, token_ref: pai1-agent}}
    gpu: {{vram_gb: 32, quant: bnb8, train_slots: 1, infer_slots: 2,
          budgets: {{train_gb: 18, infer_gb: 11}}}}
    presets: {{batch: [1, 4, 8]}}
    priority: 0
    student_selectable: true
    enabled: true
  - id: node2
    label: "campus RTX 5090 Ti"
    image: robovla-runner-cuda:2026.10
    workdir: /tmp/w
    agent: {{lan_ip: "10.0.0.2", port: 8601, token_ref: node2-agent}}
    gpu: {{vram_gb: 32, quant: bnb8, train_slots: 1, infer_slots: 2,
          budgets: {{train_gb: 18, infer_gb: 11}}}}
    presets: {{batch: [1, 4, 8]}}
    priority: 0
    student_selectable: false
    enabled: false
  - id: node3
    label: "campus RTX 4070 Ti (batch-1 only)"
    image: robovla-runner-cuda:2026.10
    workdir: /tmp/w
    agent: {{lan_ip: "10.0.0.3", port: 8601, token_ref: node3-agent}}
    gpu: {{vram_gb: 16, quant: bnb8, train_slots: 1, infer_slots: 1,
          budgets: {{train_gb: 14, infer_gb: 11}}}}
    presets: {{batch: [1]}}
    priority: 1
    student_selectable: false
    enabled: false
"""


def _episode_zip(ep_count: int = 101) -> bytes:
    """A valid dataset zip (101 episodes clears the 100-episode QA floor)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(ep_count):
            ep = f"ep_{i:04d}"
            frames = f"episodes/{ep}/frames/"
            for f in range(3):
                b = io.BytesIO()
                Image.new("RGB", (32, 24), color=(i % 256, f, 100)).save(b, "JPEG")
                zf.writestr(f"{frames}{f}.jpg", b.getvalue())
            actions = [{"t": k * 0.1, "vx": 0.1 * (i % 7), "vy": 0.0, "wz": 0.05}
                       for k in range(3)]
            zf.writestr(f"episodes/{ep}/actions.json", json.dumps(actions))
            zf.writestr(f"episodes/{ep}/episode.json", json.dumps(
                {"id": ep, "task": "stop-at-target", "prompt": "p",
                 "success": True, "pilot": "s01",
                 "created": "2026-10-20T10:00:00Z"}))
    return buf.getvalue()


class ClassEnv:
    """The whole class: console + 3 nodes + teacher + 12 students."""

    def __init__(self, tmp: pathlib.Path) -> None:
        self.tmp = tmp
        self._t = 1000.0  # the injected monotonic (rate-limit time source)
        self._logins = 0  # logins since the last window advance
        self.passwords: dict[str, str] = {}

        self.fleet = parse_fleet(_fleet_yaml(str(tmp / "backups")))
        self.storage = Storage(tmp / "state.db")
        self.runners = {
            "pai1": FakeRunner(free_gb=30.0, total_gb=32.0),
            "node2": FakeRunner(free_gb=30.0, total_gb=32.0),
            "node3": FakeRunner(free_gb=15.0, total_gb=16.0),
        }
        self.svc = ConsoleService(
            tmp, self.fleet, self.storage, self.runners,
            clock=lambda: CLOCK, monotonic=lambda: self._t, tick_s=0.01)
        self.client = TestClient(create_app(self.svc))
        self._seed_class()

    # -- time seam ------------------------------------------------------------------

    def advance(self, s: float) -> None:
        self._t += s

    # -- class setup (teacher, roster, credentials) ---------------------------------

    def _seed_class(self) -> None:
        from robovla.http.auth import hash_password

        self.storage.create_user("teacher", "teacher", hash_password("teach123"),
                                 display_name="Ms T", active=True, at=CLOCK)
        self.as_teacher()
        r = self.client.post("/api/admin/users/import",
                             json={"csv": ROSTER.read_text()})
        assert r.status_code == 200, r.text
        assert r.json()["created"] == list(STUDENTS)
        for uid in STUDENTS:
            r = self.client.patch(f"/api/admin/users/{uid}", json={"active": True})
            assert r.status_code == 200, r.text
            r = self.client.post(f"/api/admin/users/{uid}/reset-password")
            assert r.status_code == 200, r.text
            self.passwords[uid] = r.json()["password"]

    # -- auth (login rate limit: 5/min/IP, one shared TestClient IP) -----------------

    def _login(self, username: str, password: str) -> None:
        if self._logins and self._logins % 5 == 0:
            self.advance(61.0)  # let the 60 s window slide
        self._logins += 1
        r = self.client.post("/api/auth/login",
                             json={"username": username, "password": password})
        assert r.status_code == 200, r.text

    def as_teacher(self) -> None:
        self._login("teacher", "teach123")

    def as_student(self, uid: str) -> None:
        assert uid in STUDENTS, uid
        self._login(uid, self.passwords[uid])

    # -- the student workflow ---------------------------------------------------------

    def upload(self, uid: str) -> str:
        """Chunked upload + finalize -> a 'ready' dataset id."""
        self.as_student(uid)
        ds = f"{uid}/stop-at-target/v1"
        blob = _episode_zip()
        sha = hashlib.sha256(blob).hexdigest()
        third = len(blob) // 3
        parts = [blob[:third], blob[third:2 * third], blob[2 * third:]]
        for i, p in enumerate(parts):
            r = self.client.post(
                "/api/datasets/upload",
                data={"dataset_id": ds, "total_chunks": str(len(parts)),
                      "chunk_index": str(i), "sha256": sha},
                files={"file": (f"c{i}.bin", p, "application/octet-stream")})
            assert r.status_code == 200, r.text
        r = self.client.post("/api/datasets/upload/finalize",
                             json={"dataset_id": ds, "total_chunks": len(parts),
                                   "sha256": sha})
        assert r.status_code == 200 and r.json()["status"] == "ready", r.text
        return ds

    def ticket(self, uid: str) -> str:
        self.as_student(uid)
        r = self.client.post("/api/tickets", json={
            "hypothesis": f"raise the stop accuracy (class day {uid})",
            "config_summary": "batch 4, 3 epochs", "prediction": "loss < 2"})
        assert r.status_code == 200, r.text
        return r.json()["ticket_id"]

    def submit_train(self, uid: str, ds: str, ticket_id: str,
                     batch: int = 4, node_id: str = "auto") -> dict:
        """POST /api/runs + the FakeRunner script (on every node — the
        admission node is only known at dispatch). Returns the response."""
        self.as_student(uid)
        r = self.client.post("/api/runs", json={
            "dataset_id": ds, "job_type": "train", "ticket_id": ticket_id,
            "node_id": node_id, "config": {"batch": batch}})
        assert r.status_code == 200, r.text
        jid = r.json()["job_id"]
        for runner in self.runners.values():
            runner.script(jid, steps=STEPS,
                          artifacts=[{"kind": "adapter",
                                      "path": f"artifacts/{jid}/adapter.safetensors",
                                      "nbytes": 1_000_000}])
        return r.json()

    # -- admin (teacher) controls -------------------------------------------------------

    def patch_node(self, nid: str, **fields: object) -> None:
        self.as_teacher()
        r = self.client.patch(f"/api/admin/nodes/{nid}", json=fields)
        assert r.status_code == 200, r.text

    def requeue(self, jid: str, node_id: str | None) -> dict:
        self.as_teacher()
        r = self.client.post(f"/api/runs/{jid}/requeue",
                             json={"node_id": node_id})
        assert r.status_code == 200, r.text
        return r.json()

    def bump(self, jid: str) -> None:
        self.as_teacher()
        r = self.client.post(f"/api/runs/{jid}/bump")
        assert r.status_code == 200, r.text

    def kill(self, jid: str) -> None:
        self.as_teacher()
        r = self.client.post(f"/api/runs/{jid}/kill")
        assert r.status_code == 200, r.text

    # -- service driving ---------------------------------------------------------------

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            asyncio.run(self.svc.tick())

    def state(self, jid: str) -> str:
        return self.svc.storage.get_job(jid)["state"]

    def run_states(self) -> dict[str, str]:
        rows = self.svc.storage.list_jobs(state_in=(
            "queued", "running", "finishing", "done", "failed", "killed"))
        return {r["job_id"]: r["state"] for r in rows}

    def queue(self) -> list[dict]:
        return self.client.get("/api/runs/queue").json()["queue"]

    def close(self) -> None:
        self.client.close()
        self.storage.close()


@pytest.fixture(scope="module")
def cls(tmp_path_factory: pytest.TempPathFactory) -> ClassEnv:
    """The scripted class day — one env shared by the ordered phases."""
    e = ClassEnv(tmp_path_factory.mktemp("class"))
    yield e
    e.close()
