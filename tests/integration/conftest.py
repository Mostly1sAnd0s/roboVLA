"""Integration conftest (design §11): the FastAPI app with fakes at every
seam (storage = real SQLite in tmp, runner = FakeRunner, fs = tmp dir), so the
whole service + SSE path runs on CPU. One console + one enabled node
(``testnode``), 32 GB free VRAM, 1 train / 2 infer slots — the launch shape.
"""

from __future__ import annotations

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

NODE = "testnode"

FLEET = """
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
    backup: {enabled: true, at: "02:30", dest: /tmp/bk}
nodes:
  - id: testnode
    label: "test node"
    image: robovla-runner-cuda:2026.10
    workdir: /tmp/w
    agent: {lan_ip: "10.0.0.9", port: 8601, token_ref: testnode-agent}
    gpu: {vram_gb: 32, quant: bnb8, train_slots: 1, infer_slots: 2,
          budgets: {train_gb: 18, infer_gb: 11}}
    presets: {batch: [1, 4, 8]}
    priority: 0
    student_selectable: true
    enabled: true
"""


def episode_zip(ep_count: int = 101, episodes_per_task: str = "stop-at-target") -> bytes:
    """A valid dataset zip (>=100 episodes to clear the QA floor)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(ep_count):
            ep = f"ep_{i:04d}"
            frames = f"episodes/{ep}/frames/"
            for f in range(3):
                b = io.BytesIO()
                Image.new("RGB", (32, 24), color=(i % 256, f, 100)).save(b, "JPEG")
                zf.writestr(f"{frames}{f}.jpg", b.getvalue())
            actions = [{"t": k * 0.1, "vx": 0.1 * (i % 7), "vy": 0.0, "wz": 0.05} for k in range(3)]
            zf.writestr(f"episodes/{ep}/actions.json", json.dumps(actions))
            zf.writestr(f"episodes/{ep}/episode.json", json.dumps(
                {"id": ep, "task": episodes_per_task, "prompt": "p", "success": True,
                 "pilot": "s01", "created": "2026-10-20T10:00:00Z"}))
    return buf.getvalue()


class Env:
    def __init__(self, tmp: pathlib.Path, free_gb: float = 30.0,
                 ui_dir: pathlib.Path | None = None):
        self.tmp = tmp
        self.fleet = parse_fleet(FLEET)
        self.storage = Storage(tmp / "state.db")
        self.runner = FakeRunner(free_gb=free_gb)
        self.svc = ConsoleService(
            tmp, self.fleet, self.storage, {NODE: self.runner},
            clock=lambda: "2026-10-21T09:00:00Z", tick_s=0.01)
        self.app = create_app(self.svc, ui_dir=ui_dir)
        self.client = TestClient(self.app)
        self._seed_users()

    def _seed_users(self):
        from robovla.http.auth import hash_password
        self.svc.storage.create_user("teacher", "teacher", hash_password("teach123"),
                                     display_name="Ms T", active=True,
                                     at="2026-10-21T09:00:00Z")
        for i in (1, 7):
            self.svc.storage.create_user(f"s{i:02d}", "student", hash_password(f"stu{i}123"),
                                         display_name=f"Student {i}", active=True,
                                         at="2026-10-21T09:00:00Z")

    # -- login helpers (return a logged-in TestClient share) -------------------
    def login(self, username: str, password: str) -> TestClient:
        r = self.client.post("/api/auth/login", json={"username": username,
                                                      "password": password})
        assert r.status_code == 200, r.text
        return self.client

    def as_student(self, uid: str = "s01") -> TestClient:
        return self.login(uid, f"stu{int(uid[1:])}123")

    def as_teacher(self) -> TestClient:
        return self.login("teacher", "teach123")

    def upload_ready(self, uid: str = "s01", task: str = "stop-at-target",
                     ep_count: int = 101) -> str:
        """Chunked upload + finalize -> a 'ready' dataset id."""
        ds = f"{uid}/{task}/v1"
        blob = episode_zip(ep_count)
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
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ready", r.text
        return ds

    def make_ticket(self, uid: str = "s01") -> str:
        r = self.client.post("/api/tickets", json={
            "hypothesis": "h", "config_summary": "c", "prediction": "p"})
        assert r.status_code == 200, r.text
        return r.json()["ticket_id"]

    def tick(self, n: int = 1):
        import asyncio
        for _ in range(n):
            asyncio.run(self.svc.tick())

    def close(self):
        self.client.close()
        self.storage.close()


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()
