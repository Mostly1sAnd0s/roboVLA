"""The full console <-> agent loop on CPU (design §4, §9.9, §10).

Real console app + real ConsoleService + real SQLite + real FastAPI agent
app + real JobRunner + a REAL job subprocess (tests/fixtures/fake_job.py).
The two HTTP hops are in-process: HttpRunner (console -> agent :8601) and
the bundle pull (agent -> console :8501) ride a MockTransport router that
delegates each netloc to the other side's TestClient — no sockets, no GPU.

This is the test matrix row "the agent protocol is real": dataset upload ->
prep -> ticket -> submit -> tick -> dispatch -> agent bundle pull (agent
token verified against the console DB) -> fixed-argv subprocess ->
incremental metrics/val -> done + artifacts + audit. It also exercises the
console tolerating the agent's ``queued`` view while the bundle is pulled.
"""

from __future__ import annotations

import asyncio
import importlib.util as iu
import json
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.config import AgentConfig
from agent.jobs import JobRunner
from agent.server import LogRing, build_app, make_puller
from agent.serve import SlotServing
from agent.vram import VramProbe
from robovla.core.config import parse_fleet
from robovla.http.app import create_app
from robovla.http.auth import hash_password, token_lookup_key
from robovla.runner import HttpRunner
from robovla.service import ConsoleService
from robovla.storage import Storage

NODE = "testnode"
CONSOLE_NETLOC = "10.0.0.5:8501"
AGENT_NETLOC = "10.0.0.9:8601"
AGENT_TOKEN = "tok-agent"
FAKE_JOB = Path(__file__).parents[1] / "fixtures" / "fake_job.py"

# Reuse the launch fleet + dataset-zip helper (one definition, DRY): load the
# sibling conftest as a plain module (defs only — safe to exec).
_cc_spec = iu.spec_from_file_location(
    "rob_loop_conftest", Path(__file__).parent / "conftest.py")
_cc = iu.module_from_spec(_cc_spec)
_cc_spec.loader.exec_module(_cc)
FLEET, episode_zip = _cc.FLEET, _cc.episode_zip


# -- the in-process LAN: MockTransport router by netloc ---------------------------


def _router_client(routes: dict[str, TestClient], base_url: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        # Route by host:port. base_url (the agent endpoint) makes HttpRunner's
        # relative paths resolve; the bundle pull's absolute console URL
        # overrides it, so one client serves both directions of the LAN.
        netloc = f"{request.url.host}:{request.url.port}"
        tc = routes.get(netloc)
        if tc is None:
            raise httpx.ConnectError(f"no route for {netloc}")
        path = request.url.path + (f"?{request.url.query}"
                                   if request.url.query else "")
        return tc.request(request.method, path,
                          headers=dict(request.headers),
                          content=request.content)

    return httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))


def _canned_probe() -> VramProbe:
    import subprocess

    def run(argv: list[str], **kw):
        if "query-gpu" in " ".join(argv):
            return subprocess.CompletedProcess(argv, 0,
                                               stdout="NVIDIA GeForce RTX 5090, 32608, 2560\n")
        return subprocess.CompletedProcess(argv, 0, stdout="")

    return VramProbe("cuda", runner=run)


# -- the loop test ------------------------------------------------------------------


def test_console_agent_full_loop(tmp_path):
    data_dir = tmp_path / "console"
    workdir = tmp_path / "agent-workdir"
    routes: dict[str, TestClient] = {}
    lan = _router_client(routes, base_url=f"http://{AGENT_NETLOC}")  # filled once both apps exist

    # -- console side (real service + app + storage) -----------------------------
    fleet = parse_fleet(FLEET)
    storage = Storage(data_dir / "state.db")
    runner = HttpRunner(f"http://{AGENT_NETLOC}", AGENT_TOKEN, client=lan)
    svc = ConsoleService(data_dir, fleet, storage, {NODE: runner},
                         clock=lambda: "2026-10-21T09:00:00Z", tick_s=0.01)
    app = create_app(svc)
    console = TestClient(app)

    # the per-node agent token hash the console verifies the bundle pull with
    # (node add would store this; §9.8: plaintext lives in the container env)
    storage.set_agent_token(NODE, hash_password(AGENT_TOKEN),
                            token_lookup_key(AGENT_TOKEN), at="2026-10-21T09:00:00Z")

    # -- agent side (real app; probe faked — no GPU in CI) -------------------------
    (workdir / "models" / "openvla-7b").mkdir(parents=True)
    log = LogRing()
    agent_cfg = AgentConfig(
        token=AGENT_TOKEN, workdir=str(workdir),
        model_dir=str(workdir / "models" / "openvla-7b"),
        image_tag=fleet.nodes[0].image, version="0.1.0",
        console_url=f"http://{CONSOLE_NETLOC}", gpu="cuda")

    puller = make_puller(agent_cfg, log, client=lan)  # production puller

    jobs = JobRunner(workdir,
                     argv_builder=lambda s, jd: [sys.executable, str(FAKE_JOB),
                                                 "--spec", str(Path(jd) / "spec.json")],
                     grace_s=1.0, puller=puller, log=log.log)
    agent = TestClient(build_app(
        agent_cfg, jobs=jobs, probe=_canned_probe(),
        serving=SlotServing(lambda r, a: _NullModel()), log=log))

    routes[AGENT_NETLOC] = agent
    routes[CONSOLE_NETLOC] = console

    # -- the class flow: login, upload, ticket, submit ----------------------------
    _seed_users(svc, storage)
    student = TestClient(app)
    r = student.post("/api/auth/login", json={"username": "s01", "password": "stu1123"})
    assert r.status_code == 200, r.text

    ds = f"s01/stop-at-target/v1"
    blob = episode_zip(101, "stop-at-target")
    import hashlib
    sha = hashlib.sha256(blob).hexdigest()
    third = len(blob) // 3
    parts = [blob[:third], blob[third:2 * third], blob[2 * third:]]
    for i, p in enumerate(parts):
        r = student.post("/api/datasets/upload",
                         data={"dataset_id": ds, "total_chunks": str(len(parts)),
                               "chunk_index": str(i), "sha256": sha},
                         files={"file": (f"c{i}.bin", p, "application/octet-stream")})
        assert r.status_code == 200, r.text
    r = student.post("/api/datasets/upload/finalize",
                     json={"dataset_id": ds, "total_chunks": len(parts), "sha256": sha})
    assert r.status_code == 200 and r.json()["status"] == "ready", r.text

    r = student.post("/api/tickets", json={"hypothesis": "h", "config_summary": "c",
                                           "prediction": "p"})
    assert r.status_code == 200, r.text
    ticket_id = r.json()["ticket_id"]

    r = student.post("/api/runs", json={
        "dataset_id": ds, "job_type": "train", "ticket_id": ticket_id,
        "node_id": "auto", "config": {"batch": 4}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # -- ticks drive: dispatch -> (queued: pull) -> running -> done ----------------
    deadline = time.monotonic() + 45
    states = []
    while time.monotonic() < deadline:
        asyncio.run(svc.tick())
        row = svc.storage.get_job(job_id)
        states.append(row["state"])
        if row["state"] in ("done", "failed", "killed"):
            break
        time.sleep(0.05)

    row = svc.storage.get_job(job_id)
    assert row["state"] == "done", f"states seen: {states} | error: {row}"
    assert row["node_id"] == NODE

    # artifacts (adapter) persisted on the console
    arts = svc.storage.artifacts_for(job_id)
    assert arts and arts[0]["kind"] == "adapter" and arts[0]["nbytes"] > 0

    # the agent pulled the real bundle into its workdir (agent-token auth, §9.9)
    pulled = workdir / "datasets" / "s01__stop-at-target__v1"
    assert (pulled / "episodes" / "ep_0000" / "episode.json").is_file()
    assert (pulled / "episodes" / "ep_0000" / "frames" / "0.jpg").is_file()

    # audit: the pull was logged with the node as actor (§9.9)
    pulls = storage.audit_entries(action="dataset.pull")
    assert pulls, "dataset.pull audit entry missing"
    assert pulls[-1]["actor"] == NODE

    # SSE history: the job's stream saw running + done + metrics (the real path)
    hist = svc.bus.replay(job_id, 0)
    types = {e[1] for e in hist}
    assert {"state", "metrics"} <= types
    assert any(e[1] == "state" and e[2].get("state") == "running" for e in hist)
    assert any(e[1] == "state" and e[2].get("state") == "done" for e in hist)

    # the agent's log shows the pull + job lifecycle
    agent_log = agent.get("/log", headers={"Authorization": f"Bearer {AGENT_TOKEN}"})
    assert agent_log.status_code == 200
    assert "dataset ready" in agent_log.json()["log"]


def _seed_users(svc: ConsoleService, storage: Storage) -> None:
    at = "2026-10-21T09:00:00Z"
    storage.create_user("s01", "student", hash_password("stu1123"),
                        display_name="Student 1", active=True, at=at)
    storage.create_user("teacher", "teacher", hash_password("teach123"),
                        display_name="Ms T", active=True, at=at)


class _NullModel:
    """Serving model for the loop (inference is not the point here)."""

    def predict(self, image_b64, prompt):
        return [0.5] * 7

    def unload(self):
        pass
