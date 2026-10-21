"""agent.server — the :8601 HTTP surface (design §4, §9.9).

Tested over ASGI (httpx.ASGITransport — no sockets), with REAL collaborators
where the logic is the point (JobRunner + real subprocess jobs, LogRing) and
fakes only at the hardware seam (VramProbe over a canned smi runner, the
serving model). Error shape and codes must match what HttpRunner expects:
{"error": {"code", "message"}} + STATUS_BY_CODE, and 401 for bad tokens.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.config import AgentConfig
from agent.jobs import JobRunner
from agent.serve import SlotServing
from agent.server import LogRing, build_app, make_agent_app
from agent.vram import VramProbe
from robovla.core.spec import build_job_spec

TOKEN = "tok-agent"
FAKE_JOB = Path(__file__).parents[1] / "fixtures" / "fake_job.py"

GPU_LINE = "NVIDIA GeForce RTX 5090, 32608, 23552\n"


class ScriptedModel:
    """Deterministic serving model (CPU)."""

    def __init__(self, action=(0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0), ms=1.0):
        self.action = list(action)
        self.ms = ms
        self.unloaded = False

    def predict(self, image_b64, prompt):
        time.sleep(self.ms / 1000)
        return self.action

    def unload(self):
        self.unloaded = True


def _factory(model: ScriptedModel):
    def factory(run_id, adapter):
        return model
    return factory


def _canned_probe(jobs: JobRunner):
    """Canned nvidia-smi whose process list includes the live job pids —
    so agent_spawned marking is tested against REAL processes."""
    def run(argv: list[str], **kw) -> subprocess.CompletedProcess:
        if "query-gpu" in " ".join(argv):
            return subprocess.CompletedProcess(argv, 0, stdout=GPU_LINE)
        lines = []
        for pid in sorted(jobs.spawned_pids()):
            lines.append(f"{pid}, robovla_job, 9216")
        lines.append("4242, ninfer, 30208")  # host-level process (not ours)
        return subprocess.CompletedProcess(argv, 0, stdout="\n".join(lines) + "\n")
    return VramProbe("cuda", runner=run)


def _cfg(tmp_path, **kw) -> AgentConfig:
    return AgentConfig(
        token=TOKEN, workdir=str(tmp_path),
        model_dir=str(tmp_path / "models" / "openvla-7b"),
        image_tag="robovla-runner-cuda:2026.10", version="0.1.0",
        console_url="", gpu="cuda",
        **kw,
    )


@pytest.fixture
def rig(tmp_path):
    log = LogRing()
    jobs = JobRunner(tmp_path,
                     argv_builder=lambda s, jd: [sys.executable, str(FAKE_JOB),
                                                 "--spec", str(Path(jd) / "spec.json")],
                     grace_s=0.3, log=log.log)
    probe = _canned_probe(jobs)
    model = ScriptedModel()
    serving = SlotServing(_factory(model), log=log.log)
    cfg = _cfg(tmp_path)
    app = build_app(cfg, jobs=jobs, probe=probe, serving=serving, log=log)
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    return {"app": app, "client": client, "jobs": jobs, "cfg": cfg,
            "log": log, "model": model}


def _spec(job_id: str = "j_20261021_0101", **kw) -> dict:
    base = dict(job_id=job_id, student="s07", job_type="train",
                dataset="s07/task-stop-at-target/v1", node_id="pai1",
                node_agent="10.0.0.9:8601", budget_gb=18, batch=4,
                seed=42, overrides={"epochs": 2})
    base.update(kw)
    return build_job_spec(**base).to_dict()


def _wait_state(client, job_id, states, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = client.get(f"/jobs/{job_id}").json()
        if v["state"] in states:
            return v
        time.sleep(0.02)
    raise AssertionError(f"{job_id} never reached {states}")


# -- auth (ONE dependency, every route) --------------------------------------------


@pytest.mark.parametrize("method,path", [
    ("GET", "/health"), ("GET", "/vram"), ("GET", "/log"),
    ("GET", "/jobs/nope"), ("POST", "/jobs"),
    ("POST", "/serve"), ("POST", "/predict"),
    ("DELETE", "/serve/slot-1"),
])
def test_missing_or_bad_token_is_401(rig, method, path, tmp_path):
    plain = TestClient(rig["app"])
    r = plain.request(method, path, json={}) if method == "POST" else plain.request(method, path)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "auth"

    bad = TestClient(rig["app"])
    bad.headers["Authorization"] = "Bearer wrong"
    r = bad.request(method, path, json={}) if method == "POST" else bad.request(method, path)
    assert r.status_code == 401 and r.json()["error"]["code"] == "auth"


# -- error shape ---------------------------------------------------------------------


def test_error_shape_matches_http_runner_expectations(rig):
    r = rig["client"].get("/jobs/j_00000000_0000")
    assert r.status_code == 404
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == "not_found"


def test_bad_request_shapes(rig):
    assert rig["client"].post("/jobs", json={}).status_code == 400  # no spec
    bad = _spec(job_id="j_20261021_0102")
    bad["job_id"] = "bogus"
    assert rig["client"].post("/jobs", json={"spec": bad}).status_code == 400
    r = rig["client"].post("/vram/purge", json={"pids": "nope"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation"
    # purge of a host-level (non-agent) pid is rejected, not executed
    r = rig["client"].post("/vram/purge", json={"pids": [4242]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation"


# -- health / vram / log ---------------------------------------------------------------


def test_health_fields(rig, tmp_path):
    r = rig["client"].get("/health")
    assert r.status_code == 200
    h = r.json()
    assert h["ok"] is True
    assert h["image"] == "robovla-runner-cuda:2026.10"
    assert h["version"] == "0.1.0"
    assert h["gpu"] == "NVIDIA GeForce RTX 5090 31.84GB"
    assert h["disk_free_gb"] > 0
    assert h["model_dir"] == ""  # model dir absent -> join probe fails cleanly
    (tmp_path / "models" / "openvla-7b").mkdir(parents=True)
    assert rig["client"].get("/health").json()["model_dir"] == "models/openvla-7b"


def test_vram_marks_live_jobs_agent_spawned(rig):
    rig["client"].post("/jobs", json={"spec": _spec(job_id="j_20261021_0103")})
    _wait_state(rig["client"], "j_20261021_0103", {"running"})
    v = rig["client"].get("/vram").json()
    assert v["free_gb"] == pytest.approx(8.84)
    procs = {p["pid"]: p for p in v["procs"]}
    live = list(rig["jobs"].spawned_pids())
    assert len(live) == 1
    assert procs[live[0]]["agent_spawned"] is True
    assert procs[live[0]]["name"] == "robovla_job"
    assert procs[4242]["agent_spawned"] is False  # host-level (the 27B twin)


def test_vram_purge_kills_live_job(rig):
    rig["client"].post("/jobs", json={"spec": _spec(seed=888, job_id="j_20261021_0104")})
    _wait_state(rig["client"], "j_20261021_0104", {"running"})
    pid = next(iter(rig["jobs"].spawned_pids()))
    r = rig["client"].post("/vram/purge", json={"pids": [pid]})
    assert r.status_code == 200 and r.json()["killed"] == 1
    assert _wait_state(rig["client"], "j_20261021_0104", {"killed"})["state"] == "killed"
    # once the supervisor reaps the pid, only the host-level twin remains
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        procs = rig["client"].get("/vram").json()["procs"]
        if all(p["pid"] != pid for p in procs):
            break
        time.sleep(0.02)
    assert procs == [{"pid": 4242, "name": "ninfer", "vram_gb": 29.5,
                      "agent_spawned": False}]


def test_agent_log_tail(rig):
    rig["client"].post("/jobs", json={"spec": _spec(job_id="j_20261021_0105")})
    _wait_state(rig["client"], "j_20261021_0105", {"running"})
    r = rig["client"].get("/log")
    assert r.status_code == 200
    assert "submitted type=train" in r.json()["log"]
    r2 = rig["client"].get("/log", params={"lines": 1})
    assert len(r2.json()["log"].strip().splitlines()) == 1


# -- jobs through HTTP -----------------------------------------------------------------


def test_jobs_full_flow_over_http(rig):
    r = rig["client"].post("/jobs", json={"spec": _spec(job_id="j_20261021_0106")})
    assert r.status_code == 200 and r.json() == {"ok": True}
    # duplicate submit -> 409 conflict (the oracle's code)
    r2 = rig["client"].post("/jobs", json={"spec": _spec(job_id="j_20261021_0106")})
    assert r2.status_code == 409 and r2.json()["error"]["code"] == "conflict"

    v = _wait_state(rig["client"], "j_20261021_0106", {"done"})
    assert v["artifacts"][0]["kind"] == "adapter"
    # terminal view carries the final checkpoint value (epochs=2 -> 4 steps,
    # checkpoints at 2 and 4 -> accuracy 0.6 then 0.7)
    assert v["val_accuracy"] == 0.7
    logs = rig["client"].get("/jobs/j_20261021_0106/logs").json()["log"]
    assert "step" in logs
    tail = rig["client"].get("/jobs/j_20261021_0106/logs",
                             params={"offset": 2}).json()["log"]
    assert len(tail.strip().splitlines()) < len(logs.strip().splitlines())
    arts = rig["client"].get("/jobs/j_20261021_0106/artifacts").json()["artifacts"]
    assert arts and arts[0]["nbytes"] > 0


def test_kill_over_http(rig):
    rig["client"].post("/jobs", json={"spec": _spec(seed=888, job_id="j_20261021_0107")})
    _wait_state(rig["client"], "j_20261021_0107", {"running"})
    r = rig["client"].post("/jobs/j_20261021_0107/kill")
    assert r.status_code == 200 and r.json() == {"ok": True}
    v = rig["client"].get("/jobs/j_20261021_0107").json()
    assert v["state"] == "killed" and v["error"] is None
    # kill of an unknown job -> 404
    assert rig["client"].post("/jobs/j_00000000_9999/kill").status_code == 404


def test_infer_spec_rejected_at_submit(rig):
    r = rig["client"].post("/jobs", json={"spec": _spec(job_type="infer",
                                                        job_id="j_20261021_0108")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation"


# -- serving slots (§9.5) ---------------------------------------------------------------


def test_serve_predict_release(rig):
    r = rig["client"].post("/serve", json={"run_id": "j_20261021_0109",
                                           "adapter": "runs/j_1/ckpt/adapter.safetensors"})
    assert r.status_code == 200
    handle = r.json()["handle"]
    assert handle.startswith("slot-")

    p = rig["client"].post("/predict", json={"handle": handle,
                                             "image": "aW1hZ2U=", "prompt": "stop"})
    assert p.status_code == 200
    out = p.json()
    assert out["action_norm"] == [0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0]
    assert out["latency_ms"] >= 0.9  # the scripted 1 ms sleep was measured

    assert rig["client"].delete(f"/serve/{handle}").status_code == 200
    assert rig["model"].unloaded is True
    # released handles stop working (404, like the oracle)
    r2 = rig["client"].post("/predict", json={"handle": handle,
                                              "image": "aW1hZ2U=", "prompt": "x"})
    assert r2.status_code == 404 and r2.json()["error"]["code"] == "not_found"
    assert rig["client"].delete("/serve/slot-99").status_code == 404


def test_serving_validation_and_internal(rig):
    r = rig["client"].post("/serve", json={"run_id": "", "adapter": "a"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation"
    r = rig["client"].post("/predict", json={"handle": "slot-1",
                                             "image": "aW1hZ2U=", "prompt": "x"})
    assert r.status_code == 404

    # a model that breaks the 7-dim contract -> 500 internal (console checks too)
    bad = SlotServing(_factory(ScriptedModel(action=[0.1] * 6)))
    app = build_app(rig["cfg"], jobs=rig["jobs"], probe=rig["app"].state.probe,
                    serving=bad, log=rig["log"])
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {TOKEN}"
    h = c.post("/serve", json={"run_id": "r1", "adapter": "a"}).json()["handle"]
    r = c.post("/predict", json={"handle": h, "image": "aW1hZ2U=", "prompt": "x"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal"


# -- production wiring -------------------------------------------------------------------


def test_make_agent_app_wires_from_config(tmp_path):
    import agent.server as server_mod
    real_probe = VramProbe
    server_mod.VramProbe = lambda gpu: _canned_probe(JobRunner(tmp_path))  # no smi in CI
    try:
        app = make_agent_app(_cfg(tmp_path))
    finally:
        server_mod.VramProbe = real_probe
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {TOKEN}"
    assert c.get("/health").status_code == 200
    assert c.get("/vram").status_code == 200
    # wrong token is rejected on the production app too
    c2 = TestClient(app)
    assert c2.get("/health").status_code == 401
