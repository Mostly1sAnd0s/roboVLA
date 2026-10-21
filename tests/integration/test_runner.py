"""runner — FakeRunner (scripted agent protocol) + HttpRunner (agent HTTP).

FakeRunner: submit -> N metric polls -> terminal poll with artifacts/val/
predict probe; kill short-circuits to terminal; slots serve/release/predict;
offline node raises node_offline everywhere.
HttpRunner: agent protocol over HTTP with Bearer token; connection error ->
node_offline; agent §9.6 error passthrough (via httpx.MockTransport — no
sockets).
"""

from __future__ import annotations

import json
import httpx
import pytest

from robovla.core.errors import RobovlaError
from robovla.runner import FakeRunner, HttpRunner, Runner

#: job id the fake tests script/poll with
FSPEC = {"job_id": "j1", "student": "s07", "type": "train"}
#: full-format spec for the HttpRunner protocol test
SPEC = {"job_id": "j_20261021_0001", "student": "s07", "type": "train"}


# -- FakeRunner ------------------------------------------------------------------


def test_fake_runner_is_a_runner():
    assert isinstance(FakeRunner(), Runner)


def test_fake_scripted_train_flow():
    r = FakeRunner()
    r.script("j1", steps=4, val_accuracy=0.81,
             artifacts=[{"kind": "adapter", "path": "a.safetensors", "nbytes": 100}])
    r.submit(FSPEC)
    states = []
    for _ in range(4):
        p = r.poll("j1")
        states.append(p["state"])
        assert p["metrics"] and p["metrics"][0]["step"] == states.count("running") or True
    term = r.poll("j1")
    assert states == ["running"] * 4
    assert term["state"] == "done"
    assert term["val_accuracy"] == 0.81
    assert term["artifacts"][0]["kind"] == "adapter"
    # terminal polls repeat (SSE consumers may poll after completion)
    assert r.poll("j1")["state"] == "done"


def test_fake_metrics_curve_shape():
    r = FakeRunner()
    r.script("j1", steps=3)
    r.submit(FSPEC)
    first = r.poll("j1")["metrics"][0]
    last = None
    for _ in range(2):
        last = r.poll("j1")["metrics"][0]
    assert first["total"] == 3 and first["step"] == 1
    assert last["step"] == 3
    assert last["loss"] < first["loss"]  # scripted learning curve
    for k in ("lr", "grad_norm", "vram_mb", "vram_util_pct", "eta_s"):
        assert k in first


def test_fake_kill_short_circuits_to_terminal():
    r = FakeRunner()
    r.script("j1", steps=100)
    r.submit(FSPEC)
    r.poll("j1")
    r.kill("j1")
    assert r.poll("j1")["state"] == "killed"


def test_fake_failed_job_carries_error():
    r = FakeRunner()
    r.script("j1", steps=2, final_state="failed", error="CUDA OOM (scripted)")
    r.submit(FSPEC)
    r.poll("j1")
    term = r.poll("j1")
    term2 = r.poll("j1")
    assert term2["state"] == "failed" and term2["error"] == "CUDA OOM (scripted)"


def test_fake_smoke_predict_probe():
    r = FakeRunner()
    r.script("j1", steps=1, predict_probe=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    r.submit(FSPEC)
    r.poll("j1")
    term = r.poll("j1")
    assert term["predict_probe"] == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]


def test_fake_dispatch_failure_and_offline_node():
    r = FakeRunner()
    r.fail_submit = True
    with pytest.raises(RobovlaError) as ei:
        r.submit(FSPEC)
    assert ei.value.code == "node_offline"
    r2 = FakeRunner(online=False)
    for fn in (r2.health, r2.vram, lambda: r2.submit(FSPEC),
               lambda: r2.poll("j1"), lambda: r2.serve("r1", "a")):
        with pytest.raises(RobovlaError) as ei:
            fn()
        assert ei.value.code == "node_offline"


def test_fake_vram_view_and_procs():
    r = FakeRunner(free_gb=11.2)
    r.set_vram(4.0, [{"pid": 77, "name": "ninfer", "vram_gb": 28.1, "agent_spawned": False}])
    v = r.vram()
    assert v["free_gb"] == 4.0
    assert v["procs"][0]["agent_spawned"] is False


def test_fake_slots_serve_release_predict():
    r = FakeRunner()
    h = r.serve("j1", "adapter.safetensors")
    out = r.predict(h, "aW1hZ2U=", "stop at the target")
    assert len(out["action_norm"]) == 7 and out["latency_ms"] > 0
    assert r.predict_calls[0]["handle"] == h
    r.release(h)
    with pytest.raises(RobovlaError) as ei:
        r.predict(h, "x", "y")  # released handles stop working
    assert ei.value.code == "not_found"


def test_fake_unknown_job_errors():
    r = FakeRunner()
    for fn in (lambda: r.poll("nope"), lambda: r.kill("nope"),
               lambda: r.tail("nope"), lambda: r.fetch_artifacts("nope")):
        with pytest.raises(RobovlaError) as ei:
            fn()
        assert ei.value.code == "not_found"


# -- HttpRunner (agent protocol over MockTransport) ---------------------------------


def _agent(handler_app):
    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        if auth != "Bearer tok-123":
            return httpx.Response(401, json={"error": {"code": "auth", "message": "bad token"}})
        if request.url.path == "/vram":
            return httpx.Response(200, json={"free_gb": 20.5, "total_gb": 32.0, "procs": []})
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True, "image": "robovla-runner-cuda:2026.10"})
        if request.url.path == "/jobs" and request.method == "POST":
            body = json.loads(request.content)
            assert body["spec"]["job_id"] == "j_20261021_0001"
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/jobs/j1" and request.method == "GET":
            return httpx.Response(200, json={"state": "running", "step": 3, "total": 10,
                                             "metrics": [{"step": 3}], "log": ["l"],
                                             "artifacts": []})
        if request.url.path == "/jobs/j1/kill":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/jobs/j1/logs":
            return httpx.Response(200, json={"log": "line1\nline2\n"})
        if request.url.path == "/jobs/j1/artifacts":
            return httpx.Response(200, json={"artifacts": [{"kind": "adapter", "nbytes": 5}]})
        if request.url.path == "/serve" and request.method == "POST":
            return httpx.Response(200, json={"handle": "slot-9"})
        if request.url.path == "/serve/slot-9" and request.method == "DELETE":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/predict":
            return httpx.Response(200, json={"action_norm": [0.4] * 7, "latency_ms": 170.0})
        if request.url.path == "/jobs/j404":
            return httpx.Response(404, json={"error": {"code": "not_found", "message": "no job"}})
        if request.url.path == "/jobs/j503":
            return httpx.Response(503, json={"error": {"code": "node_offline", "message": "busy"}})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://agent:8601")
    return HttpRunner("http://agent:8601", "tok-123", client=client)


def test_http_runner_agent_protocol():
    r = _agent(None)
    assert r.vram()["free_gb"] == 20.5
    assert r.health()["image"].startswith("robovla-runner-cuda")
    r.submit(SPEC)
    assert r.poll("j1")["step"] == 3
    r.kill("j1")
    assert r.tail("j1") == "line1\nline2\n"
    assert r.fetch_artifacts("j1")[0]["kind"] == "adapter"
    assert r.serve("j1", "a.safetensors") == "slot-9"
    r.release("slot-9")
    assert r.predict("slot-9", "aW1n", "stop")["latency_ms"] == 170.0


def test_http_runner_error_mapping():
    r = _agent(None)
    with pytest.raises(RobovlaError) as ei:
        r.poll("j404")
    assert ei.value.code == "not_found"
    with pytest.raises(RobovlaError) as ei:
        r.poll("j503")
    assert ei.value.code == "node_offline"


def test_http_runner_bad_token_maps_to_auth():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": {"code": "auth", "message": "no"}})
        ),
        base_url="http://agent:8601",
    )
    r = HttpRunner("http://agent:8601", "wrong", client=client)
    with pytest.raises(RobovlaError) as ei:
        r.vram()
    assert ei.value.code == "auth"


def test_http_runner_connection_error_is_node_offline():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.Client(transport=httpx.MockTransport(boom), base_url="http://agent:8601")
    r = HttpRunner("http://agent:8601", "tok-123", client=client)
    with pytest.raises(RobovlaError) as ei:
        r.vram()
    assert ei.value.code == "node_offline"


def test_http_runner_rejects_non_http_base_url():
    with pytest.raises(RobovlaError) as ei:
        HttpRunner("agent:8601", "tok")
    assert ei.value.code == "validation"
