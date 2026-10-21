"""The image's ENTRYPOINT boots the agent, serves ``/health``, and dispatches
jobs with the configured job bin.

Phase 1 (host65 first real boot, 2026-10-21) found two dead-on-arrival
defects the CPU suite never saw (it drives ``make_agent_app`` directly):
  1. both Dockerfiles shipped ``ENTRYPOINT [".../python", "-m",
     "agent.server"]`` — a module with NO ``__main__`` guard: the process
     imported and exited silently (exit 0, zero output) — dead agent;
  2. ``ROBOVLA_JOB_BIN`` (documented in agent.config) never reached the
     spawned argv — the agent always used the image default
     ``/opt/robovla/bin``.
These rows run the EXACT shipped entrypoint argv (only the image's venv
python is swapped for the test venv's) and require a live ``/health`` plus,
for the dispatch row, a real job whose argv.json carries the configured bin.
CPU-safe: without a GPU the probe degrades to facts, not a 500.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).parents[2]
ENTRY_RE = re.compile(r"^ENTRYPOINT \[(.*)\]\s*$", re.M)
JOB_ID = "j_20261021_9001"


def _entrypoint_argv(dockerfile: Path) -> list[str]:
    """The shipped ENTRYPOINT argv (the image's venv python stays argv[0])."""
    m = ENTRY_RE.search(dockerfile.read_text())
    assert m, f"no ENTRYPOINT line in {dockerfile.name}"
    return [s.strip().strip('"') for s in m.group(1).split(",") if s.strip()]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _agent_env(tmp_path: Path, port: int, token: str, workdir: Path) -> dict:
    return dict(
        os.environ,
        ROBOVLA_AGENT_TOKEN=token,
        ROBOVLA_WORKDIR=str(workdir),
        ROBOVLA_MODEL_DIR=str(workdir / "models" / "openvla-7b"),
        ROBOVLA_IMAGE_TAG="robovla-runner-cuda:test",
        ROBOVLA_GPU="cuda",
        ROBOVLA_JOB_BIN=str(tmp_path / "bin"),
        ROBOVLA_AGENT_PORT=str(port),
    )


def _boot_agent(argv: list[str], env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        argv, env=env, cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait_health(base: str, token: str, proc: subprocess.Popen) -> dict | None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            r = httpx.get(f"{base}/health",
                          headers={"Authorization": f"Bearer {token}"},
                          timeout=2.0)
        except httpx.HTTPError:
            r = None
        if r is not None and r.status_code == 200:
            return r.json()
        time.sleep(0.2)
    return None


def _drain(proc: subprocess.Popen) -> str:
    """Bounded output capture: read() on a LIVE process deadlocks (EOF waits
    for the process to exit, exit waits for the test to finish), so drain
    only after the process is gone."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return proc.stdout.read() if proc.stdout else ""


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize("dockerfile",
                         ["images/Dockerfile.cuda", "images/Dockerfile.rocm"])
def test_dockerfile_entrypoint_serves_health(tmp_path: Path,
                                             dockerfile: str) -> None:
    argv = _entrypoint_argv(REPO / dockerfile)
    assert "python" in Path(argv[0]).name, f"argv[0] is the venv python: {argv}"
    argv = [sys.executable] + argv[1:]  # the test venv, not the image's
    port = _free_port()
    token = "entrypoint-test-token"
    workdir = tmp_path / "workdir"
    (workdir / "models" / "openvla-7b").mkdir(parents=True)
    proc = _boot_agent(argv, _agent_env(tmp_path, port, token, workdir))
    try:
        body = _wait_health(f"http://127.0.0.1:{port}", token, proc)
        out = _drain(proc)
        assert body is not None, (
            f"{dockerfile} ENTRYPOINT {argv[1:]} did not serve /health; "
            f"output: {out!r}")
        assert body["image"] == "robovla-runner-cuda:test"
        assert body["disk_free_gb"] > 0
        assert body["model_dir"] == "models/openvla-7b"
        # CPU CI has no GPU: the probe degrades to facts (gpu null), not a 500
        assert body["gpu"] is None or isinstance(body["gpu"], str)
    finally:
        _stop(proc)


def test_entrypoint_dispatch_uses_configured_job_bin(tmp_path: Path) -> None:
    """ROBOVLA_JOB_BIN must reach the spawned argv (agent.config documents
    the override; the first real dispatch on host65 showed the agent
    spawning the image default /opt/robovla/bin instead — config dead-end)."""
    from robovla.core.spec import build_job_spec

    port = _free_port()
    token = "entrypoint-dispatch-token"
    workdir = tmp_path / "workdir"
    (workdir / "models" / "openvla-7b").mkdir(parents=True)
    # a trivial job program: the dispatch wiring is what is under test
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "robovla_job").write_text("import sys\nsys.exit(0)\n")
    proc = _boot_agent([sys.executable, "-m", "agent"],
                       _agent_env(tmp_path, port, token, workdir))
    try:
        base = f"http://127.0.0.1:{port}"
        assert _wait_health(base, token, proc) is not None, _drain(proc)
        spec = build_job_spec(
            job_id=JOB_ID, student="s00", job_type="smoke",
            dataset="s01/stop-at-target/v1", node_id="avc",
            node_agent=f"127.0.0.1:{port}", budget_gb=18.0, batch=4
        ).to_dict()
        auth = {"Authorization": f"Bearer {token}"}
        r = httpx.post(f"{base}/jobs", json={"spec": spec}, headers=auth,
                       timeout=5.0)
        assert r.status_code == 200, f"submit: {r.status_code} {r.text}"
        deadline = time.monotonic() + 30
        state = None
        while time.monotonic() < deadline:
            v = httpx.get(f"{base}/jobs/{JOB_ID}", headers=auth,
                          timeout=5.0).json()
            state = v["state"]
            if state in ("done", "failed"):
                break
            time.sleep(0.2)
        argv = json.loads((workdir / "runs" / JOB_ID /
                           "argv.json").read_text())
        assert argv[1] == f"{bin_dir}/robovla_job", (
            f"job_bin override not wired: agent spawned {argv[1]!r}")
        assert state == "done", (
            f"trivial job should exit done via the configured bin; state={state}")
    finally:
        _stop(proc)
