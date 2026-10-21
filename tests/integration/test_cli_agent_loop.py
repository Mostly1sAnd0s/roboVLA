"""``cli gpu_selftest`` against a REAL agent (design line 503:
``uv run robovla cli gpu_selftest --node pai1``).

Same harness as the console<->agent loop, minus the console: the CLI builds
the JobSpec (shared validator, system student s00), ``HttpRunner`` speaks the
:8601 protocol — with only the *transport* swapped for a MockTransport
delegation, and the agent runs a REAL ``fake_job`` subprocess under its
supervisor. Self-contained lab mode: the agent has no ``console_url``, so no
bundle pull — the provisioning job runs on local data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.config import AgentConfig
from agent.jobs import JobRunner
from agent.server import LogRing, build_app
from agent.serve import SlotServing
from agent.vram import VramProbe
from robovla import cli
from robovla.runner import HttpRunner

URL = "http://10.0.0.9:8601"
TOKEN = "tok-agent"
JOB = "j_20261021_0001"
DS = "s01/stop-at-target/v1"
FAKE_JOB = Path(__file__).parents[1] / "fixtures" / "fake_job.py"


class _NullModel:
    """Serving model (inference is not the point here)."""

    def predict(self, image_b64, prompt):
        return [0.5] * 7

    def unload(self):
        pass


def _canned_probe() -> VramProbe:
    def run(argv, **kw):
        if "query-gpu" in " ".join(argv):
            return subprocess.CompletedProcess(
                argv, 0, stdout="NVIDIA GeForce RTX 5090, 32608, 2560\n")
        return subprocess.CompletedProcess(argv, 0, stdout="")

    return VramProbe("cuda", runner=run)


def _lan(agent: TestClient) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path + (f"?{request.url.query}"
                                   if request.url.query else "")
        r = agent.request(request.method, path, headers=dict(request.headers),
                          content=request.content)
        return httpx.Response(r.status_code, content=r.content, headers=r.headers)

    return httpx.Client(base_url=URL, transport=httpx.MockTransport(handler))


@pytest.fixture()
def agent_env(tmp_path, monkeypatch):
    """Real agent app + real JobRunner + a socket-free HttpRunner seam."""
    workdir = tmp_path / "work"
    (workdir / "models" / "openvla-7b").mkdir(parents=True)
    log = LogRing()
    cfg = AgentConfig(token=TOKEN, workdir=str(workdir),
                      model_dir=str(workdir / "models" / "openvla-7b"),
                      image_tag="robovla-runner-cuda:dev", version="0.1.0",
                      console_url="", gpu="cuda")
    jobs = JobRunner(workdir,
                     argv_builder=lambda s, jd: [sys.executable, str(FAKE_JOB),
                                                 "--spec", str(Path(jd) / "spec.json")],
                     grace_s=1.0, log=log.log)
    app = TestClient(build_app(cfg, jobs=jobs, probe=_canned_probe(),
                               serving=SlotServing(lambda r, a: _NullModel()),
                               log=log))
    monkeypatch.setattr(cli, "HttpRunner",
                        lambda url, token: HttpRunner(URL, token, client=_lan(app)))
    return workdir, jobs, log


def test_cli_gpu_selftest_drives_real_agent(agent_env, capsys):
    workdir, jobs, log = agent_env
    rc = cli.main(["cli", "gpu_selftest", "--node", "pai1", "--dataset", DS,
                   "--agent-url", URL, "--token", TOKEN,
                   "--job-id", JOB, "--timeout", "60"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"{JOB}: done" in out

    # The agent really ran the subprocess: supervisor registry + the spec it
    # wrote + its own service log.
    assert jobs.known_job(JOB)
    spec = json.loads((workdir / "runs" / JOB / "spec.json").read_text())
    assert spec["type"] == "gpu_selftest"
    assert spec["student"] == cli.SYSTEM_STUDENT  # provisioning, not class work
    assert spec["dataset"] == DS
    assert any("running pid=" in ln for ln in log.tail(500).splitlines())


def test_cli_gpu_selftest_wrong_token_is_auth_error(agent_env, capsys):
    rc = cli.main(["cli", "gpu_selftest", "--node", "pai1", "--dataset", DS,
                   "--agent-url", URL, "--token", "wrong-token",
                   "--job-id", JOB, "--timeout", "60"])
    assert rc == 1
    assert "auth" in capsys.readouterr().err
