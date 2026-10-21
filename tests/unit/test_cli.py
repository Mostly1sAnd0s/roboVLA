"""cli.py — ``robovla cli`` (design line 488/503): backup, restore,
gpu_selftest (via the agent), model_download, node bootstrap.

Unit tests run the entrypoint directly (``cli.main([...])``) with fakes at
the seams: backup/restore over real temp dirs (the shared backup seam),
gpu_selftest against ``FakeRunner`` (the oracle), model_download with an
injected fetcher, node bootstrap over a temp workdir. The CLI never talks to
a real network or GPU here.
"""

from __future__ import annotations

import pytest

from robovla import backup, cli
from robovla.core.errors import RobovlaError
from robovla.runner.fake import FakeRunner, DONE, FAILED
from robovla.storage import Storage

T0 = "2026-10-21T09:00:00Z"
JOB = "j_20261021_0001"
DS = "s01/stop-at-target/v1"


# -- backup / restore -------------------------------------------------------------


def test_backup_command(tmp_path, capsys):
    data = tmp_path / "data"
    s = Storage(data / "state.db")
    s.set_kv("k", "v")
    s.close()
    (data / "datasets" / "s01" / "x" / "v1").mkdir(parents=True)
    (data / "datasets" / "s01" / "x" / "v1" / "ep.json").write_bytes(b"e")

    rc = cli.main(["cli", "backup", "--data-dir", str(data),
                   "--dest", str(tmp_path / "bk")])
    assert rc == 0
    s2 = Storage(data / "state.db")
    rows = s2.list_backups()
    s2.close()
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    out = capsys.readouterr().out
    assert rows[0]["dest"].rsplit("/", 1)[-1] in out  # reports the bundle name


def test_restore_command_roundtrip(tmp_path):
    data = tmp_path / "data"
    s = Storage(data / "state.db")
    s.set_kv("k", "v")
    s.close()
    info = backup.make_backup_bundle(data, tmp_path / "bk",
                                     clock=lambda: T0)
    rc = cli.main(["cli", "restore", "--bundle", info["path"],
                   "--data-dir", str(tmp_path / "fresh")])
    assert rc == 0
    s2 = Storage(tmp_path / "fresh" / "state.db")
    assert s2.get_kv("k") == "v"
    s2.close()


def test_restore_command_conflict_needs_force(tmp_path, capsys):
    data = tmp_path / "data"
    Storage(data / "state.db").close()
    info = backup.make_backup_bundle(data, tmp_path / "bk", clock=lambda: T0)
    fresh = tmp_path / "fresh"
    s = Storage(fresh / "state.db")
    s.close()

    assert cli.main(["cli", "restore", "--bundle", info["path"],
                     "--data-dir", str(fresh)]) == 1
    assert "force" in capsys.readouterr().err
    assert cli.main(["cli", "restore", "--bundle", info["path"],
                     "--data-dir", str(fresh), "--force"]) == 0


# -- node bootstrap ----------------------------------------------------------------


def test_node_bootstrap_layout(tmp_path):
    wd = tmp_path / "node"
    assert cli.main(["cli", "node", "bootstrap", "--workdir", str(wd)]) == 0
    for sub in ("datasets", "runs", "artifacts", "models"):
        assert (wd / sub).is_dir()
    assert cli.main(["cli", "node", "bootstrap", "--workdir", str(wd)]) == 0  # idempotent
    # explicit model dir outside the default
    md = tmp_path / "elsewhere" / "models"
    assert cli.main(["cli", "node", "bootstrap", "--workdir", str(wd),
                     "--model-dir", str(md)]) == 0
    assert md.is_dir()


def test_node_bootstrap_refuses_file_workdir(tmp_path, capsys):
    f = tmp_path / "f"
    f.write_bytes(b"x")
    assert cli.main(["cli", "node", "bootstrap", "--workdir", str(f)]) == 1
    assert "validation" in capsys.readouterr().err or "file" in capsys.readouterr().err


# -- model_download ------------------------------------------------------------------


def test_model_download_with_fake_fetch(tmp_path):
    calls: list[tuple] = []

    def fetch(url: str, path) -> int:
        calls.append((url, path))
        from pathlib import Path
        Path(path).write_bytes(b"weights")
        return 9

    out = cli.download_model("https://h/r/openvla-7b.bin",
                             tmp_path / "models", fetch=fetch)
    assert out == {"path": str(tmp_path / "models" / "openvla-7b.bin"), "nbytes": 9}
    assert calls[0][0] == "https://h/r/openvla-7b.bin"
    with pytest.raises(RobovlaError) as ei:
        cli.download_model("https://h/r/openvla-7b.bin", tmp_path / "models",
                           fetch=fetch)
    assert ei.value.code == "conflict"  # existing file: teacher removes it first


def test_model_download_explicit_name(tmp_path):
    out = cli.download_model("https://h/r/model", tmp_path / "m", name="base.safetensors",
                             fetch=lambda u, p: (open(p, "wb").write(b"x"), 1)[1])
    assert out["path"].endswith("m/base.safetensors")


# -- gpu_selftest (FakeRunner = the oracle) --------------------------------------------


def _spec_fields(runner: FakeRunner, job_id: str = JOB) -> dict:
    return runner.submitted_specs[job_id]


def test_gpu_selftest_success_and_spec_shape():
    runner = FakeRunner(online=True)
    runner.script(JOB, steps=3, final_state=DONE, val_accuracy=0.91)
    rc = cli.run_selftest(runner=runner, node_id="pai1", dataset=DS,
                          agent="10.0.0.9:8601", job_id=JOB,
                          timeout_s=30, poll_s=0.001)
    assert rc == 0
    spec = _spec_fields(runner)
    assert spec["type"] == "gpu_selftest"
    assert spec["student"] == "s00"  # the reserved system student
    assert spec["dataset"] == DS
    assert spec["node"] == {"id": "pai1", "agent": "10.0.0.9:8601"}
    assert spec["limits"]["budget_gb"] > 0


def test_gpu_selftest_failure_returns_1():
    runner = FakeRunner()
    runner.script(JOB, steps=1, final_state=FAILED, error="no CUDA device")
    rc = cli.run_selftest(runner=runner, node_id="pai1", dataset=DS,
                          agent="10.0.0.9:8601", job_id=JOB,
                          timeout_s=30, poll_s=0.001)
    assert rc == 1


def test_gpu_selftest_offline_returns_1():
    rc = cli.run_selftest(runner=FakeRunner(online=False), node_id="pai1",
                          dataset=DS, agent="10.0.0.9:8601", job_id=JOB,
                          timeout_s=30, poll_s=0.001)
    assert rc == 1


def test_gpu_selftest_duplicate_job_id_is_conflict():
    runner = FakeRunner()
    runner.script(JOB, steps=1)
    cli.run_selftest(runner=runner, node_id="pai1", dataset=DS,
                     agent="10.0.0.9:8601", job_id=JOB,
                     timeout_s=30, poll_s=0.001)
    with pytest.raises(RobovlaError) as ei:
        cli.run_selftest(runner=runner, node_id="pai1", dataset=DS,
                         agent="10.0.0.9:8601", job_id=JOB,
                         timeout_s=30, poll_s=0.001)
    assert ei.value.code == "conflict"


# -- gpu_selftest argument wiring (no network) ------------------------------------------


def test_gpu_selftest_requires_agent_endpoint(capsys, monkeypatch):
    monkeypatch.delenv("ROBOVLA_AGENT_TOKEN", raising=False)
    rc = cli.main(["cli", "gpu_selftest", "--node", "pai1", "--dataset", DS,
                   "--token", "t"])
    assert rc == 1
    assert "--agent-url" in capsys.readouterr().err


def test_gpu_selftest_requires_token(capsys, monkeypatch):
    monkeypatch.delenv("ROBOVLA_AGENT_TOKEN", raising=False)
    rc = cli.main(["cli", "gpu_selftest", "--node", "pai1", "--dataset", DS,
                   "--agent-url", "http://10.0.0.9:8601"])
    assert rc == 1
    assert "token" in capsys.readouterr().err


def test_gpu_selftest_bad_dataset_is_validation(capsys, monkeypatch):
    monkeypatch.delenv("ROBOVLA_AGENT_TOKEN", raising=False)
    rc = cli.main(["cli", "gpu_selftest", "--node", "pai1", "--dataset", "nope",
                   "--agent-url", "http://10.0.0.9:8601", "--token", "t"])
    assert rc == 1
    assert "dataset" in capsys.readouterr().err


# -- entrypoint shape ---------------------------------------------------------------------


def test_main_no_command_exits_2():
    with pytest.raises(SystemExit) as ei:
        cli.main([])
    assert ei.value.code == 2
