"""server — the console production bootstrap (fleet.yaml → ConsoleService →
uvicorn). Every I/O is a seam: fleet file read, per-node agent tokens from
the host env (fleet ``token_ref`` names — plaintext never in fleet.yaml,
§9.8), uvicorn itself. These tests use a fake uvicorn (no sockets) and tmp
fleet files; the HttpRunner base URL/token come from the validated fleet.
"""

import copy
from pathlib import Path

import pytest
import yaml

from robovla import server
from robovla.core.errors import RobovlaError
from robovla.runner import FakeRunner

# One-node fleet for the seam tests (the full shipped fleet is pinned in
# test_config.TestShippedFleetFile against the spec fixture).
FLEET = {
    "version": 1,
    "console": {
        "bind": "127.0.0.1",
        "port": 8501,
        "tls": False,
        "data_dir": "/tmp/robovla-data",
        "session_ttl_h": 12,
        "limits": {
            "student": {
                "max_running": 1,
                "max_queued": 2,
                "max_dataset_bytes": 2147483648,
                "max_run_min": 180,
                "run_now": True,
            },
            "admin_rate": "60/min",
        },
        "retention": {
            "run_logs_per_student": 20,
            "merged_export_days": 14,
            "backup": {"enabled": True, "at": "02:30", "dest": "/tmp/robovla-backups"},
        },
    },
    "nodes": [
        {
            "id": "pai1",
            "label": "pai1 (RTX 5090, primary)",
            "image": "robovla-runner-cuda:2026.10",
            "workdir": "/home/adam/robovla",
            "agent": {"lan_ip": "10.0.0.10", "port": 8601, "token_ref": "pai1-agent"},
            "gpu": {
                "vram_gb": 32,
                "quant": "bnb8",
                "train_slots": 1,
                "infer_slots": 2,
                "budgets": {"train_gb": 18, "infer_gb": 11},
            },
            "presets": {"batch": [1, 4, 8]},
            "priority": 0,
            "student_selectable": True,
            "enabled": True,
        },
    ],
}


def _fleet_file(tmp_path: Path, data_dir: Path) -> Path:
    d = copy.deepcopy(FLEET)
    d["console"]["data_dir"] = str(data_dir)
    p = tmp_path / "fleet.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def _args(tmp_path: Path, **kw):
    base = dict(fleet=str(tmp_path / "fleet.yaml"), data_dir=None, bind=None,
                port=None)
    base.update(kw)
    return argparse_ns(base)


def argparse_ns(d):  # a Namespace without importing argparse in every test
    import argparse
    return argparse.Namespace(**d)


class _Uvicorn:
    """The uvicorn seam: capture what serve() would hand the real one."""

    def __init__(self):
        self.calls: list[tuple] = []

    def run(self, app, host, port, **kw):
        self.calls.append((app, host, port))


class TestLoadFleet:
    def test_missing_file_is_not_found(self, tmp_path):
        with pytest.raises(RobovlaError) as e:
            server.load_fleet(tmp_path / "nope.yaml")
        assert e.value.code == "not_found"

    def test_malformed_yaml_is_validation(self, tmp_path):
        p = tmp_path / "fleet.yaml"
        p.write_text("console: [")
        with pytest.raises(RobovlaError) as e:
            server.load_fleet(p)
        assert e.value.code == "validation"

    def test_good_file_parses(self, tmp_path):
        p = _fleet_file(tmp_path, tmp_path / "data")
        f = server.load_fleet(p)
        assert f.console.port == 8501 and f.console.bind == "127.0.0.1"
        assert [n.id for n in f.nodes] == ["pai1"]


class TestBuildService:
    def test_http_runner_per_node_with_env_token(self, tmp_path):
        fleet = server.load_fleet(_fleet_file(tmp_path, tmp_path / "data"))
        svc = server.build_service(fleet, env={"pai1-agent": "sekrit"})
        assert set(svc.runners) == {"pai1"}
        r = svc.runners["pai1"]
        assert r._client.headers["Authorization"] == "Bearer sekrit"
        assert str(r._client.base_url) == "http://10.0.0.10:8601"
        assert svc.storage.path == tmp_path / "data" / "state.db"
        svc.storage.close()

    def test_missing_token_env_rejected_and_named(self, tmp_path):
        fleet = server.load_fleet(_fleet_file(tmp_path, tmp_path / "data"))
        with pytest.raises(RobovlaError) as e:
            server.build_service(fleet, env={})
        assert e.value.code == "validation"
        assert "pai1-agent" in e.value.message

    def test_empty_token_env_rejected(self, tmp_path):
        fleet = server.load_fleet(_fleet_file(tmp_path, tmp_path / "data"))
        with pytest.raises(RobovlaError) as e:
            server.build_service(fleet, env={"pai1-agent": "  "})
        assert e.value.code == "validation"

    def test_explicit_runners_skip_env_tokens(self, tmp_path):
        fleet = server.load_fleet(_fleet_file(tmp_path, tmp_path / "data"))
        fake = FakeRunner()
        svc = server.build_service(fleet, runners={"pai1": fake}, env={})
        assert svc.runners["pai1"] is fake
        svc.storage.close()

    def test_data_dir_override(self, tmp_path):
        fleet = server.load_fleet(_fleet_file(tmp_path, tmp_path / "data"))
        svc = server.build_service(
            fleet, data_dir=tmp_path / "alt", env={"pai1-agent": "t"})
        assert svc.storage.path == tmp_path / "alt" / "state.db"
        svc.storage.close()


class TestServe:
    def test_binds_fleet_defaults_and_wires_service(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        _fleet_file(tmp_path, data)
        monkeypatch.setenv("pai1-agent", "t")
        uv = _Uvicorn()
        rc = server.serve(_args(tmp_path), uvicorn=uv)
        assert rc == 0 and len(uv.calls) == 1
        app, host, port = uv.calls[0]
        assert host == "127.0.0.1" and port == 8501
        assert app.state.svc.storage.path == data / "state.db"
        assert app.state.svc.runners["pai1"]._client.headers[
            "Authorization"] == "Bearer t"

    def test_cli_bind_port_override(self, tmp_path, monkeypatch):
        _fleet_file(tmp_path, tmp_path / "data")
        monkeypatch.setenv("pai1-agent", "t")
        uv = _Uvicorn()
        rc = server.serve(_args(tmp_path, bind="0.0.0.0", port=9999),
                          uvicorn=uv)
        assert rc == 0
        _, host, port = uv.calls[0]
        assert (host, port) == ("0.0.0.0", 9999)

    def test_serves_the_repo_ui_when_present(self, tmp_path, monkeypatch):
        _fleet_file(tmp_path, tmp_path / "data")
        monkeypatch.setenv("pai1-agent", "t")
        uv = _Uvicorn()
        server.serve(_args(tmp_path), uvicorn=uv)
        app = uv.calls[0][0]
        # /ui is mounted (the repo's ui/ dir exists at test time)
        assert "/ui" in [getattr(r, "path", None) for r in app.routes]

    def test_missing_fleet_file_propagates_not_found(self, tmp_path):
        uv = _Uvicorn()
        with pytest.raises(RobovlaError) as e:
            server.serve(_args(tmp_path), uvicorn=uv)
        assert e.value.code == "not_found"
        assert uv.calls == []

    def test_env_tokens_come_from_os_environ(self, tmp_path, monkeypatch):
        _fleet_file(tmp_path, tmp_path / "data")
        monkeypatch.setenv("pai1-agent", "env-token")
        uv = _Uvicorn()
        rc = server.serve(_args(tmp_path), uvicorn=uv)
        assert rc == 0
        app = uv.calls[0][0]
        assert app.state.svc.runners["pai1"]._client.headers[
            "Authorization"] == "Bearer env-token"
