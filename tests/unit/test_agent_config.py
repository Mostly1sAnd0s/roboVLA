"""agent.config — env-driven configuration (design §9.8/§10: config, not code)."""

from __future__ import annotations

import pytest

from agent.config import from_env
from robovla.core.errors import RobovlaError


def test_defaults():
    cfg = from_env({"ROBOVLA_AGENT_TOKEN": "tok-1"})
    assert cfg.token == "tok-1"
    assert cfg.workdir == "/home/adam/robovla"
    assert cfg.model_dir == "/home/adam/robovla/models/openvla-7b"
    assert cfg.image_tag == "robovla-runner-cuda:dev"
    assert cfg.console_url == ""          # pull disabled by default
    assert cfg.gpu == "cuda"
    assert cfg.bind == "0.0.0.0"
    assert cfg.port == 8601
    assert cfg.kill_grace_s == 10.0
    assert cfg.version == "0.1.0"


def test_full_env():
    cfg = from_env({
        "ROBOVLA_AGENT_TOKEN": "tok-2",
        "ROBOVLA_WORKDIR": "/data/robovla",
        "ROBOVLA_MODEL_DIR": "/data/robovla/models/openvla-7b",
        "ROBOVLA_IMAGE_TAG": "robovla-runner-cuda:2026.10",
        "ROBOVLA_CONSOLE_URL": "http://10.0.0.5:8501/",
        "ROBOVLA_GPU": "rocm",
        "ROBOVLA_AGENT_BIND": "192.168.1.20",
        "ROBOVLA_AGENT_PORT": "9000",
        "ROBOVLA_AGENT_LOG": "/var/log/robovla-agent.log",
        "ROBOVLA_KILL_GRACE_S": "30",
        "ROBOVLA_JOB_BIN": "/opt/stack/bin",
    })
    assert cfg.console_url == "http://10.0.0.5:8501"  # trailing slash trimmed
    assert cfg.gpu == "rocm" and cfg.port == 9000 and cfg.kill_grace_s == 30.0
    assert cfg.job_bin == "/opt/stack/bin"


def test_missing_token_is_validation():
    with pytest.raises(RobovlaError) as ei:
        from_env({})
    assert ei.value.code == "validation" and "ROBOVLA_AGENT_TOKEN" in ei.value.message


def test_bad_gpu_is_validation():
    with pytest.raises(RobovlaError) as ei:
        from_env({"ROBOVLA_AGENT_TOKEN": "t", "ROBOVLA_GPU": "metal"})
    assert ei.value.code == "validation"


def test_bad_port_and_grace_are_validation():
    for env in ({"ROBOVLA_AGENT_TOKEN": "t", "ROBOVLA_AGENT_PORT": "0"},
                {"ROBOVLA_AGENT_TOKEN": "t", "ROBOVLA_AGENT_PORT": "nope"},
                {"ROBOVLA_AGENT_TOKEN": "t", "ROBOVLA_KILL_GRACE_S": "0"},
                {"ROBOVLA_AGENT_TOKEN": "t", "ROBOVLA_KILL_GRACE_S": "-5"}):
        with pytest.raises(RobovlaError) as ei:
            from_env(env)
        assert ei.value.code == "validation"
