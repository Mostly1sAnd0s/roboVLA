"""Agent configuration — all from the container environment (design §9.8/§10).

Hardware and endpoints live in config, never in code: the systemd unit /
quadlet sets these env vars per node. No node name, IP, or token appears
anywhere in the agent source.

    ROBOVLA_AGENT_TOKEN    pre-shared per-node token (shown once at node add;
                           same plaintext the console holds for the reverse
                           bundle-pull check, §9.8/§9.9). REQUIRED.
    ROBOVLA_WORKDIR        host bind mount with the standard node layout
                           (datasets/ runs/ artifacts/ models/; default
                           /home/adam/robovla)
    ROBOVLA_MODEL_DIR      base-model dir under workdir (join probe: "model
                           dir present", §8.2)
    ROBOVLA_IMAGE_TAG      self-reported image tag (join probe: "image tag
                           matches", §8.2)
    ROBOVLA_CONSOLE_URL    console base URL for the bundle pull ("", or unset
                           => pull disabled; the job runs on what's local)
    ROBOVLA_GPU            cuda | rocm (selects the smi probe; default cuda)
    ROBOVLA_AGENT_BIND     default 0.0.0.0 (the container's port is
                           console-only, §9.8)
    ROBOVLA_AGENT_PORT     default 8601
    ROBOVLA_AGENT_LOG      agent service log file ("" => in-memory ring only)
    ROBOVLA_KILL_GRACE_S   SIGTERM grace before SIGKILL (§8.3; default 10)
    ROBOVLA_JOB_BIN        image bin dir holding the job programs (default
                           /opt/robovla/bin)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from robovla.core.errors import RobovlaError

from .argv import DEFAULT_JOB_BIN

DEFAULT_WORKDIR = "/home/adam/robovla"
DEFAULT_MODEL_REL = "models/openvla-7b"  # relative to workdir
DEFAULT_IMAGE_TAG = "robovla-runner-cuda:dev"
DEFAULT_GPU = "cuda"
GPUS = frozenset({"cuda", "rocm"})


@dataclass(frozen=True)
class AgentConfig:
    token: str
    workdir: str
    model_dir: str
    image_tag: str
    version: str
    console_url: str
    gpu: str
    bind: str = "0.0.0.0"
    port: int = 8601
    log_path: str = ""
    kill_grace_s: float = 10.0
    job_bin: str = DEFAULT_JOB_BIN


def _fail(message: str) -> None:
    raise RobovlaError("validation", message)


def _env_int(value: str, name: str, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        _fail(f"{name}: {value!r} is not an integer")
    if not (lo <= n <= hi):
        _fail(f"{name}: must be in [{lo}, {hi}]")
    return n


def _env_float(value: str, name: str, exclusive_lo: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        _fail(f"{name}: {value!r} is not a number")
    if not f > exclusive_lo:
        _fail(f"{name}: must be > {exclusive_lo:g}")
    return f


def from_env(env: Mapping[str, str] | None = None) -> AgentConfig:
    """Build the config from an env mapping (default: ``os.environ``)."""
    e = os.environ if env is None else env
    token = e.get("ROBOVLA_AGENT_TOKEN", "").strip()
    if not token:
        _fail("ROBOVLA_AGENT_TOKEN is required (per-node pre-shared token)")
    workdir = e.get("ROBOVLA_WORKDIR", DEFAULT_WORKDIR).strip() or DEFAULT_WORKDIR
    model_dir = e.get("ROBOVLA_MODEL_DIR", "").strip() or os.path.join(
        workdir, DEFAULT_MODEL_REL)
    gpu = e.get("ROBOVLA_GPU", DEFAULT_GPU).strip() or DEFAULT_GPU
    if gpu not in GPUS:
        _fail(f"ROBOVLA_GPU: must be one of {sorted(GPUS)}")
    port = _env_int(e.get("ROBOVLA_AGENT_PORT", "8601"), "ROBOVLA_AGENT_PORT",
                    1, 65535)
    grace = _env_float(e.get("ROBOVLA_KILL_GRACE_S", "10"), "ROBOVLA_KILL_GRACE_S")
    from . import __version__
    return AgentConfig(
        token=token,
        workdir=workdir,
        model_dir=model_dir,
        image_tag=e.get("ROBOVLA_IMAGE_TAG", DEFAULT_IMAGE_TAG).strip() or DEFAULT_IMAGE_TAG,
        version=__version__,
        console_url=e.get("ROBOVLA_CONSOLE_URL", "").rstrip("/"),
        gpu=gpu,
        bind=e.get("ROBOVLA_AGENT_BIND", "0.0.0.0").strip() or "0.0.0.0",
        port=port,
        log_path=e.get("ROBOVLA_AGENT_LOG", ""),
        kill_grace_s=grace,
        job_bin=e.get("ROBOVLA_JOB_BIN", DEFAULT_JOB_BIN).strip() or DEFAULT_JOB_BIN,
    )
