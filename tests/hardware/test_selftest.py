"""Hardware-gated selftest rows (design §11: ``GPU_TESTS=1``, manual).

Run on a GPU node after building the image (the tag is pinned per node in
fleet.yaml; default names below, overridable via env):

    GPU_TESTS=1 uv run pytest tests/hardware -m hardware

These drive the REAL job program inside the REAL image (GPU passthrough +
model volume, fixed argv, no shell) — the Phase 1 gate: the stack pins are
finalized when ``gpu_selftest`` passes on the target node (§6, §12). The
official crosscheck row runs the same way with ``type: crosscheck`` once the
openvla pin is verified.

Prereqs (skipped with a reason when absent, so a CPU box with GPU_TESTS=1
degrades to skips, not failures): the podman image present, and the base
model in the model dir — the dir must be the HF hub-cache layout (contains
``models--openvla--openvla-7b/``), populated by ``huggingface-cli download
openvla/openvla-7b --cache-dir <dir>`` or by an earlier online job
(docs/runner-image.md). The job runs with ``HF_HUB_CACHE=/models`` so the
engine loads the base model offline from the mounted volume.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).parents[2]
IMAGE_CUDA = os.environ.get("ROBOVLA_IMAGE_CUDA", "robovla-runner-cuda:2026.10")
IMAGE_ROCM = os.environ.get("ROBOVLA_IMAGE_ROCM", "robovla-runner-rocm:2026.10")
MODEL_DIR = pathlib.Path(os.environ.get(
    "ROBOVLA_MODEL_DIR",
    "~/.cache/huggingface/hub")).expanduser()  # the HF hub-cache dir
WORK_DIR = pathlib.Path(os.environ.get(
    "ROBOVLA_WORK_DIR", "~/robovla-selftest")).expanduser()

needs_gpu = pytest.mark.skipif(
    os.environ.get("GPU_TESTS") != "1",
    reason="manual hardware row: set GPU_TESTS=1 (design §11)")


def _image_present(tag: str) -> bool:
    r = subprocess.run(["podman", "image", "exists", tag],
                       capture_output=True, text=True)
    return r.returncode == 0


def _spec(job_type: str) -> dict:
    from robovla.core.spec import build_job_spec

    return build_job_spec(
        job_id="j_20261021_0001", student="s00", job_type=job_type,
        dataset="s01/stop-at-target/v1", node_id="pai1",
        node_agent="10.0.0.9:8601", budget_gb=18.0).to_dict()


def _run_job(image: str, spec: dict) -> subprocess.CompletedProcess:
    """The job program, in-container: GPU devices + workdir + model volume,
    the agent's exact fixed argv (python + /opt/robovla/bin/robovla_job)."""
    job_dir = WORK_DIR / "runs" / spec["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "spec.json").write_text(json.dumps(spec))
    argv = [
        "podman", "run", "--rm",
        "--device", "/dev/kfd", "--device", "/dev/dri",
        "--cap-add", "SYS_PTRACE", "--security-opt", "seccomp=unconfined",
        "--volume", f"{WORK_DIR}:/data",
        "--volume", f"{MODEL_DIR}:/models:ro",
        "-e", "ROBOVLA_WORKDIR=/data",
        "-e", "ROBOVLA_MODEL_DIR=/models",
        "-e", "HF_HUB_CACHE=/models",
        image, "/opt/venv/bin/python", "/opt/robovla/bin/robovla_job",
        "--spec", f"/data/runs/{spec['job_id']}/spec.json",
    ]
    return subprocess.run(argv, capture_output=True, text=True)


def _prereqs(image: str, job_type: str) -> None:
    if not _image_present(image):
        pytest.skip(f"image {image} not present (podman build -f images/"
                    f"Dockerfile.{ 'cuda' if 'cuda' in image else 'rocm' } .)")
    if not MODEL_DIR.is_dir():
        pytest.skip(f"model not downloaded to {MODEL_DIR} "
                    "(robovla cli model_download)")
    # selftests need no dataset pulled; a missing episodes/ is tolerated
    # (predict/QLoRA sections skip, per the engine's contract)


def _assert_selftest_ok(spec: dict) -> None:
    job_dir = WORK_DIR / "runs" / spec["job_id"]
    probe = json.loads((job_dir / "probe.json").read_text())
    assert len(probe) == 7 and all(0.0 <= v <= 1.0 for v in probe)
    rows = [json.loads(l) for l in (job_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows and all(r["total"] == 10 for r in rows)
    assert rows[-1]["step"] == 10  # the 10-step QLoRA check completed


@needs_gpu
def test_gpu_selftest_cuda():
    _prereqs(IMAGE_CUDA, "gpu_selftest")
    spec = _spec("gpu_selftest")
    cp = _run_job(IMAGE_CUDA, spec)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    _assert_selftest_ok(spec)


@needs_gpu
def test_home_selftest_rocm():
    _prereqs(IMAGE_ROCM, "home_selftest")
    spec = _spec("home_selftest")
    cp = _run_job(IMAGE_ROCM, spec)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    _assert_selftest_ok(spec)
