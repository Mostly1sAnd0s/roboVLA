"""Hardware-gated selftest rows (design §11: ``GPU_TESTS=1``, manual).

Run on a GPU node after building the image (the tag is pinned per node in
fleet.yaml; default names below, overridable via env):

    GPU_TESTS=1 uv run pytest tests/hardware -m hardware

These drive the REAL job program inside the REAL image (GPU passthrough +
model volume, fixed argv, no shell) — the Phase 1 gate: the stack pins are
finalized when ``gpu_selftest`` passes on the target node (§6, §12). The
``test_crosscheck_cuda`` row runs the official engine the same way: the
wrapper materializes the deterministic 50-episode subset as an RLDS/TFDS
dataset and execs the pinned ``vla-scripts/finetune.py`` (long row — run
detached on a 12 GB node).

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
import sys

import pytest

REPO = pathlib.Path(__file__).parents[2]
IMAGE_CUDA = os.environ.get("ROBOVLA_IMAGE_CUDA", "robovla-runner-cuda:2026.10.1")
IMAGE_ROCM = os.environ.get("ROBOVLA_IMAGE_ROCM", "robovla-runner-rocm:2026.10")
DATASET = "s01/stop-at-target/v1"  # the deterministic selftest dataset
CROSSCHECK_DATASET = "s01/crosscheck-50/v1"  # the crosscheck row's 50-ep dataset
CROSSCHECK_EPS, CROSSCHECK_FRAMES = 50, 20
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
        dataset=DATASET, node_id="pai1",
        node_agent="10.0.0.9:8601", budget_gb=18.0).to_dict()


def _device_flags(image: str) -> list[str]:
    """GPU passthrough per platform (matches the quadlet,
    systemd/robovla-runner.service): NVIDIA = CDI (one-time root
    ``nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml``); the ROCm
    leg is plain device passthrough + unconfined seccomp."""
    if "cuda" in image:
        return ["--device", "nvidia.com/gpu=all"]
    return ["--device", "/dev/kfd", "--device", "/dev/dri",
            "--cap-add", "SYS_PTRACE", "--security-opt", "seccomp=unconfined"]


def _run_job(image: str, spec: dict, program: str = "robovla_job") -> subprocess.CompletedProcess:
    """The job program, in-container: GPU devices + workdir + model volume,
    the agent's exact fixed argv (python + /opt/robovla/bin/<program> —
    ``robovla_job`` for the engine jobs, ``finetune`` for the crosscheck)."""
    job_dir = WORK_DIR / "runs" / spec["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "spec.json").write_text(json.dumps(spec))
    argv = [
        "podman", "run", "--rm",
        # The image ENTRYPOINT is the agent (python -m agent, pinned by
        # tests/integration/test_agent_entrypoint.py). podman appends the
        # after-image argv to the ENTRYPOINT, so without this override the
        # job's argv becomes the agent's args (agent boots, exits 1 on the
        # missing ROBOVLA_AGENT_TOKEN — caught on host65, 2026-10-22).
        # Override with the job's python; the fixed argv after the image
        # name is then exec'd verbatim (the agent's exact fixed argv).
        "--entrypoint", "/opt/venv/bin/python",
        *_device_flags(image),
        "--volume", f"{WORK_DIR}:/data",
        "--volume", f"{MODEL_DIR}:/models:ro",
        "-e", "ROBOVLA_WORKDIR=/data",
        "-e", "ROBOVLA_MODEL_DIR=/models",
        "-e", "HF_HUB_CACHE=/models",
        image, f"/opt/robovla/bin/{program}",
        "--spec", f"/data/runs/{spec['job_id']}/spec.json",
    ]
    return subprocess.run(argv, capture_output=True, text=True)


def _prereqs_image(image: str) -> None:
    if not _image_present(image):
        pytest.skip(f"image {image} not present (podman build -f images/"
                    f"Dockerfile.{ 'cuda' if 'cuda' in image else 'rocm' } .)")
    if not MODEL_DIR.is_dir():
        pytest.skip(f"model not downloaded to {MODEL_DIR} "
                    "(huggingface-cli download openvla/openvla-7b --cache-dir …)")


def _materialize_dataset(dataset: str, episodes: int, frames: int, what: str) -> None:
    """The deterministic synthetic dataset in the job workdir
    (scripts/make_selftest_dataset.py — refuses to overwrite, so idempotent).
    Without it the job skips the dataset-bound sections and the row passes
    VACUOUSLY (caught on host65, 2026-10-22: job exit 0 in ~15 s, no
    probe.json)."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_selftest_dataset.py"),
         "--workdir", str(WORK_DIR), "--dataset", dataset,
         "--episodes", str(episodes), "--frames", str(frames)],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"{what} dataset materialization failed: "
                    f"{r.stdout + r.stderr}")


def _assert_selftest_ok(spec: dict) -> None:
    job_dir = WORK_DIR / "runs" / spec["job_id"]
    probe = json.loads((job_dir / "probe.json").read_text())
    assert len(probe) == 7 and all(0.0 <= v <= 1.0 for v in probe)
    rows = [json.loads(l) for l in (job_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows and all(r["total"] == 10 for r in rows)
    assert rows[-1]["step"] == 10  # the 10-step QLoRA check completed


@needs_gpu
def test_gpu_selftest_cuda():
    _prereqs_image(IMAGE_CUDA)
    _materialize_dataset(DATASET, 6, 8, "selftest")
    spec = _spec("gpu_selftest")
    cp = _run_job(IMAGE_CUDA, spec)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    _assert_selftest_ok(spec)


@needs_gpu
def test_home_selftest_rocm():
    _prereqs_image(IMAGE_ROCM)
    _materialize_dataset(DATASET, 6, 8, "selftest")
    spec = _spec("home_selftest")
    cp = _run_job(IMAGE_ROCM, spec)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    _assert_selftest_ok(spec)


def _assert_crosscheck_ok(spec: dict) -> None:
    """The file-contract manifest from the official run: adapter (LoRA) +
    merged model (bf16 full weights) under the job dir, with the official
    dataset_statistics.json (the official predict path needs it next to the
    merged model to de-normalize actions)."""
    job_dir = WORK_DIR / "runs" / spec["job_id"]
    art = json.loads((job_dir / "artifacts.json").read_text())
    assert [a["kind"] for a in art] == ["adapter", "merged"]
    adapter = (job_dir / art[0]["path"]).resolve()
    merged = (job_dir / art[1]["path"]).resolve()
    assert (adapter / "adapter_model.safetensors").is_file()
    assert (merged / "dataset_statistics.json").is_file()
    assert any(f.suffix in (".bin", ".safetensors")
               for f in merged.iterdir() if f.is_file())


@needs_gpu
def test_crosscheck_cuda():
    """The OFFICIAL engine on the deterministic 50-episode subset (the
    crosscheck rework, 2026-10-22 — gate M5's "real training verified"):
    the wrapper materializes the subset as an RLDS/TFDS dataset and runs
    the pinned vla-scripts/finetune.py (torchrun nproc 1, official 4-bit,
    WANDB_MODE=disabled). LONG row: 1000 frames // (batch 1 x grad_accum 8)
    = 125 steps/epoch x 3 epochs — on a 12 GB node run it detached (setsid
    nohup + poll the job log) and expect ~2-4 h; the 32 GB fleet nodes at
    batch 4-8 are 4-8x faster."""
    from robovla.core.spec import build_job_spec

    _prereqs_image(IMAGE_CUDA)
    _materialize_dataset(CROSSCHECK_DATASET, CROSSCHECK_EPS, CROSSCHECK_FRAMES,
                         "crosscheck")
    # batch 1: the 4070-class reality (the 32 GB fleet nodes take 4-8)
    spec = build_job_spec(
        job_id="j_20261022_0002", student="s01", job_type="crosscheck",
        dataset=CROSSCHECK_DATASET, node_id="pai1",
        node_agent="10.0.0.9:8601", budget_gb=18.0, batch=1).to_dict()
    cp = _run_job(IMAGE_CUDA, spec, program="finetune")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    _assert_crosscheck_ok(spec)
