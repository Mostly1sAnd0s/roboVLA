"""Hardware-gated SmolVLA smoke row (the Pascal box, design §11 pattern).

The SmolVLA family's image gate — the pin rule: a new image tag is
UNTRUSTED until this smoke passes on the target silicon. It drives the
REAL wrapper (images/smolvla_finetune.py) inside the REAL image (GPU
passthrough + model volume, fixed argv, no shell): dataset
materialization from the shared selftest fixture, the seeded split, one
official lerobot-train step with the house trackio sink, the checkpoint
layout, the serving.json stamp, and the artifact manifest.

Run on the Pascal node (GTX 1070) after building the image:

    GPU_TESTS=1 ROBOVLA_IMAGE_SMOLVLA_PASCAL=\
        robovla-runner-smolvla-cuda-pascal:2026.11.0 \
        uv run pytest tests/hardware/test_smolvla.py -m hardware

Prereqs (skipped with a reason when absent — the CPU box degrades to
skips, not failures): the podman image present, and BOTH model ids of
the SmolVLA offline pair in the model dir (the HF hub-cache layout —
``models--lerobot--smolvla_base/`` AND
``models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/``; the loader
pulls the backbone from within from_pretrained).

Why fp32 (handoff 2026-11-08, agreed with the owner): Pascal sm_61 has
no bf16 kernels (hard "not implemented for BFloat16") and crippled
fp16 math; the row's spec therefore carries quant="fp32" — the family-
gated profile the wrapper maps to --accelerator.mixed_precision "no"
(YAML-quoted: lerobot parses CLI values through YAML, where a bare
`no` is the boolean False — pinned in test_smolvla_finetune). The 1070
is a SmolVLA-ONLY box: openvla-7b does not fit 8 GB.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).parents[2]
IMAGE = os.environ.get(
    "ROBOVLA_IMAGE_SMOLVLA_PASCAL",
    "robovla-runner-smolvla-cuda-pascal:2026.11.0")
DATASET = "s01/smolvla-smoke/v1"
N_EP, N_FRAMES = 6, 8  # the shared selftest fixture shape
MODEL_DIR = pathlib.Path(os.environ.get(
    "ROBOVLA_MODEL_DIR", "~/.cache/huggingface/hub")).expanduser()
WORK_DIR = pathlib.Path(os.environ.get(
    "ROBOVLA_WORK_DIR", "~/robovla-selftest")).expanduser()

needs_gpu = pytest.mark.skipif(
    os.environ.get("GPU_TESTS") != "1",
    reason="manual hardware row: set GPU_TESTS=1 (design §11)")

HF_MODEL_DIRS = ("models--lerobot--smolvla_base",
                 "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct")


def _image_present(tag: str) -> bool:
    r = subprocess.run(["podman", "image", "exists", tag],
                       capture_output=True, text=True)
    return r.returncode == 0


def _prereqs() -> None:
    if not _image_present(IMAGE):
        pytest.skip(f"image {IMAGE} not present (podman build -f images/"
                    "Dockerfile.cuda-smolvla-pascal .)")
    for d in HF_MODEL_DIRS:
        if not (MODEL_DIR / d).is_dir():
            pytest.skip(f"model volume {MODEL_DIR} lacks {d} — the smolvla "
                        "offline pair needs the base policy AND its SmolVLM2 "
                        "backbone (robovla cli model_download)")


def _materialize_dataset() -> None:
    """The deterministic synthetic dataset (scripts/make_selftest_dataset.py
    — refuses to overwrite, so idempotent). Without it the wrapper raises
    dataset_not_ready; a row that skips the data must never pass
    vacuously (the host65 lesson pinned in test_selftest.py)."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_selftest_dataset.py"),
         "--workdir", str(WORK_DIR), "--dataset", DATASET,
         "--episodes", str(N_EP), "--frames", str(N_FRAMES)],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"smoke dataset materialization failed: "
                    f"{r.stdout + r.stderr}")


def _spec() -> dict:
    from robovla.core.spec import build_job_spec

    return build_job_spec(
        job_id="j_20261108_0001", student="s00", job_type="smoke",
        dataset=DATASET, node_id="pascal1",
        node_agent="10.0.0.9:8601", budget_gb=7.0, batch=1,
        model="lerobot/smolvla_base", quant="fp32").to_dict()


@needs_gpu
def test_smoke_cuda_pascal():
    """One official lerobot-train step, in-image, fp32 on Pascal. The
    assertions are the wrapper's success contract (the CPU suite pins the
    pieces; this row proves the image carries the whole stack): exit 0,
    sink rows in the standard metrics.jsonl (the fp32 run's vram_mb is
    the number the fleet budget is tuned against), the kind-adapter
    manifest pointing at the pretrained_model dir, and the serving.json
    stamp inside it."""
    _prereqs()
    _materialize_dataset()
    spec = _spec()
    job_dir = WORK_DIR / "runs" / spec["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "spec.json").write_text(json.dumps(spec))
    cp = subprocess.run([
        "podman", "run", "--rm",
        # ENTRYPOINT override rule (test_selftest.py, host65): podman
        # appends the after-image argv to the agent ENTRYPOINT — without
        # this the job argv becomes the agent's args.
        "--entrypoint", "/opt/venv/bin/python",
        # CDI passthrough rule (test_selftest.py): the plain
        # --device /dev/nvidia* lines only work rootful.
        "--device", "nvidia.com/gpu=all",
        # SELinux rule (this box, first in-image boot 2026-11-08): with
        # the container label enforced, NVML fails "Insufficient
        # Permissions" — SELinux-unconfined GPU containers need
        # label=disable (the same flag the llama1070 launcher carries).
        "--security-opt", "label=disable",
        "--volume", f"{WORK_DIR}:/data",
        "--volume", f"{MODEL_DIR}:/models:ro",
        "-e", "ROBOVLA_WORKDIR=/data",
        "-e", "ROBOVLA_MODEL_DIR=/models",
        "-e", "HF_HUB_CACHE=/models",
        "-e", "HF_HUB_OFFLINE=1",
        IMAGE, "/opt/robovla/bin/smolvla_job",
        "--spec", f"/data/runs/{spec['job_id']}/spec.json",
    ], capture_output=True, text=True)
    (job_dir / "official_output.log").write_text(cp.stdout + cp.stderr)
    assert cp.returncode == 0, cp.stdout + cp.stderr

    # the sink: at least one house metrics row for the single step
    rows = [json.loads(l) for l in
            (job_dir / "metrics.jsonl").read_text().splitlines()
            if l.strip()]
    assert rows and rows[-1]["step"] == 1 and rows[-1]["total"] == 1
    assert isinstance(rows[-1]["loss"], float)

    # the artifact contract: kind=adapter, workdir-relative, loadable dir
    art = json.loads((job_dir / "artifacts.json").read_text())
    assert [a["kind"] for a in art] == ["adapter"]
    model_dir = WORK_DIR / art[0]["path"]
    assert not art[0]["path"].startswith("/")
    assert (model_dir / "config.json").is_file()
    assert any(f.name.endswith(".safetensors") for f in model_dir.iterdir())

    # the serving stamp inside the artifact (the §9.5 denorm pair)
    meta = json.loads((model_dir / "serving.json").read_text())
    assert meta["family"] == "smolvla" and meta["action_dim"] == 3
