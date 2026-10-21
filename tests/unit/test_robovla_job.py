"""The in-image job program (images/robovla_job.py + engine + contract) —
CPU-testable up to the lazy GPU-stack boundary.

``images/`` is a root-level directory (not a wheel package): it is baked into
the runner image at /opt/robovla/bin, and these tests load the modules by
file path (the same importlib pattern the loop test uses for conftest). The
lazy-import boundary is the seam: everything up to the stack load (arg
parsing, the SHARED JobSpec validator, the file contract, dispatch, crosscheck
argv translation) runs on CPU; a broken/off-GPU stack fails with a clean
``internal`` error, never a bare traceback.

The file contract is the one the agent's supervisor reads (and the fake_job
fixture implements): metrics.jsonl rows, val.json, probe.json, artifacts.json
(workdir-relative paths + nbytes), ckpt/ under workdir/runs/<job_id>/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from robovla.core.errors import RobovlaError
from robovla.core.spec import build_job_spec
from robovla.core.split import crosscheck_subset

# The repo root hosts the `images` package (baked into the image; not in the
# wheel) — put it on the path and import it like any package.
REPO = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from images import contract, crosscheck_finetune as xf  # noqa: E402
from images import engine  # noqa: E402
from images import robovla_job as job  # noqa: E402

SPEC_PATH = "/wd/runs/j_20261021_0001/spec.json"


def _spec(job_type: str, **kw) -> dict:
    return build_job_spec(
        job_id="j_20261021_0001", student="s01", job_type=job_type,
        dataset="s01/stop-at-target/v1", node_id="pai1",
        node_agent="10.0.0.9:8601", budget_gb=18.0, **kw).to_dict()


# -- contract: the file shapes the agent supervisor reads -----------------------


def test_job_dirs_derivation():
    job_dir, workdir = contract.job_dirs(SPEC_PATH)
    assert str(job_dir) == "/wd/runs/j_20261021_0001"
    assert str(workdir) == "/wd"  # workdir/runs/<job_id>


def test_metrics_row_contract_keys(tmp_path):
    row = contract.metrics_row(3, 10, 0.9, 5e-4, 1.2, 15000, 78, 210)
    assert row == {"step": 3, "total": 10, "loss": 0.9, "lr": 5e-4,
                   "grad_norm": 1.2, "vram_mb": 15000, "vram_util_pct": 78,
                   "eta_s": 210}
    contract.write_metrics(tmp_path, row)
    with open(tmp_path / "metrics.jsonl") as f:
        assert json.loads(f.readline()) == row


def test_val_and_probe(tmp_path):
    contract.write_val(tmp_path, 0.87)
    assert json.loads((tmp_path / "val.json").read_text()) == {"accuracy": 0.87}
    contract.write_probe(tmp_path, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert len(json.loads((tmp_path / "probe.json").read_text())) == 7
    for bad in ([0.1] * 6, [float("nan")] * 7, ["x"] * 7):
        with pytest.raises(RobovlaError) as ei:
            contract.write_probe(tmp_path, bad)
        assert ei.value.code == "validation"


def test_artifacts_manifest_relative_paths(tmp_path):
    wd = tmp_path / "wd"
    art = wd / "runs" / "j_1" / "ckpt" / "adapter.safetensors"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"x" * 12)
    manifest = contract.write_artifacts(tmp_path / "runs" / "j_1", wd,
                                        [("adapter", art)])
    assert manifest == [{"kind": "adapter",
                         "path": "runs/j_1/ckpt/adapter.safetensors",
                         "nbytes": 12}]
    # escape is refused
    with pytest.raises(RobovlaError) as ei:
        contract.write_artifacts(tmp_path / "runs" / "j_1", wd,
                                 [("adapter", tmp_path / "elsewhere.bin")])
    assert ei.value.code == "validation"


def test_artifact_kind_map():
    assert contract.artifact_kind("train") == "adapter"
    assert contract.artifact_kind("validate") == "adapter"
    assert contract.artifact_kind("merge") == "merged"
    assert contract.artifact_kind("smoke") is None
    assert contract.artifact_kind("gpu_selftest") is None


# -- job program: args, validation, dispatch ------------------------------------


def test_main_missing_spec_flag():
    assert job.main([]) == 2
    assert job.main(["--nope"]) == 2
    assert job.main(["--spec"]) == 2  # flag without value


def test_main_missing_or_bad_spec_file(tmp_path, capsys):
    assert job.main(["--spec", "/nonexistent/spec.json"]) == 1
    assert "validation" in capsys.readouterr().err
    bad = tmp_path / "spec.json"
    bad.write_bytes(b"not json")
    assert job.main(["--spec", str(bad)]) == 1
    assert "validation" in capsys.readouterr().err


def test_main_schema_validation_uses_shared_validator(tmp_path, capsys):
    spec = _spec("train")
    spec["optim"]["batch"] = "four"  # valid JSON, invalid schema
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    assert job.main(["--spec", str(p)]) == 1
    err = capsys.readouterr().err
    assert "validation" in err and "batch" in err


def test_dispatch_rejects_infer_and_crosscheck():
    with pytest.raises(RobovlaError) as ei:
        job.dispatch(_spec("infer"))
    assert ei.value.code == "validation" and "served" in ei.value.message
    with pytest.raises(RobovlaError) as ei:
        job.dispatch(_spec("crosscheck"))
    assert ei.value.code == "validation"


def test_dispatch_off_gpu_fails_clean_not_traceback(tmp_path, capsys):
    """The CPU boundary: the stack import fails, but the job reports a clean
    §9.6 ``internal`` error (a broken image must not dump a traceback)."""
    for t in ("smoke", "train", "validate", "merge",
              "gpu_selftest", "home_selftest"):
        spec = _spec(t)
        p = tmp_path / "wd" / "runs" / spec["job_id"] / "spec.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec))
        assert job.main(["--spec", str(p)]) == 1
        assert "stack unavailable" in capsys.readouterr().err


# -- engine: pure helpers (binning) + lazy boundary -------------------------------


def test_action_bins_edges_and_clamp():
    assert engine.action_bins([0.0] * 7) == (0,) * 7
    assert engine.action_bins([1.0] * 7) == (255,) * 7
    assert engine.action_bins([0.5] * 7) == (128,) * 7  # midpoint bin
    assert engine.action_bins([-9.0, 9.0, 0.92, 0.08, 1.0, 0.0, 0.5]) == \
        (0, 255, 235, 20, 255, 0, 128)  # clamped to range


def test_probe_from_bins_roundtrip():
    orig = [0.0, 0.1, 0.5, 0.7, 0.9, 0.99, 1.0]
    bins = engine.action_bins(orig)
    back = engine.probe_from_bins(bins)
    assert len(back) == 7
    for o, b in zip(orig, back):
        assert b == pytest.approx(o, abs=1.0 / 255 + 1e-9)  # one bin of slack


# -- Phase 1: the official action-token mapping (pure; probe-verified on silicon) --


def test_norm_to_bin_edges_and_floor():
    # floor, not round (the official digitize semantics): 0.5 -> 127, not 128
    assert engine.norm_to_bin(0.0) == 0
    assert engine.norm_to_bin(1.0) == 255
    assert engine.norm_to_bin(0.5) == 127
    assert engine.norm_to_bin(0.5 + 1.0 / 255) == 128
    assert engine.norm_to_bin(-9.0) == 0 and engine.norm_to_bin(9.0) == 255
    # the exact round-trip the engine relies on (float-safe for all 256 bins)
    for b in range(256):
        assert engine.norm_to_bin(b / 255.0) == b


def test_token_id_roundtrip_and_clamp():
    V = 32000  # the openvla-7b vocab (Llama-2-7b-hf)
    for b in range(256):
        tid = engine.norm_to_token_id(b / 255.0, V)
        assert engine.token_id_to_bin(tid, V) == b
    # the action tokens are the last 256 vocab tokens
    assert engine.norm_to_token_id(1.0, V) == V - 256
    assert engine.norm_to_token_id(0.0, V) == V - 1
    # out-of-range ids clamp to the edge bins (the official decode clipping)
    assert engine.token_id_to_bin(0, V) == 255
    assert engine.token_id_to_bin(V, V) == 0


def test_empty_token_constant():
    # the "" token id — the official predict_action hardcodes it; the Phase 1
    # probe verified decode([29871]) == "" (tok("") is NOT a derivation)
    assert engine.EMPTY_TOKEN_ID == 29871


def test_human_msg_extraction():
    from robovla.core.spec import DEFAULT_PROMPT_TEMPLATE
    # canonical template -> the bare human message (the builder re-wraps)
    assert engine._human_msg(DEFAULT_PROMPT_TEMPLATE) == \
        "What action should the robot take to {task}?"
    # non-canonical (student-custom) templates pass through as-is
    custom = "Please tell me the action for {task}."
    assert engine._human_msg(custom) == custom


# -- crosscheck wrapper: spec -> official finetune argv ----------------------------


def _fake_dataset(tmp_path, n: int) -> pathlib.Path:
    """A dataset root the agent would have pulled: episodes/ep_NNNN/."""
    root = tmp_path / "datasets" / "s01__stop-at-target__v1"
    for i in range(n):
        (root / "episodes" / f"ep_{i:04d}").mkdir(parents=True)
    return root


def test_crosscheck_argv_and_materialized_subset(tmp_path):
    spec = _spec("crosscheck", seed=7)
    root = _fake_dataset(tmp_path, 60)
    job_dir = tmp_path / "wd" / "runs" / spec["job_id"]
    argv = xf.translate(spec, root, job_dir)

    # the official script, pinned in the image, with the shared recipe values
    assert argv[0].endswith("python") and argv[1].endswith("finetune.py")
    d = dict(zip(argv[2::2], argv[3::2]))
    assert d["--data_path"].startswith(str(job_dir))  # the materialized subset
    assert d["--pretrained_model_name_or_path"] == spec["model"]
    assert d["--max_seq_length"] == "287"
    assert d["--lora_r"] == "32" and d["--lora_alpha"] == "32"
    assert d["--learning_rate"] == str(spec["optim"]["lr"])
    assert d["--num_train_epochs"] == str(spec["optim"]["epochs"])
    assert d["--per_device_train_batch_size"] == str(spec["optim"]["batch"])

    ids = [f"ep_{i:04d}" for i in range(60)]
    expected = set(crosscheck_subset(ids, spec["student"], spec["dataset"]))
    eps_dir = pathlib.Path(d["--data_path"]) / "episodes"
    got = {p.name for p in eps_dir.iterdir()}
    assert got == expected and len(got) == 50


def test_crosscheck_smaller_dataset_copies_all(tmp_path):
    spec = _spec("crosscheck")
    root = _fake_dataset(tmp_path, 30)
    job_dir = tmp_path / "wd" / "runs" / spec["job_id"]
    argv = xf.translate(spec, root, job_dir)
    d = dict(zip(argv[2::2], argv[3::2]))
    eps = pathlib.Path(d["--data_path"]) / "episodes"
    assert len(list(eps.iterdir())) == 30


# -- dataset loading (CPU: engine._samples is pure — no GPU stack) ---------------


def _real_dataset(root, episodes: int = 2, frames: int = 3) -> pathlib.Path:
    """The node dataset layout end-to-end: episodes/ep_NNNN/{episode.json,
    actions.json, frame_*.jpg} + norm_stats.json (what _episodes_raw /
    _samples read; Phase 1 host65: the engine imported load_episode from the
    wrong module — zero coverage kept it alive)."""
    from PIL import Image

    ds = root / "datasets" / "s01__stop-at-target__v1"
    for ep in range(1, episodes + 1):
        d = ds / "episodes" / f"ep_{ep:04d}"
        d.mkdir(parents=True)
        (d / "episode.json").write_text(json.dumps(
            {"id": f"ep_{ep:04d}", "task": "stop at the target"}))
        (d / "actions.json").write_text(json.dumps([
            {"t": i * 0.1, "vx": 0.1 * ep + 0.01 * i, "vy": 0.2 * ep,
             "wz": 0.3 * ep} for i in range(frames)]))
        for i in range(frames):
            Image.new("RGB", (224, 224), (ep * 30, 100, i * 40)).save(
                d / f"frame_{i + 1:04d}.jpg", "JPEG")
    (ds / "norm_stats.json").write_text(json.dumps(
        {"min": [-1.0] * 7, "max": [1.0] * 7}))
    return ds


def test_samples_loads_dataset_layout(tmp_path):
    ds = _real_dataset(tmp_path, episodes=20)
    spec = {"student": "s00", "dataset": "s01/stop-at-target/v1",
            "prompt_template": "In: What action should the robot take to {task}?\nOut:"}
    train, held, stats = engine._samples(ds, spec)
    # 20 episodes x 3 frames = 60 samples; the seeded per-episode hash split
    # (split.HELD_OUT_FRACTION, (s00, stop-at-target)) keeps exactly 5 episodes
    # (ep_0003/0009/0012/0015/0017) held out — deterministic, assert exactly.
    assert len(train) == 45 and len(held) == 15
    frame, prompt, norm, target = train[0]
    assert frame.endswith(".jpg") and pathlib.Path(frame).is_file()
    assert prompt == ("In: What action should the robot take to "
                      "stop at the target?\nOut:")
    assert len(norm) == 7 and all(0.0 <= v <= 1.0 for v in norm)
    assert norm == target  # the sample carries the same normalized 7d twice
    assert stats.min == (-1.0,) * 7 and stats.max == (1.0,) * 7


def test_samples_requires_norm_stats(tmp_path):
    ds = _real_dataset(tmp_path)
    (ds / "norm_stats.json").unlink()
    with pytest.raises(RobovlaError) as ei:
        engine._samples(ds, {"student": "s00",
                             "dataset": "s01/stop-at-target/v1",
                             "prompt_template": "x {task} y"})
    assert ei.value.code == "dataset_not_ready"
