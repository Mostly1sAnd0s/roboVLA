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


# -- crosscheck wrapper: spec -> official finetune argv (rework, 2026-10-22) -------
#
# The official vla-scripts/finetune.py (pinned openvla SHA) is DDP + draccus and
# takes RLDS/TFDS input only: translate() materializes the deterministic
# 50-episode core.crosscheck_subset as an official RLDS dataset (through the
# injected ``writer`` seam — the CPU suite never imports tensorflow) and builds
# the official fixed argv (torchrun, single GPU).


def _fake_rlds_writer():
    calls = []

    def writer(root, name, samples):
        calls.append({"root": str(root), "name": name, "samples": samples})

    return writer, calls


def _flags(argv) -> dict:
    """The draccus flag pairs after the torchrun launch prefix."""
    return dict(zip(argv[7::2], argv[8::2]))


def test_crosscheck_argv_and_materialized_subset(tmp_path):
    spec = _spec("crosscheck", seed=7)
    ds = _real_dataset(tmp_path, episodes=60)
    job_dir = tmp_path / "wd" / "runs" / spec["job_id"]
    writer, calls = _fake_rlds_writer()
    argv, dirs = xf.translate(spec, ds, job_dir, writer=writer)

    # the official script, pinned in the image, single-GPU torchrun launch
    assert argv[0].endswith("torchrun")
    assert argv[1:6] == ["--standalone", "--nnodes", "1",
                         "--nproc-per-node", "1"]
    assert argv[6].endswith("vla-scripts/finetune.py")
    d = _flags(argv)
    assert d["--vla_path"] == spec["model"]
    assert d["--data_root_dir"].startswith(str(job_dir))
    assert d["--dataset_name"] == "toto"  # OXE config the RLDS subset borrows
    assert d["--run_root_dir"] == str(job_dir / "crosscheck_out")
    assert d["--adapter_tmp_dir"] == str(job_dir / "crosscheck_out" / "adapter-tmp")
    assert d["--batch_size"] == str(spec["optim"]["batch"])
    assert d["--grad_accumulation_steps"] == str(spec["optim"]["grad_accum"])
    assert d["--learning_rate"] == str(spec["optim"]["lr"])
    assert d["--image_aug"] == "true"
    assert d["--use_lora"] == "true"
    assert d["--lora_rank"] == "32"
    assert d["--use_quantization"] == "true"  # bnb8 -> official 4-bit (its only quant)
    assert d["--save_latest_checkpoint_only"] == "true"
    assert d["--run_id_note"] == spec["job_id"]
    # the official loop is step-based over an infinite (repeating) dataset:
    # 50 eps x 3 frames // (batch 4 x grad_accum 2) = 18 steps/epoch x 3 epochs
    assert d["--max_steps"] == "54"
    assert d["--save_steps"] == "54"  # one final checkpoint (+ LoRA merge)

    # the deterministic subset, materialized through the writer seam
    assert len(calls) == 1
    c = calls[0]
    assert c["name"] == "toto"
    assert c["root"] == str(job_dir / "crosscheck_rlds")
    got = {s["steps"][0]["episode_index"] for s in c["samples"]}
    expected = set(crosscheck_subset(
        [f"ep_{i:04d}" for i in range(1, 61)], spec["student"], spec["dataset"]))
    assert got == {int(e[3:]) for e in expected} and len(c["samples"]) == 50
    assert all(len(s["steps"]) == 3 for s in c["samples"])
    # the resolved artifact dirs (official exp_id subdirs) under the job dir
    assert str(dirs["run_dir"]).startswith(str(job_dir / "crosscheck_out"))
    assert str(dirs["adapter_dir"]).startswith(str(job_dir / "crosscheck_out"))


def test_crosscheck_bf16_runs_official_unquantized(tmp_path):
    spec = _spec("crosscheck", quant="bf16")
    ds = _real_dataset(tmp_path)
    job_dir = tmp_path / "wd" / "runs" / spec["job_id"]
    argv, _dirs = xf.translate(spec, ds, job_dir, writer=_fake_rlds_writer()[0])
    # bf16 has no bnb on the rocm leg: the official side runs unquantized
    assert _flags(argv)["--use_quantization"] == "false"


def test_crosscheck_smaller_dataset_uses_all(tmp_path):
    spec = _spec("crosscheck")
    ds = _real_dataset(tmp_path, episodes=30)
    job_dir = tmp_path / "wd" / "runs" / spec["job_id"]
    writer, calls = _fake_rlds_writer()
    argv, _dirs = xf.translate(spec, ds, job_dir, writer=writer)
    assert len(calls[0]["samples"]) == 30  # fewer than 50 -> the whole dataset
    assert _flags(argv)["--max_steps"] == "33"  # 90 frames // 8 = 11/epoch x 3


def test_crosscheck_main_cpu_contract(tmp_path, monkeypatch):
    """main() on CPU with both seams faked: the shared validator + fixed argv
    + the env merge (WANDB_MODE=disabled over the inherited env, not a
    replacement) + the artifacts contract on exit 0; a nonzero official exit
    skips the manifest and propagates the code."""
    from subprocess import CompletedProcess

    from images import rlds_writer as rw
    spec = _spec("crosscheck")
    p = tmp_path / "wd" / "runs" / spec["job_id"] / "spec.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(spec))
    _real_dataset(tmp_path / "wd")  # main derives ds_root from the spec path
    monkeypatch.setattr(rw, "write_rlds", _fake_rlds_writer()[0])
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["env"] = argv, kw.get("env")
        return CompletedProcess(argv, 0 if seen.get("rc", 0) == 0 else 1)

    monkeypatch.setattr(xf, "run_fixed_argv", fake_run)
    # the official script would create these (exp_id subdirs); main writes the
    # manifest from them on exit 0
    o = spec["optim"]
    exp = rw.exp_id(spec["model"], rw.RLDS_NAME, o["batch"] * o["grad_accum"],
                    o["lr"], spec["lora"]["rank"], 0.0, True, spec["job_id"],
                    spec["image_aug"])
    (p.parent / "crosscheck_out" / exp).mkdir(parents=True)
    (p.parent / "crosscheck_out" / "adapter-tmp" / exp).mkdir(parents=True)

    assert xf.main(["--spec", str(p)]) == 0
    assert seen["env"]["WANDB_MODE"] == "disabled"
    assert "PATH" in seen["env"]  # merged, not a bare replacement
    art = json.loads((p.parent / "artifacts.json").read_text())
    assert [a["kind"] for a in art] == ["adapter", "merged"]

    seen["rc"] = 1
    (p.parent / "artifacts.json").unlink()
    assert xf.main(["--spec", str(p)]) == 1
    assert not (p.parent / "artifacts.json").exists()


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
