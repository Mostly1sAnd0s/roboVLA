"""images.smolvla_finetune — the SmolVLA family's job program (the
crosscheck pattern for the lerobot stack).

The suite drives the REAL file seams over a temp workdir (episodes,
norm stats, spec.json under runs/<job>/): the seeded split is applied
with the SHARED core splitter (the writer must receive exactly the train
episodes — never the held-out ones), the fixed argv is pinned flag by
flag (the lerobot v0.7.0 surface), the sink env contract is the crosscheck
vocabulary, and the success path stamps serving.json INTO the trained
policy dir + writes the kind-"adapter" manifest (slot claims and
service.adapter_path work unchanged). The official trainer itself is the
proc seam (monkeypatched here; real at the hardware row).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from PIL import Image

from images import lerobot_writer, smolvla_finetune
from images.smolvla_finetune import final_model_dir, main, translate
from robovla.core.errors import RobovlaError
from robovla.core.spec import build_job_spec
from robovla.core.split import split

STUDENT = "s01"
DATASET = "s01/park-at-target/v1"
TASK = "park at target"  # the episode task string (language input)
TASK_ID = "park-at-target"  # the dataset-id task — the SHARED split key
                            # (engine._samples uses spec.dataset parts[1])
JOB = "j_20261107_0001"
N_EP, N_FRAMES = 6, 4


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (7, 7, 7)).save(buf, format="JPEG")
    return buf.getvalue()


def _workdir(tmp_path) -> tuple[Path, Path]:
    """<workdir> with a pulled dataset (episodes + norm stats)."""
    workdir = tmp_path / "w"
    ds = workdir / "datasets" / "s01__park-at-target__v1" / "episodes"
    for i in range(N_EP):
        ep = ds / f"ep_{i:04d}"
        ep.mkdir(parents=True)
        (ep / "episode.json").write_text(json.dumps(
            {"id": f"ep_{i:04d}", "task": TASK, "success": True}))
        (ep / "actions.json").write_text(json.dumps([
            {"t": k * 0.1, "vx": 0.1 * k, "vy": 0.0, "wz": 0.05 * k}
            for k in range(N_FRAMES)]))
        for k in range(N_FRAMES):
            (ep / f"frame_{k:04d}.jpg").write_bytes(_jpeg())
    stats = {"min": [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
             "max": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]}
    (ds.parent / "norm_stats.json").write_text(json.dumps(stats))
    return workdir, ds.parent


def _spec(job_type: str = "train", model: str = "lerobot/smolvla_base",
          **kw) -> dict:
    return build_job_spec(
        job_id=JOB, student=STUDENT, job_type=job_type, dataset=DATASET,
        node_id="avc", node_agent="10.0.0.9:8601", budget_gb=12,
        model=model, quant="bf16", **kw,
    ).to_dict()


def _write_spec(workdir: Path, spec: dict) -> Path:
    job_dir = workdir / "runs" / spec["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    p = job_dir / "spec.json"
    p.write_text(json.dumps(spec))
    return p


@pytest.fixture
def recorder(monkeypatch):
    """The materialize seam: a recorder WRITER (translate takes the
    callable; .calls carries what it received)."""
    calls = []

    def fake_writer(root, repo_id, samples, **kw):
        Path(root).mkdir(parents=True, exist_ok=True)
        calls.append({"root": Path(root), "repo_id": repo_id,
                      "samples": samples})
    fake_writer.calls = calls
    monkeypatch.setattr(lerobot_writer, "write_lerobot_dataset", fake_writer)
    return fake_writer


def _train_ids() -> list[str]:
    ids = [f"ep_{i:04d}" for i in range(N_EP)]
    return list(split(ids, STUDENT, TASK_ID).train)


# -- translate -----------------------------------------------------------------------


def test_translate_applies_the_shared_seeded_split(tmp_path, recorder):
    workdir, ds_root = _workdir(tmp_path)
    argv, dirs, env = translate(_spec(), ds_root,
                                workdir / "runs" / JOB, writer=recorder)
    written = [s["id"] for s in recorder.calls[0]["samples"]]
    assert written == _train_ids()
    all_ids = {f"ep_{i:04d}" for i in range(N_EP)}
    held = all_ids - set(written)  # the seeded held-out share never trains
    assert not (held & set(written))
    assert recorder.calls[0]["repo_id"] == f"robovla/{JOB}"


def test_translate_argv_pins_the_lerobot_surface(tmp_path, recorder):
    workdir, ds_root = _workdir(tmp_path)
    spec = _spec()
    train = set(_train_ids())
    frames = len(train) * N_FRAMES
    steps = max(1, (frames // spec["optim"]["batch"]) * spec["optim"]["epochs"])
    argv, dirs, env = translate(spec, ds_root, workdir / "runs" / JOB,
                                writer=recorder)

    def flag(name):
        i = argv.index(name)
        return argv[i + 1]

    assert argv[0] == smolvla_finetune.TRAIN_BIN
    assert flag("--dataset.repo_id") == f"robovla/{JOB}"
    assert flag("--dataset.root").endswith("lerobot_ds")
    assert flag("--dataset.eval_split") == "0"  # house split already applied
    assert flag("--dataset.image_transforms.enable") == "true"
    assert flag("--policy.path") == "lerobot/smolvla_base"
    assert flag("--policy.input_features") == "null"  # derive from dataset
    assert flag("--policy.push_to_hub") == "false"
    assert flag("--seed") == str(spec["seed"])
    assert flag("--steps") == str(steps)
    assert flag("--batch_size") == str(spec["optim"]["batch"])
    assert flag("--log_freq") == "1"
    assert flag("--save_freq") == str(steps)      # one final checkpoint
    assert flag("--num_workers") == "0"           # no shm assumptions
    # explicit optimizer/scheduler (the preset path ignores a house lr)
    assert flag("--use_policy_training_preset") == "false"
    assert flag("--optimizer.type") == "adamw"
    assert flag("--optimizer.lr") == str(spec["optim"]["lr"])
    assert flag("--scheduler.type") == smolvla_finetune.SCHEDULER_TYPE
    assert flag("--scheduler.peak_lr") == str(spec["optim"]["lr"])
    assert flag("--accelerator.mixed_precision") == "bf16"
    assert flag("--accelerator.gradient_accumulation.steps") == \
        str(spec["optim"]["grad_accum"])
    assert flag("--trackio.enable") == "true"
    assert flag("--trackio.disable_artifact") == "true"
    # the argv is FIXED: the only spec-derived path values are under the
    # job dir; no user free text ever rides the command line
    assert all(isinstance(x, str) for x in argv)
    assert TASK not in " ".join(argv)


def test_translate_smoke_is_one_step(tmp_path, recorder):
    workdir, ds_root = _workdir(tmp_path)
    argv, _dirs, env = translate(_spec("smoke"), ds_root,
                                 workdir / "runs" / JOB, writer=recorder)
    assert argv[argv.index("--steps") + 1] == "1"
    from images import trackio_sink
    assert env[trackio_sink.ENV_TOTAL] == "1"


def test_translate_sink_env_contract(tmp_path, recorder):
    workdir, ds_root = _workdir(tmp_path)
    train = set(_train_ids())
    spec = _spec()
    steps = max(1, (len(train) * N_FRAMES // spec["optim"]["batch"])
                * spec["optim"]["epochs"])
    _argv, dirs, env = translate(spec, ds_root, workdir / "runs" / JOB,
                                 writer=recorder)
    from images import trackio_sink
    assert env[trackio_sink.ENV_SINK] == str(
        workdir / "runs" / JOB / "metrics.jsonl")
    assert env[trackio_sink.ENV_TOTAL] == str(steps)
    assert env["PYTHONPATH"] == smolvla_finetune.SINK_DIR


def test_translate_empty_train_split_is_dataset_not_ready(tmp_path, recorder,
                                                          monkeypatch):
    # every episode held out: the writer must never be called with []
    workdir, ds_root = _workdir(tmp_path)
    monkeypatch.setattr("images.smolvla_finetune.split",
                        lambda ids, s, t: type(
                            "S", (), {"train": (), "held_out_set": set(ids)})())
    with pytest.raises(RobovlaError) as ei:
        translate(_spec(), ds_root, workdir / "runs" / JOB, writer=recorder)
    assert ei.value.code == "dataset_not_ready"


# -- final_model_dir -------------------------------------------------------------------


def test_final_model_dir_prefers_last_then_max(tmp_path):
    ck = tmp_path / "checkpoints"
    (ck / "000004" / "pretrained_model").mkdir(parents=True)
    (ck / "000002" / "pretrained_model").mkdir(parents=True)
    dirs = {"checkpoints": ck, "last": ck / "last"}
    assert final_model_dir(dirs) == ck / "000004" / "pretrained_model"
    (ck / "last").symlink_to("000002")
    assert final_model_dir(dirs) == ck / "000002" / "pretrained_model"


def test_final_model_dir_missing_is_internal(tmp_path):
    dirs = {"checkpoints": tmp_path / "none", "last": tmp_path / "last"}
    with pytest.raises(RobovlaError) as ei:
        final_model_dir(dirs)
    assert ei.value.code == "internal"


# -- main (the entry the image bin execs) ------------------------------------------


class _FakeCP:
    def __init__(self, rc, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _fake_trainer(monkeypatch, rc=0, checkpoint=True):
    calls = []

    def fake_run(argv, env=None, **kw):
        calls.append({"argv": argv, "env": env})
        if checkpoint:
            out = Path(argv[argv.index("--output_dir") + 1])
            (out / "checkpoints" / "000004" / "pretrained_model"
             ).mkdir(parents=True, exist_ok=True)
        return _FakeCP(rc, "official stdout", "official stderr")
    monkeypatch.setattr(smolvla_finetune, "run_fixed_argv", fake_run)
    return calls


def test_main_success_stamps_serving_and_manifests(tmp_path, recorder,
                                                   monkeypatch, capsys):
    workdir, _ = _workdir(tmp_path)
    spec = _spec()
    p = _write_spec(workdir, spec)
    calls = _fake_trainer(monkeypatch)
    assert main(["--spec", str(p)]) == 0
    model_dir = (workdir / "runs" / JOB / "lerobot_out" / "checkpoints"
                 / "000004" / "pretrained_model")
    meta = json.loads((model_dir / "serving.json").read_text())
    assert meta["family"] == "smolvla" and meta["action_dim"] == 3
    assert meta["norm_stats"]["min"][0] == -1.0  # the dataset's own stats
    manifest = json.loads(
        (workdir / "runs" / JOB / "artifacts.json").read_text())
    assert manifest[0]["kind"] == "adapter"
    assert manifest[0]["path"].endswith("pretrained_model")
    assert not Path(manifest[0]["path"]).is_absolute()  # workdir-relative
    # sink env merged over the inherited env, sink dir FIRST on PYTHONPATH
    env = calls[0]["env"]
    assert env["PYTHONPATH"] == smolvla_finetune.SINK_DIR
    # the captured official streams reached the job log
    out = capsys.readouterr().out
    assert "official stdout" in out


def test_main_forwards_inherited_pythonpath(tmp_path, recorder, monkeypatch):
    workdir, _ = _workdir(tmp_path)
    p = _write_spec(workdir, _spec())
    monkeypatch.setenv("PYTHONPATH", "/keep/me")
    calls = _fake_trainer(monkeypatch)
    assert main(["--spec", str(p)]) == 0
    assert calls[0]["env"]["PYTHONPATH"] == (
        smolvla_finetune.SINK_DIR + __import__("os").pathsep + "/keep/me")


def test_main_nonzero_exit_passes_through_no_artifacts(tmp_path, recorder,
                                                       monkeypatch):
    workdir, _ = _workdir(tmp_path)
    p = _write_spec(workdir, _spec())
    _fake_trainer(monkeypatch, rc=3, checkpoint=False)
    assert main(["--spec", str(p)]) == 3
    assert not (workdir / "runs" / JOB / "artifacts.json").exists()


def test_main_zero_exit_without_checkpoint_is_internal(tmp_path, recorder,
                                                       monkeypatch, capsys):
    workdir, _ = _workdir(tmp_path)
    p = _write_spec(workdir, _spec())
    _fake_trainer(monkeypatch, rc=0, checkpoint=False)
    assert main(["--spec", str(p)]) == 1
    assert "internal" in capsys.readouterr().err


def test_main_rejects_openvla_model_and_wrong_type(tmp_path, recorder,
                                                   monkeypatch, capsys):
    workdir, _ = _workdir(tmp_path)
    p = _write_spec(workdir, _spec(model="openvla/openvla-7b"))
    assert main(["--spec", str(p)]) == 1
    assert "smolvla family program" in capsys.readouterr().err
    spec = _spec()
    spec["type"] = "merge"
    p2 = _write_spec(workdir, spec)
    assert main(["--spec", str(p2)]) == 1


def test_main_unreadable_spec_is_validation(tmp_path, capsys):
    assert main(["--spec", str(tmp_path / "nope.json")]) == 1
    assert "unreadable spec" in capsys.readouterr().err
    assert main([]) == 2  # usage
