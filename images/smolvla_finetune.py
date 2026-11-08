"""smolvla_finetune — the SmolVLA family's job program (the crosscheck
pattern applied to the lerobot stack).

The agent's argv routes the SmolVLA model family's smoke/train specs to
THIS wrapper (agent/argv.py ``program_for``; the family is derived from
the spec's model id, ``core.spec.model_family``). Like the crosscheck
wrapper it runs the OFFICIAL trainer — lerobot v0.7.0 ``lerobot-train``
(``lerobot-train``, the pinned console script) — so students train the
SmolVLA base checkpoint with the same code the model card documents,
through the fixed-argv house proc seam (no shell, ever), inside the
SmolVLA runner image (``images/Dockerfile.cuda-smolvla``).

The wrapper: validates the spec with the SHARED validator, materializes
the seeded train split of the agent-pulled bundle as a local
LeRobotDataset (``images/lerobot_writer.py`` — the trainer's only data
input), execs the official trainer with a FIXED argv, and on a clean exit
(a) stamps ``serving.json`` into the trained policy dir — the family +
dataset-stats metadata the serving seam requires (``robovla.smolvla``),
and (b) writes the file-contract artifact manifest (kind ``adapter`` so
``service.adapter_path`` / slot claims / the UI work unchanged — the path
is a full policy dir, not a LoRA adapter; the family routing at serve
time decides how to load it).

Metrics: lerobot-train logs every ``--log_freq`` loop steps through
Trackio; the wrapper shadows that module with the house sink
(``images/trackio_sink.py``, image path ``/opt/robovla/sink/trackio.py``,
FIRST on the child's PYTHONPATH), so the curve lands in the job's
standard ``metrics.jsonl`` (one metrics channel — the agent poll / SSE /
sparkline consume it with no change; the wandb-sink precedent). The
child's captured stdout/stderr are forwarded to the wrapper's streams —
the job log carries the official's own output (the crosscheck's
host65-attempt-5 rule).

Recipe deltas vs the openvla engine (documented, deliberate):

  * quant: the spec's ``quant`` maps onto lerobot's ONE precision knob —
    ``--accelerator.mixed_precision`` (``MIXED_PRECISION`` below). bf16 is
    the campus default; fp32 (``no``) is the Pascal profile (sm_61 has no
    bf16 math). The wrapper never passes ``--policy.dtype`` — the policy
    weights stay float32 either way, and bnb never exists in this image
    (the spec validator rejects bnb8 for the family, one site).
  * LoRA: the DEFAULT fine-tune is SmolVLA's own preset — the VLM and
    vision encoder FROZEN, the ~100 M action expert trained (base-config
    ``train_expert_only``). The spec's ``lora`` fields are unused here
    (lerobot PEFT is a follow-up); ``max_seq_len`` is openvla context,
    ignored (catalog-fixed, §6).
  * prompt: the dataset's task string is the language input — the
    openvla ``In:/Out:`` template (spec ``prompt_template``) does NOT
    apply to SmolVLA (the VLM's chat processor owns the format).
  * split: the held-out episodes (``core.split.split``, seeded per
    student + dataset) are excluded from the written dataset; the
    official ``--dataset.eval_split`` stays 0. Held-out EVALUATION of a
    SmolVLA checkpoint (the openvla validate job's analog) is a
    documented follow-up — merging is meaningless for this family (the
    checkpoint is a full policy dir).
  * optimizer/scheduler: the house lr on SmolVLA's presets — adamw
    (weight_decay 1e-10, grad_clip 10, the base-config values) + the
    ``cosine_decay_with_warmup`` schedule with lerobot's auto-scaling
    (short runs scale the 1000/30000 warmup/decay into the run, the
    behavior the SmolVLA fine-tunes themselves use).
  * steps: lerobot counts MICRO-batch loop steps; ``--steps`` is
    ``epochs x (frames // batch)`` (lerobot_writer.lerobot_steps) and
    gradient accumulation rides on
    ``--accelerator.gradient_accumulation.steps`` (the effective batch
    matches the house recipe).
  * smoke: one loop step + one final checkpoint (the load/forward/
    backward/checkpoint path — the §6 smoke gate minus the predict
    probe; the probe needs a serving round-trip and lands with the
    validate follow-up).
  * the trainer's Trackio artifact uploads are disabled
    (``--trackio.disable_artifact=true``); the sink keeps artifact
    calls as loud-never-crash no-ops in case the flag drifts.

Offline model source: the base checkpoint + its SmolVLM2 backbone come
from the node's model volume (``HF_HUB_CACHE=/models``,
``HF_HUB_OFFLINE=1`` — the quadlet env, same as the openvla image).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from robovla.core.errors import RobovlaError
from robovla.core.norm_stats import NormStats
from robovla.core.spec import validate_job_spec
from robovla.core.split import split
from robovla.fs import sanitize_dataset_id
from robovla.proc import run_fixed_argv
from robovla.smolvla import serving_meta_dict

from . import contract, engine, lerobot_writer, trackio_sink

#: Image layout (the Dockerfile owns these paths).
TRAIN_BIN = "/opt/venv/bin/lerobot-train"

#: The sink dir (the Dockerfile COPYs images/trackio_sink.py in as
#: <SINK_DIR>/trackio.py — FIRST on the child's PYTHONPATH, shadowing the
#: pip-installed package; the CPU test pins the wrapper constant and the
#: Dockerfile dest together, the wandb-sink rule).
SINK_DIR = "/opt/robovla/sink"

#: Trackio project name (feeds the sink's init line only — the values
#: exist so the official argv is complete and readable in logs).
TRACKIO_PROJECT = "robovla-smolvla"

#: Spec ``quant`` -> ``--accelerator.mixed_precision`` (the ONLY precision
#: knob this wrapper passes). "no" keeps the run float32 end to end — the
#: Pascal/sm_61 profile (bf16 kernels do not exist there; fp16 math is
#: crippled on GeForce Pascal). An unlisted quant is a loud validation
#: error, never a silent bf16 default.
#:
#: THE fp32 VALUE CARRIES YAML QUOTES ON PURPOSE (lerobot 0.7.0 cuda run,
#: 2026-11-08): lerobot parses every CLI value through YAML, where a bare
#: ``no`` decodes to the boolean False and AcceleratorConfig's
#: ``mixed_precision: str`` field then dies with a draccus ParsingError.
#: The argv element ``"no"`` (literal quote characters, fixed argv — no
#: shell involved) decodes to the STRING "no". bf16 needs no quoting.
MIXED_PRECISION = {"bf16": "bf16", "fp32": '"no"'}

#: The scheduler shape around the house lr (SmolVLA base-config presets;
#: lerobot auto-scales warmup/decay into runs shorter than decay steps).
SCHEDULER_TYPE = "cosine_decay_with_warmup"
SCHEDULER_WARMUP = 1000
SCHEDULER_DECAY = 30000
SCHEDULER_DECAY_LR = 2.5e-6
OPTIM_WEIGHT_DECAY = 1e-10
OPTIM_GRAD_CLIP = 10.0


def translate(spec: dict, ds_root: str | Path, job_dir: str | Path,
              *, writer=None) -> tuple[list[str], dict[str, Path], dict[str, str]]:
    """Materialize the train split (LeRobot format) and build the official argv.

    ``ds_root`` is the agent-pulled dataset root
    (``<workdir>/datasets/<owner>__<task>__vN>``); the dataset lands at
    ``<job_dir>/lerobot_ds/`` (the official trainer sees only the seeded
    TRAIN episodes — the held-out split never trains, the same rule as
    the console engine and the crosscheck). Returns
    ``(argv, dirs, env)``: ``dirs`` locates the trained policy dir
    (resolved from the checkpoint layout AFTER a clean exit); ``env`` is
    the sink contract (target = the job's standard metrics.jsonl, the
    derived total, and the sink dir FIRST on PYTHONPATH) — ``main``
    merges it over the inherited env. ``writer`` is the materialization
    seam (default :func:`images.lerobot_writer.write_lerobot_dataset`;
    CPU tests inject a recorder so the suite never imports lerobot).
    """
    ds_root, job_dir = Path(ds_root), Path(job_dir)
    if writer is None:
        writer = lerobot_writer.write_lerobot_dataset  # call-time seam

    # The shared episode seam (same source the console engine's _samples
    # reads) and the SHARED seeded split — both families train identical
    # episodes. Actions arrive 7-dim (course convention); the writer
    # slices the (vx, vy, wz) head.
    eps = engine._episodes_raw(ds_root)
    ids = [e[0] for e in eps]
    parts = split(ids, spec["student"], spec["dataset"].split("/")[1])
    train_ids = set(parts.train)
    by_id = {e[0]: e for e in eps}

    samples = []
    frames_total = 0
    for eid in parts.train:
        _eid, task, frames, acts = by_id[eid]
        samples.append(lerobot_writer.build_episode_sample(
            eid, task, [Path(f).read_bytes() for f in frames], acts))
        frames_total += len(acts)
    if not train_ids or not samples:
        raise RobovlaError("dataset_not_ready",
                           f"train split is empty ({len(ids)} episodes)")

    mp = MIXED_PRECISION.get(spec["quant"])
    if mp is None:
        raise RobovlaError("validation",
                           f"quant {spec['quant']!r} has no smolvla "
                           f"mixed-precision mapping (known: "
                           f"{sorted(MIXED_PRECISION)})")

    o = spec["optim"]
    steps = lerobot_writer.lerobot_steps(frames_total, o["batch"], o["epochs"])
    if spec["type"] == "smoke":
        steps = 1  # the §6 smoke gate: one step, one checkpoint

    ds_dir = job_dir / "lerobot_ds"
    repo_id = f"robovla/{spec['job_id']}"
    writer(ds_dir, repo_id, samples)

    out_dir = job_dir / "lerobot_out"
    lr = str(o["lr"])
    argv = [
        TRAIN_BIN,
        "--dataset.repo_id", repo_id,
        "--dataset.root", str(ds_dir),
        "--dataset.eval_split", "0",  # the house split already applied
        "--dataset.image_transforms.enable",
        "true" if spec["image_aug"] else "false",
        "--policy.path", spec["model"],
        # infer the I/O features from the dataset (1 camera, 3-dim
        # action) instead of the hub config's 3-camera/6-dim defaults —
        # the documented reset (configs/policies.py line 59 note).
        "--policy.input_features", "null",
        "--policy.device", "cuda",
        "--policy.push_to_hub", "false",
        "--output_dir", str(out_dir),
        "--job_name", spec["job_id"],
        "--seed", str(spec["seed"]),
        "--steps", str(steps),
        "--batch_size", str(o["batch"]),
        "--num_workers", "0",  # no DataLoader shm assumptions in the container
        "--log_freq", "1",     # one sink row per loop step (class curve)
        "--save_checkpoint", "true",
        "--save_freq", str(steps),  # one final checkpoint only
        # explicit optimizer/scheduler (the policy-preset shortcut would
        # IGNORE a house lr override — configs/train.py validate():349).
        "--use_policy_training_preset", "false",
        "--optimizer.type", "adamw",
        "--optimizer.lr", lr,
        "--optimizer.weight_decay", str(OPTIM_WEIGHT_DECAY),
        "--optimizer.grad_clip_norm", str(OPTIM_GRAD_CLIP),
        "--scheduler.type", SCHEDULER_TYPE,
        "--scheduler.num_warmup_steps", str(SCHEDULER_WARMUP),
        "--scheduler.num_decay_steps", str(SCHEDULER_DECAY),
        "--scheduler.peak_lr", lr,
        "--scheduler.decay_lr", str(SCHEDULER_DECAY_LR),
        "--accelerator.mixed_precision", mp,
        "--accelerator.gradient_accumulation.steps", str(o["grad_accum"]),
        "--trackio.enable", "true",        # -> the house sink (shadowed)
        "--trackio.project", TRACKIO_PROJECT,
        "--trackio.capture_console", "false",
        "--trackio.disable_artifact", "true",
    ]
    dirs = {
        "output_dir": out_dir,
        "checkpoints": out_dir / "checkpoints",
        "last": out_dir / "checkpoints" / "last",
    }
    env = {
        trackio_sink.ENV_SINK: str(job_dir / "metrics.jsonl"),
        trackio_sink.ENV_TOTAL: str(steps),
        "PYTHONPATH": SINK_DIR,
    }
    return argv, dirs, env


def final_model_dir(dirs: dict[str, Path]) -> Path:
    """The trained policy dir from the checkpoint layout (after exit 0).

    lerobot checkpoints live at ``<output>/checkpoints/<step>/`` with a
    ``last`` symlink updated per save (common/train_utils.py); the
    loadable policy dir is ``<step>/pretrained_model``. The wrapper saves
    once (``--save_freq=steps``); the numbered-dir fallback tolerates a
    layout drift — an absent policy dir is a loud ``internal``.
    """
    checkpoints = dirs["checkpoints"]
    cand = None
    last = dirs["last"]
    if last.is_dir():
        cand = last
    elif checkpoints.is_dir():
        steps = sorted(d for d in checkpoints.iterdir() if d.is_dir())
        cand = steps[-1] if steps else None
    if cand is None:
        raise RobovlaError("internal",
                           f"official run left no checkpoint under {checkpoints}")
    # resolve(): hand back the REAL step dir, not the `last` symlink path
    model = (cand / "pretrained_model").resolve()
    if not model.is_dir():
        raise RobovlaError("internal",
                           f"checkpoint {cand} has no pretrained_model dir")
    return model


def _write_serving_meta(spec: dict, model_dir: Path, ds_root: Path) -> None:
    """Stamp the serving metadata INTO the artifact (robovla.smolvla).

    The dataset's min/max norm stats are the §9.5 denormalization pair;
    the serving seam reads them from here, so a served run can never be
    un-normalized with stats that drifted from the training data.
    """
    p = ds_root / "norm_stats.json"
    if not p.is_file():
        raise RobovlaError("dataset_not_ready", f"no norm stats in {ds_root}")
    d = json.loads(p.read_text())
    try:
        stats = NormStats(min=tuple(d["min"]), max=tuple(d["max"]))
    except (KeyError, TypeError, RobovlaError) as e:
        raise RobovlaError("dataset_not_ready",
                           f"dataset norm stats unusable: {e}") from e
    (model_dir / "serving.json").write_text(
        json.dumps(serving_meta_dict(stats), indent=2))


def _finish_ok(spec: dict, job_dir: Path, workdir: Path, ds_root: Path,
               dirs: dict[str, Path]) -> None:
    """Serving stamp + the artifact manifest (kind ``adapter`` — the
    contract's train kind; the path is a full SmolVLA policy dir)."""
    model_dir = final_model_dir(dirs)
    _write_serving_meta(spec, model_dir, ds_root)
    manifest = contract.write_artifacts(
        job_dir, workdir, [("adapter", model_dir)])
    engine._log(f"[{spec['job_id']}] smolvla artifacts: "
                + ", ".join(a["path"] for a in manifest))


def main(argv: list[str] | None = None) -> int:
    """Entry: ``smolvla_job --spec <spec.json>`` (the agent's fixed argv)."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "--spec":
        print("smolvla_job: usage: smolvla_job --spec <spec.json>",
              file=sys.stderr)
        return 2
    spec_path = argv[1]
    # The spec-read stage only (the crosscheck rule: nothing else may be
    # swallowed by this handler — a writer error must not mislabel).
    try:
        with open(spec_path) as f:
            raw = json.load(f)
        spec = validate_job_spec(raw).to_dict()
    except RobovlaError as e:
        print(f"smolvla_job: {e.code}: {e.message}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"smolvla_job: validation: unreadable spec {spec_path}: {e}",
              file=sys.stderr)
        return 1
    # The prepare stage (family + type check, split, dataset
    # materialization, argv): RobovlaErrors carry their own §9.6 code.
    try:
        if spec["type"] not in ("smoke", "train"):
            raise RobovlaError("validation",
                               f"not a smolvla job type: {spec['type']!r}")
        if "smolvla" not in spec["model"].lower():
            raise RobovlaError(
                "validation",
                f"smolvla_job is the smolvla family program, got model "
                f"{spec['model']!r} (the openvla engine owns that path)")
        job_dir, workdir = contract.job_dirs(spec_path)
        ds_root = workdir / "datasets" / sanitize_dataset_id(spec["dataset"])
        train_argv, dirs, sink_env = translate(spec, ds_root, job_dir)
    except RobovlaError as e:
        print(f"smolvla_job: {e.code}: {e.message}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"smolvla_job: internal: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    cp = None
    try:
        # run_fixed_argv passes `env` straight to subprocess (REPLACES it):
        # merge over the inherited env, the sink dir FIRST on PYTHONPATH
        # (shadowing the pip-installed trackio package for the child).
        env = {**os.environ, **sink_env}
        inherited = os.environ.get("PYTHONPATH")
        if inherited:
            env["PYTHONPATH"] = sink_env["PYTHONPATH"] + os.pathsep + inherited
        cp = run_fixed_argv(train_argv, env=env)
        # Forward the captured child streams into the job log (the
        # crosscheck host65-attempt-5 rule: a swallowed child hides its
        # real failure behind a bare exit code).
        if cp.stdout:
            sys.stdout.write(cp.stdout)
            sys.stdout.flush()
        if cp.stderr:
            sys.stderr.write(cp.stderr)
            sys.stderr.flush()
        if cp.returncode == 0:
            _finish_ok(spec, job_dir, workdir, ds_root, dirs)
    except RobovlaError as e:
        print(f"smolvla_job: {e.code}: {e.message}", file=sys.stderr)
        return 1
    rc = cp.returncode if cp is not None else 1
    print(f"[{spec['job_id']}] smolvla finished "
          f"(lerobot-train, exit {rc})", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
