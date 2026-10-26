"""crosscheck_finetune — the crosscheck job's program (design §6).

The agent's argv routes ``type: crosscheck`` to THIS wrapper (not
robovla_job): the crosscheck runs the OFFICIAL openvla ``finetune.py`` so
students compare their console engine against the official engine on
identical data — a deterministic 50-episode subset seeded per student +
dataset version (core.crosscheck_subset; the same episodes every re-run).

The wrapper: validates the spec with the SHARED validator, materializes the
subset as an official RLDS/TFDS dataset under the job dir
(``images/rlds_writer.py`` — the official script's only dataset input), and
execs the official script with a FIXED argv through the house proc seam
(no shell, ever). The official script's captured stdout/stderr are
forwarded to the wrapper's own streams (the proc seam captures the child's
output, host65 attempt 5), so the agent's job log carries them. The
official's wandb stream is captured by the house sink (below): the wrapper
shadows the real wandb package on the child's PYTHONPATH and the official
log payload lands in the job's standard metrics.jsonl. On a clean exit it
writes the file-contract artifact manifest (adapter + merged model) for
the console.

Official-vs-recipe notes (pinned openvla SHA c8f03f48, read 2026-10-22):

* quant: the official script is **4-bit only** (``use_quantization`` ->
  BitsAndBytesConfig nf4). A bnb8 spec therefore runs the official side at
  4-bit — the closest official mode (the console side stays 8-bit; the
  curves compare in shape, not bit-for-bit). bf16 runs unquantized.
* LoRA alpha: the official script hardcodes
  ``lora_alpha = min(lora_rank, 16)`` (alpha 16 at rank 32); there is no
  alpha flag.
* no seed flag in the official config: its TF shuffle is unseeded (the
  console engine trains its seeded split in order — compare the curves'
  shape/convergence, not point-for-point parity).
* no max_seq_length flag: the collator pads to the batch's longest sequence
  (~287 with the official 256 image tokens + prompt + 7 action tokens).
* build-time workaround (host65 attempt 6, 2026-10-22): the pinned script
  cannot complete ONE QLoRA step on this image's stack — its hardcoded
  DDP wrap (no static_graph) + the gradient checkpointing that
  prepare_model_for_kbit_training enables meet torch 2.9.0's implicit
  NON-reentrant checkpoint default and the first backward dies ("Expected
  to mark a variable ready only once"; openvla/openvla#90, a
  maintainer-acknowledged latent bug — no argv/env reaches the DDP args).
  The image therefore applies images/finetune_patch.diff at build time
  (--check against the SHA checkout, patched file sha256-pinned): the
  exact #90 fix (static_graph + explicitly reentrant checkpointing),
  mathematically neutral. The crosscheck results carry this note.
* wandb: the pinned script's only external sink is ``wandb.init`` /
  ``wandb.log`` (main process; the smoothed ``train_loss`` /
  ``action_accuracy`` / ``l1_loss`` every 10 gradient steps). LAN policy is
  credential-free wandb — the wrapper therefore shadows the real package
  with the house sink (``images/wandb_sink.py``, image path
  ``/opt/robovla/sink/wandb.py``, FIRST on the child's PYTHONPATH): the
  official log payload lands in the job's standard ``metrics.jsonl``, which
  the agent poll / SSE ``metrics`` event / the run-detail sparkline consume
  with no change (one metrics channel; the official's own process is
  otherwise untouched — disclosed here and in the results, like the
  build-time #90 patch). The sink's env contract (sink path, total steps,
  constant lr) is built by ``translate`` and merged over the inherited env
  in ``main``.
* steps, not epochs: the official loop runs ``max_steps`` optimizer steps
  over an infinite (repeating) dataset — the wrapper derives
  ``max_steps = epochs x (subset_frames // (batch x grad_accum))`` and sets
  ``save_steps = max_steps`` (one final checkpoint + LoRA merge).

Baked into the image at /opt/robovla/bin/finetune.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from robovla.core.errors import RobovlaError
from robovla.core.spec import validate_job_spec
from robovla.core.split import crosscheck_subset
from robovla.fs import sanitize_dataset_id
from robovla.proc import run_fixed_argv

from . import contract, engine, rlds_writer, wandb_sink

#: Image layout (the Dockerfiles own these paths).
TORCHRUN = "/opt/venv/bin/torchrun"
FINETUNE = "/opt/openvla/vla-scripts/finetune.py"

#: The wandb sink's image dir (the Dockerfile COPYs images/wandb_sink.py in
#: as <SINK_DIR>/wandb.py): first on the child's PYTHONPATH, so the
#: official's ``import wandb`` resolves to the house sink, not the
#: pip-installed package (the CPU test pins the wrapper constant and the
#: Dockerfile dest together — a drift is a silent loss of the curve).
SINK_DIR = "/opt/robovla/sink"

#: The official script's wandb flags (they feed the sink's init line — the
#: values exist so the official argv is complete and readable in logs).
WANDB_PROJECT = "robovla-crosscheck"
WANDB_ENTITY = "local"


def translate(spec: dict, ds_root: str | Path, job_dir: str | Path,
              *, writer=None) -> tuple[list[str], dict[str, Path]]:
    """Materialize the crosscheck subset (RLDS) and build the official argv.

    ``ds_root`` is the agent-pulled dataset root
    (``<workdir>/datasets/<owner>__<task>__vN>``); the RLDS subset lands at
    ``<job_dir>/crosscheck_rlds/`` (the official script sees only those
    episodes). Returns ``(argv, dirs, env)``: ``dirs`` are the official
    exp_id output dirs (``run_dir`` / ``adapter_dir``) — the artifact
    manifest's paths; ``env`` is the wandb-sink contract (sink target = the
    job's standard ``metrics.jsonl``, the derived max_steps, the constant
    lr, and the sink dir on PYTHONPATH) — ``main`` merges it over the
    inherited env. ``writer`` is the RLDS materialization seam (default
    :func:`images.rlds_writer.write_rlds`; CPU tests inject a recorder so
    the suite never imports tensorflow).
    """
    ds_root, job_dir = Path(ds_root), Path(job_dir)
    if writer is None:
        writer = rlds_writer.write_rlds  # resolved at call time (test seam)

    # The shared episode seam (episodes/ep_*/{episode.json, actions.json,
    # frame_*.jpg}) — the SAME source the console engine's _samples reads,
    # so both engines see identical frames + raw actions.
    eps = engine._episodes_raw(ds_root)
    ids = [e[0] for e in eps]
    subset = crosscheck_subset(ids, spec["student"], spec["dataset"])
    by_id = {e[0]: e for e in eps}

    samples = []
    frames_total = 0
    for eid in subset:
        _eid, task, frames, acts = by_id[eid]
        samples.append(rlds_writer.build_episode_sample(
            eid, task, [Path(f).read_bytes() for f in frames], acts))
        frames_total += len(acts)

    o = spec["optim"]
    eff_batch = o["batch"] * o["grad_accum"]
    max_steps = rlds_writer.max_steps_for(
        frames_total, o["batch"], o["grad_accum"], o["epochs"])
    rlds_root = job_dir / "crosscheck_rlds"
    writer(rlds_root, rlds_writer.RLDS_NAME, samples)

    exp = rlds_writer.exp_id(
        spec["model"], rlds_writer.RLDS_NAME, eff_batch, o["lr"],
        spec["lora"]["rank"], 0.0, spec["quant"] == "bnb8", spec["job_id"],
        spec["image_aug"])
    run_root = job_dir / "crosscheck_out"
    adapter_root = run_root / "adapter-tmp"

    argv = [
        TORCHRUN, "--standalone", "--nnodes", "1", "--nproc-per-node", "1",
        FINETUNE,
        "--vla_path", spec["model"],
        "--data_root_dir", str(rlds_root),
        "--dataset_name", rlds_writer.RLDS_NAME,
        "--run_root_dir", str(run_root),
        "--adapter_tmp_dir", str(adapter_root),
        "--batch_size", str(o["batch"]),
        "--max_steps", str(max_steps),
        "--save_steps", str(max_steps),  # one final checkpoint (+ merge)
        "--learning_rate", str(o["lr"]),
        "--grad_accumulation_steps", str(o["grad_accum"]),
        "--image_aug", "true" if spec["image_aug"] else "false",
        "--shuffle_buffer_size", str(min(frames_total, 100_000)),
        "--save_latest_checkpoint_only", "true",
        "--use_lora", "true",
        "--lora_rank", str(spec["lora"]["rank"]),
        # the official's only quant mode is 4-bit (module docstring, "quant")
        "--use_quantization", "true" if spec["quant"] == "bnb8" else "false",
        "--wandb_project", WANDB_PROJECT,
        "--wandb_entity", WANDB_ENTITY,
        "--run_id_note", spec["job_id"],
    ]
    env = {
        wandb_sink.ENV_SINK: str(job_dir / "metrics.jsonl"),
        wandb_sink.ENV_TOTAL: str(max_steps),
        wandb_sink.ENV_LR: str(o["lr"]),
        "PYTHONPATH": SINK_DIR,
    }
    return argv, {"run_dir": run_root / exp, "adapter_dir": adapter_root / exp}, env


def _write_artifacts(spec: dict, job_dir: Path, workdir: Path,
                     dirs: dict[str, Path]) -> None:
    """The file-contract manifest from the official output dirs (adapter +
    merged model). Called only on a clean official exit."""
    adapter, merged = dirs["adapter_dir"], dirs["run_dir"]
    if not adapter.is_dir() or not merged.is_dir():
        raise RobovlaError(
            "internal",
            "official run exited 0 without its checkpoint dirs "
            f"(adapter {adapter}, merged {merged})")
    manifest = contract.write_artifacts(
        job_dir, workdir, [("adapter", adapter), ("merged", merged)])
    engine._log(f"[{spec['job_id']}] crosscheck artifacts: "
                + ", ".join(a["path"] for a in manifest))


def main(argv: list[str] | None = None) -> int:
    """Entry: ``finetune --spec <spec.json>`` (the agent's fixed argv)."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "--spec":
        print("finetune: usage: finetune --spec <spec.json>", file=sys.stderr)
        return 2
    spec_path = argv[1]
    # The spec-read stage only: a missing (OSError) or malformed (json.
    # JSONDecodeError is a ValueError) spec file is 'unreadable spec'.
    # Nothing else may be swallowed by this handler (host65 attempt 2:
    # a writer-side ValueError leaked in here and was mislabeled).
    try:
        with open(spec_path) as f:
            raw = json.load(f)
        spec = validate_job_spec(raw).to_dict()
    except RobovlaError as e:
        print(f"finetune: {e.code}: {e.message}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"finetune: validation: unreadable spec {spec_path}: {e}",
              file=sys.stderr)
        return 1
    # The prepare stage (dispatch check + translate/RLDS materialization):
    # RobovlaErrors carry their own §9.6 code; anything else is internal.
    try:
        if spec["type"] != "crosscheck":
            raise RobovlaError(
                "validation", f"not a crosscheck job: {spec['type']!r}")
        job_dir, workdir = contract.job_dirs(spec_path)
        ds_root = workdir / "datasets" / sanitize_dataset_id(spec["dataset"])
        finetune_argv, dirs, sink_env = translate(spec, ds_root, job_dir)
    except RobovlaError as e:
        print(f"finetune: {e.code}: {e.message}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"finetune: internal: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    try:
        # proc.run_fixed_argv passes `env` straight to subprocess.run, which
        # REPLACES the environment when given — merge over the inherited env.
        # The sink dir goes FIRST on PYTHONPATH (shadowing the pip-installed
        # wandb package for the official's child), preserving any inherited
        # entries.
        env = {**os.environ, **sink_env}
        inherited = os.environ.get("PYTHONPATH")
        if inherited:
            env["PYTHONPATH"] = sink_env["PYTHONPATH"] + os.pathsep + inherited
        cp = run_fixed_argv(finetune_argv, env=env)
        # run_fixed_argv CAPTURES the official's streams (capture_output) —
        # forward both to our own stdout/stderr so the agent's job log
        # carries them (the module docstring's promise; host65 attempt 5,
        # 2026-10-22: the swallowed child output hid the real failure
        # behind a bare exit code).
        if cp.stdout:
            sys.stdout.write(cp.stdout)
            sys.stdout.flush()
        if cp.stderr:
            sys.stderr.write(cp.stderr)
            sys.stderr.flush()
        if cp.returncode == 0:
            _write_artifacts(spec, job_dir, workdir, dirs)
    except RobovlaError as e:
        print(f"finetune: {e.code}: {e.message}", file=sys.stderr)
        return 1
    print(f"[{spec['job_id']}] crosscheck finished "
          f"(official finetune.py, exit {cp.returncode})", flush=True)
    return cp.returncode


if __name__ == "__main__":
    sys.exit(main())
