"""crosscheck_finetune — the crosscheck job's program (design §6).

The agent's argv routes ``type: crosscheck`` to THIS wrapper (not
robovla_job): the crosscheck runs the OFFICIAL openvla ``finetune.py`` so
students compare their console engine against the official engine on
identical data — a deterministic 50-episode subset seeded per student +
dataset version (core.crosscheck_subset; the same episodes every re-run).

The wrapper: validates the spec with the SHARED validator, materializes the
subset under the job dir, and execs the official script with a FIXED argv
(the house exec seam — no shell, ever). The official script's stdout/stderr
are inherited, so the agent's job log carries them.

Baked into the image at /opt/robovla/bin/finetune.
NOTE(Phase 1): the official script's flag set follows the PINNED openvla
commit — verified at node join (the crosscheck is a Phase 1 gate, §12).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from robovla.core.errors import RobovlaError
from robovla.core.spec import validate_job_spec
from robovla.core.split import crosscheck_subset
from robovla.fs import sanitize_dataset_id
from robovla.proc import run_fixed_argv

from . import contract

#: Image layout (the Dockerfiles own these paths).
PYTHON = "/opt/venv/bin/python"
FINETUNE = "/opt/openvla/finetune.py"


def translate(spec: dict, ds_root: str | Path, job_dir: str | Path) -> list[str]:
    """Materialize the crosscheck subset and build the official argv.

    ``ds_root`` is the agent-pulled dataset root
    (``<workdir>/datasets/<owner>__<task>__vN>``); the subset is copied to
    ``<job_dir>/crosscheck_eps`` (the official script sees only those
    episodes). Returns the full argv (fixed; executed via the proc seam).
    """
    ds_root, job_dir = Path(ds_root), Path(job_dir)
    eps_root = ds_root / "episodes"
    if not eps_root.is_dir():
        raise RobovlaError("dataset_not_ready", f"no episodes in {ds_root}")
    ids = [d.name for d in sorted(eps_root.iterdir()) if d.is_dir()]
    if not ids:
        raise RobovlaError("dataset_not_ready", f"no episode folders in {ds_root}")
    subset = crosscheck_subset(ids, spec["student"], spec["dataset"])
    out_root = job_dir / "crosscheck_eps"
    out_eps = out_root / "episodes"
    out_eps.mkdir(parents=True, exist_ok=True)
    for eid in subset:
        shutil.copytree(eps_root / eid, out_eps / eid, dirs_exist_ok=True)
    stats = ds_root / "norm_stats.json"
    if stats.is_file():
        shutil.copy2(stats, out_root / "norm_stats.json")

    o = spec["optim"]
    argv = [
        PYTHON, FINETUNE,
        "--data_path", str(out_root),
        "--pretrained_model_name_or_path", spec["model"],
        "--output_dir", str(job_dir / "crosscheck_out"),
        "--max_seq_length", str(o["max_seq_len"]),  # fixed 287 (catalog)
        "--per_device_train_batch_size", str(o["batch"]),
        "--gradient_accumulation_steps", str(o["grad_accum"]),
        "--learning_rate", str(o["lr"]),
        "--lora_r", str(spec["lora"]["rank"]),
        "--lora_alpha", str(spec["lora"]["alpha"]),
        "--num_train_epochs", str(o["epochs"]),
    ]
    if spec["quant"] == "bnb8":
        argv += ["--load_in_8bit", "true"]
    return argv


def main(argv: list[str] | None = None) -> int:
    """Entry: ``finetune --spec <spec.json>`` (the agent's fixed argv)."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "--spec":
        print("finetune: usage: finetune --spec <spec.json>", file=sys.stderr)
        return 2
    spec_path = argv[1]
    try:
        with open(spec_path) as f:
            raw = json.load(f)
        spec = validate_job_spec(raw).to_dict()
        if spec["type"] != "crosscheck":
            raise RobovlaError(
                "validation", f"not a crosscheck job: {spec['type']!r}")
        job_dir, workdir = contract.job_dirs(spec_path)
        ds_root = workdir / "datasets" / sanitize_dataset_id(spec["dataset"])
        finetune_argv = translate(spec, ds_root, job_dir)
    except RobovlaError as e:
        print(f"finetune: {e.code}: {e.message}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"finetune: validation: unreadable spec {spec_path}: {e}",
              file=sys.stderr)
        return 1
    try:
        cp = run_fixed_argv(finetune_argv)
    except RobovlaError as e:
        print(f"finetune: {e.code}: {e.message}", file=sys.stderr)
        return 1
    print(f"[{spec['job_id']}] crosscheck finished "
          f"(official finetune.py, exit {cp.returncode})", flush=True)
    return cp.returncode


if __name__ == "__main__":
    sys.exit(main())
