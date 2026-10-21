"""JobSpec -> fixed argv (design §4, §9.8 "No shell, ever", Appendix B).

The agent derives a *fixed argv* from the validated JobSpec: one program per
job type, baked into the runner image, taking only ``--spec <path>`` — every
job-specific value rides in the spec file (validated by the same core schema
the console uses, §11 DRY), never on the command line as user text. The only
job-specific argv element is the spec path itself, under a directory named by
the pattern-validated job id (``j_\\d{8}_\\d{4}``); there is no shell to
escape for.

``crosscheck`` swaps the engine for the official ``finetune.py`` (a thin image
wrapper that translates the spec into the official script's flags — §6).
``infer`` is **served, not dispatched** (§9.5): the adapter is loaded into the
agent's serving registry, so an infer spec must never reach this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

from robovla.core.budgets import INFER_JOB_TYPES, TRAIN_JOB_TYPES
from robovla.core.errors import RobovlaError

#: Default image bin dir holding the job programs (§10: the stack lives in the
#: image; the path is env-overridable, not a machine name).
DEFAULT_JOB_BIN = "/opt/robovla/bin"

#: Every dispatchable job type (everything that consumes a train slot;
#: ``infer`` is excluded — it is served, §9.5).
DISPATCHABLE = frozenset(TRAIN_JOB_TYPES)

#: type -> program file name in the image bin dir. The in-house engine
#: (``robovla_job``) handles the console pipeline; ``crosscheck`` runs the
#: official openvla script on the deterministic 50-episode subset (§6) so
#: students compare engines on identical data.
PROGRAMS = {
    "smoke": "robovla_job",
    "train": "robovla_job",
    "validate": "robovla_job",
    "merge": "robovla_job",
    "gpu_selftest": "robovla_job",
    "home_selftest": "robovla_job",
    "crosscheck": "finetune",
}


def job_argv(
    spec: dict,
    job_dir: str | Path,
    *,
    python: str = sys.executable,
    bin_path: str = DEFAULT_JOB_BIN,
) -> list[str]:
    """The fixed argv for one validated spec (raises ``validation``)."""
    job_type = spec.get("type")
    if job_type in INFER_JOB_TYPES:
        raise RobovlaError("validation", "infer jobs are served, not dispatched (§9.5)")
    if job_type not in PROGRAMS:
        raise RobovlaError("validation",
                           f"job type {job_type!r} is not dispatchable")
    spec_path = Path(job_dir) / "spec.json"
    return [python, f"{bin_path}/{PROGRAMS[job_type]}", "--spec", str(spec_path)]
