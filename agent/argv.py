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
The SmolVLA model family (``core.spec.model_family``) swaps it again:
smoke/train route to the lerobot wrapper ``smolvla_job``; types without a
SmolVLA equivalent are rejected, never silently downgraded.
``infer`` is **served, not dispatched** (§9.5): the adapter is loaded into the
agent's serving registry, so an infer spec must never reach this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

from robovla.core.budgets import INFER_JOB_TYPES, TRAIN_JOB_TYPES
from robovla.core.errors import RobovlaError
from robovla.core.spec import MODEL_FAMILY_SMOLVLA, model_family

#: Default image bin dir holding the job programs (§10: the stack lives in the
#: image; the path is env-overridable, not a machine name).
DEFAULT_JOB_BIN = "/opt/robovla/bin"

#: Every dispatchable job type (everything that consumes a train slot;
#: ``infer`` is excluded — it is served, §9.5).
DISPATCHABLE = frozenset(TRAIN_JOB_TYPES)

#: type -> program file name in the image bin dir (openvla family). The
#: in-house engine (``robovla_job``) handles the console pipeline;
#: ``crosscheck`` runs the official openvla script on the deterministic
#: 50-episode subset (§6) so students compare engines on identical data.
PROGRAMS = {
    "smoke": "robovla_job",
    "train": "robovla_job",
    "validate": "robovla_job",
    "merge": "robovla_job",
    "gpu_selftest": "robovla_job",
    "home_selftest": "robovla_job",
    "crosscheck": "finetune",
}

#: SmolVLA family (``model_family`` routing): smoke/train run the lerobot
#: wrapper (``smolvla_job`` — the crosscheck pattern for the SmolVLA
#: stack, images/smolvla_finetune.py). The openvla-specific types have no
#: SmolVLA equivalent in v1: crosscheck/gpu_selftest/home_selftest are
#: the openvla engine comparison and the openvla image's stack gates;
#: validate/merge are openvla-adapter operations (a SmolVLA checkpoint is
#: a full policy dir — merge is meaningless, held-out eval is a follow-up).
SMOLVLA_PROGRAMS = {
    "smoke": "smolvla_job",
    "train": "smolvla_job",
}


def program_for(job_type: str, model_id: object) -> str:
    """The image bin program name for (job type, model family).

    Raises ``validation`` for a dispatchable-but-unsupported combination
    (e.g. a crosscheck spec pointing at a SmolVLA model — the comparison
    target is the openvla engine; and validate/merge/selftest on the
    smolvla family).
    """
    if model_family(model_id) == MODEL_FAMILY_SMOLVLA:
        program = SMOLVLA_PROGRAMS.get(job_type)
        if program is None:
            raise RobovlaError(
                "validation",
                f"job type {job_type!r} is not supported for the smolvla "
                "model family (v1: smoke, train)")
        return program
    return PROGRAMS[job_type]


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
    program = program_for(job_type, spec.get("model"))
    spec_path = Path(job_dir) / "spec.json"
    return [python, f"{bin_path}/{program}", "--spec", str(spec_path)]
