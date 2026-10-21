"""robovla_job — the in-image job program (design §6).

The only input is the validated ``--spec <spec.json>`` (the agent derives a
FIXED argv from the JobSpec — no shell, ever). One program per job type
(smoke/train/validate/merge/gpu_selftest/home_selftest); ``crosscheck`` is a
separate program (the official ``finetune.py`` wrapper) and ``infer`` is
served by the agent, not dispatched.

House rules: the spec is validated by the SHARED core validator (one schema,
DRY — the agent validates the same way); the observability files follow the
contract in contract.py; the GPU stack (torch/openvla/peft/bnb) is imported
LAZILY inside engine.py — a broken image fails with a clean §9.6 ``internal``
error ("runner stack unavailable"), never a bare traceback.

Baked into the image at /opt/robovla/bin/robovla_job.
"""

from __future__ import annotations

import json
import sys

from robovla.core.errors import RobovlaError
from robovla.core.spec import INFER_JOB_TYPES, validate_job_spec

#: Job types this program runs (everything the agent dispatches except the
#: crosscheck official-script job).
HANDLED = frozenset(
    {"smoke", "train", "validate", "merge", "gpu_selftest", "home_selftest"})


def _err(message: str) -> int:
    print(f"robovla_job: {message}", file=sys.stderr)
    return 1


def _load_engine():
    """The engine (lazy by design: its stack imports happen at first use)."""
    from images import engine  # noqa: PLC0415  (the lazy stack boundary)
    return engine


def dispatch(spec: dict, spec_path: str | None = None) -> int:
    """Route a validated spec to its engine entrypoint (lazy stack import).

    Raises RobovlaError for types this program does not handle (infer is
    served, crosscheck runs the official script). ``spec_path`` is needed for
    the job-dir/workdir derivation (the engine writes under it).
    """
    t = spec["type"]
    if t in INFER_JOB_TYPES:
        raise RobovlaError(
            "validation", "infer jobs are served, not dispatched (§9.5)")
    if t == "crosscheck":
        raise RobovlaError(
            "validation", "crosscheck runs the official finetune.py "
                          "(separate program; the agent's argv routes it)")
    if t not in HANDLED:
        raise RobovlaError("validation", f"unhandled job type: {t!r}")
    if spec_path is None:
        raise RobovlaError("internal", "dispatch requires the spec path")
    return _load_engine().run_job(spec, spec_path)  # lazy boundary


def main(argv: list[str] | None = None) -> int:
    """Entry: ``robovla_job --spec <spec.json>``. Exit 0/1 (2 = usage)."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "--spec":
        print("robovla_job: usage: robovla_job --spec <spec.json>", file=sys.stderr)
        return 2
    spec_path = argv[1]
    try:
        with open(spec_path) as f:
            raw = json.load(f)
        spec = validate_job_spec(raw).to_dict()  # the SHARED validator (DRY)
    except RobovlaError as e:
        return _err(f"validation: {e.message}")
    except (OSError, ValueError) as e:
        return _err(f"validation: unreadable spec {spec_path}: {e}")
    try:
        return dispatch(spec, spec_path)
    except RobovlaError as e:
        return _err(f"{e.code}: {e.message}")
    except Exception as e:  # the agent records failed + the log carries this
        return _err(f"internal: job crashed: {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
