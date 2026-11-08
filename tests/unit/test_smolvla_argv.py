"""agent.argv family routing — the SmolVLA model family swaps the engine
for the lerobot wrapper (the crosscheck precedent, applied by MODEL id).

Pinned here: smoke/train of a smolvla-named model route to the
``smolvla_job`` program; every other type on the family is REJECTED
(never silently downgraded to the openvla engine); the openvla family's
routing is untouched (the pre-family contract); infer stays served.
"""

from __future__ import annotations

import pytest

from agent.argv import PROGRAMS, SMOLVLA_PROGRAMS, job_argv, program_for
from robovla.core.errors import RobovlaError
from robovla.core.spec import build_job_spec

SMOLVLA_MODEL = "lerobot/smolvla_base"


def _spec(job_type: str, model: str = SMOLVLA_MODEL) -> dict:
    return build_job_spec(
        job_id="j_20261107_0001", student="s07", job_type=job_type,
        dataset="s07/park-at-target/v1", node_id="avc",
        node_agent="10.0.0.9:8601", budget_gb=18,
        model=model, quant="bf16",
    ).to_dict()


def test_program_for_family_routing():
    assert program_for("train", SMOLVLA_MODEL) == "smolvla_job"
    assert program_for("smoke", "me/SmolVLA-ft") == "smolvla_job"
    assert program_for("train", "openvla/openvla-7b") == "robovla_job"
    assert program_for("crosscheck", "openvla/openvla-7b") == "finetune"


@pytest.mark.parametrize("job_type",
                         ["validate", "merge", "crosscheck",
                          "gpu_selftest", "home_selftest"])
def test_smolvla_unsupported_types_rejected_loudly(job_type):
    # never silently route a smolvla spec to the openvla engine
    with pytest.raises(RobovlaError) as ei:
        program_for(job_type, SMOLVLA_MODEL)
    assert ei.value.code == "validation"
    assert "smolvla" in ei.value.message


@pytest.mark.parametrize("job_type", sorted(SMOLVLA_PROGRAMS))
def test_smolvla_argv_targets_the_wrapper(job_type):
    a = job_argv(_spec(job_type), "/w/runs/j_20261107_0001")
    assert a[1] == "/opt/robovla/bin/smolvla_job"
    assert a[2:] == ["--spec", "/w/runs/j_20261107_0001/spec.json"]


@pytest.mark.parametrize("job_type", sorted(PROGRAMS))
def test_openvla_routing_unchanged(job_type):
    a = job_argv(_spec(job_type, model="openvla/openvla-7b"),
                 "/w/runs/j_20261107_0001")
    assert a[1] == f"/opt/robovla/bin/{PROGRAMS[job_type]}"


def test_smolvla_crosscheck_never_reaches_the_official_openvla_script():
    with pytest.raises(RobovlaError):
        job_argv(_spec("crosscheck"), "/w/runs/j_20261107_0001")


def test_infer_stays_served_on_both_families():
    for model in (SMOLVLA_MODEL, "openvla/openvla-7b"):
        with pytest.raises(RobovlaError) as ei:
            job_argv(_spec("infer", model=model), "/w/runs/j_20261107_0001")
        assert ei.value.code == "validation"
