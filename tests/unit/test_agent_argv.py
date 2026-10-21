"""agent.argv — JobSpec -> fixed argv (design §9.8 "No shell, ever")."""

from __future__ import annotations

import sys

import pytest

from agent import argv as argv_mod
from agent.argv import DISPATCHABLE, job_argv
from robovla.core.budgets import TRAIN_JOB_TYPES
from robovla.core.errors import RobovlaError
from robovla.core.spec import build_job_spec


def _spec(job_type: str) -> dict:
    return build_job_spec(
        job_id="j_20261021_0001", student="s07", job_type=job_type,
        dataset="s07/task-stop-at-target/v1", node_id="pai1",
        node_agent="10.0.0.9:8601", budget_gb=18,
    ).to_dict()


def test_dispatchable_is_every_train_class_type():
    assert DISPATCHABLE == TRAIN_JOB_TYPES  # one source; infer excluded (§9.5)


@pytest.mark.parametrize("job_type", sorted(DISPATCHABLE))
def test_fixed_argv_shape(job_type):
    a = job_argv(_spec(job_type), "/w/runs/j_20261021_0001")
    assert isinstance(a, list) and len(a) == 4
    assert a[0] == sys.executable  # the venv python the agent itself runs on
    assert a[2] == "--spec"
    assert a[3] == "/w/runs/j_20261021_0001/spec.json"
    # program names are image bin paths, never user text
    assert a[1].startswith(argv_mod.DEFAULT_JOB_BIN + "/")


def test_crosscheck_swaps_the_official_engine():
    a = job_argv(_spec("crosscheck"), "/w/runs/j_20261021_0001")
    assert a[1].endswith("/finetune")  # the official openvla script (§6)


def test_infer_is_served_not_dispatched():
    with pytest.raises(RobovlaError) as ei:
        job_argv(_spec("infer"), "/w/runs/j_20261021_0001")
    assert ei.value.code == "validation"


def test_unknown_type_rejected():
    s = _spec("train")
    s["type"] = "reboot"
    with pytest.raises(RobovlaError) as ei:
        job_argv(s, "/w/runs/j_20261021_0001")
    assert ei.value.code == "validation"


def test_bin_and_python_are_overridable():
    a = job_argv(_spec("train"), "/w/runs/j_20261021_0001",
                 python="/opt/venv/bin/python", bin_path="/opt/other/bin")
    assert a[0] == "/opt/venv/bin/python"
    assert a[1] == "/opt/other/bin/robovla_job"
