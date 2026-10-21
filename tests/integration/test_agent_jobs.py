"""agent.jobs — the JobRunner supervisor, exercised with REAL subprocesses.

The job process is ``tests/fixtures/fake_job.py``: the same contract as the
image's robovla_job.py (only input: the validated spec.json; outputs:
metrics.jsonl, stdout->log, val.json, probe.json, artifacts.json). So this
suite tests the real supervision logic — spawn, incremental file polling,
reap, SIGTERM/SIGKILL kill, pull-then-start — on CPU with no GPU.

Contract oracle: ``FakeRunner`` (runner/fake.py) — same view keys, same
incremental semantics, same error codes (conflict duplicate, not_found
unknown job), terminal views repeat.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from agent.jobs import JobRunner
from robovla.core.errors import RobovlaError
from robovla.core.spec import build_job_spec

FAKE_JOB = Path(__file__).parents[1] / "fixtures" / "fake_job.py"


def _argv_builder():
    return lambda spec, jdir: [sys.executable, str(FAKE_JOB), "--spec",
                               str(Path(jdir) / "spec.json")]


def _spec(job_type: str = "train", seed: int = 42, epochs: int = 3,
          job_id: str = "j_20261021_0001") -> dict:
    return build_job_spec(
        job_id=job_id, student="s07", job_type=job_type,
        dataset="s07/task-stop-at-target/v1", node_id="pai1",
        node_agent="10.0.0.9:8601", budget_gb=18, batch=4,
        seed=seed, overrides={"epochs": epochs},
    ).to_dict()


def wait_for(cond, timeout: float = 10.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = cond()
        if v:
            return v
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")


@pytest.fixture
def runner(tmp_path):
    return JobRunner(tmp_path, argv_builder=_argv_builder(),
                     grace_s=0.3,
                     log=lambda line: None)


# -- the full done contract ---------------------------------------------------------


def test_full_flow_metrics_incremental_and_terminal_repeats(runner, tmp_path):
    runner.submit(_spec(epochs=3, job_id="j_20261021_0001"))

    seen_steps = []
    all_logs = []
    val_delivered = []
    v = None
    for _ in range(200):
        v = runner.poll("j_20261021_0001")
        seen_steps += [m["step"] for m in v["metrics"]]
        all_logs += v["log"]
        # incremental values only (the terminal view re-delivers the final
        # val/probe by design — checked separately below)
        if v["state"] != "done" and v["val_accuracy"] is not None:
            val_delivered.append(v["val_accuracy"])
        if v["state"] == "done":
            break
        time.sleep(0.02)

    assert v["state"] == "done"
    # incremental metrics: no duplicate steps across polls, 6 total (2*epochs)
    assert seen_steps == list(range(1, 7)), seen_steps
    assert any("loss=" in ln for ln in all_logs)
    # val delivered incrementally at checkpoints, strictly increasing...
    assert val_delivered and all(
        b > a for a, b in zip(val_delivered, val_delivered[1:]))
    # ...and the terminal view carries the final checkpoint value (SPOF
    # rebuilds converge from the terminal poll, §10 — it may repeat the last
    # incremental value by design).
    assert v["val_accuracy"] == 0.8
    # terminal view: full artifacts with real nbytes
    assert v["artifacts"][0]["kind"] == "adapter"
    assert v["artifacts"][0]["nbytes"] > 0
    assert v["error"] is None
    # terminal polls repeat (SPOF rebuild / console resync)
    again = runner.poll("j_20261021_0001")
    assert again["state"] == "done" and again["artifacts"] == v["artifacts"]
    assert again["metrics"] == [] and again["log"] == []
    # the job's files land in the standard layout
    assert (tmp_path / "runs" / "j_20261021_0001" / "spec.json").is_file()
    assert (tmp_path / "runs" / "j_20261021_0001" / "argv.json").is_file()
    assert (tmp_path / "runs" / "j_20261021_0001" / "log").is_file()


def test_failed_job_carries_error_from_log(runner):
    runner.submit(_spec(seed=999, job_id="j_20261021_0002"))
    wait_for(lambda: runner.poll("j_20261021_0002")["state"] == "failed")
    v = runner.poll("j_20261021_0002")
    assert v["state"] == "failed"
    assert "OOM" in v["error"] and "seed 999" in v["error"]
    assert runner.poll("j_20261021_0002")["error"] == v["error"]  # repeats


# -- kill (§8.3: SIGTERM -> grace -> SIGKILL) ----------------------------------------


def test_kill_term_ignored_job_goes_killed(runner):
    runner.submit(_spec(seed=888, job_id="j_20261021_0003"))
    wait_for(lambda: runner.poll("j_20261021_0003")["state"] == "running")
    t0 = time.monotonic()
    runner.kill("j_20261021_0003")  # blocking up to grace_s (0.3 in fixture)
    v = runner.poll("j_20261021_0003")
    assert v["state"] == "killed" and v["error"] is None
    assert time.monotonic() - t0 < 3.0
    assert not runner.spawned_pids()  # reaped: nothing left in the registry


def test_kill_unknown_job_not_found(runner):
    with pytest.raises(RobovlaError) as ei:
        runner.kill("j_99999999_0000")
    assert ei.value.code == "not_found"


def test_kill_is_idempotent_on_terminal_job(runner):
    runner.submit(_spec(epochs=1, job_id="j_20261021_0004"))
    wait_for(lambda: runner.poll("j_20261021_0004")["state"] == "done")
    runner.kill("j_20261021_0004")  # no-op, no raise
    assert runner.poll("j_20261021_0004")["state"] == "done"


def test_kill_while_queued_before_start(tmp_path):
    slow = {}

    def puller(spec, workdir):
        slow["called"] = True
        time.sleep(0.3)

    r = JobRunner(tmp_path / "w2", argv_builder=_argv_builder(), grace_s=0.2,
                  puller=puller)
    r.submit(_spec(job_id="j_20261021_0005"))
    time.sleep(0.05)  # worker is inside the (slow) pull
    r.kill("j_20261021_0005")
    wait_for(lambda: r.poll("j_20261021_0005")["state"] == "killed")
    assert r.poll("j_20261021_0005")["state"] == "killed"
    # the job process never started
    time.sleep(0.4)
    assert not (tmp_path / "w2" / "runs" / "j_20261021_0005" / "log").exists()
    assert not r.spawned_pids()


# -- bundle pull seam ------------------------------------------------------------------


def test_pull_failure_fails_the_job_fast(tmp_path):
    def bad_puller(spec, workdir):
        raise RobovlaError("dataset_not_ready",
                           "dataset 's07/x/v1' is not ready to pull")

    r = JobRunner(tmp_path, argv_builder=_argv_builder(), puller=bad_puller)
    r.submit(_spec(job_id="j_20261021_0006"))
    wait_for(lambda: r.poll("j_20261021_0006")["state"] == "failed")
    v = r.poll("j_20261021_0006")
    assert v["error"].startswith("bundle pull:")


def test_puller_sees_canonical_spec_and_workdir(tmp_path):
    seen = {}

    def puller(spec, workdir):
        seen["spec"] = spec
        seen["workdir"] = workdir

    r = JobRunner(tmp_path, argv_builder=_argv_builder(), puller=puller,
                  grace_s=0.2)
    r.submit(_spec(job_id="j_20261021_0007"))
    wait_for(lambda: r.poll("j_20261021_0007")["state"] == "done")
    assert seen["workdir"] == tmp_path
    assert seen["spec"]["job_id"] == "j_20261021_0007"
    assert seen["spec"]["node"]["agent"] == "10.0.0.9:8601"  # canonical re-serialization


def test_queued_view_while_pull_is_in_flight(tmp_path):
    def slow_puller(spec, workdir):
        time.sleep(0.25)

    r = JobRunner(tmp_path, argv_builder=_argv_builder(), puller=slow_puller)
    r.submit(_spec(job_id="j_20261021_0008"))
    v = r.poll("j_20261021_0008")  # immediately: still pulling
    assert v["state"] == "queued" and v["metrics"] == [] and v["error"] is None
    wait_for(lambda: r.poll("j_20261021_0008")["state"] == "done")


# -- smoke probe + per-type artifacts ----------------------------------------------------


def test_smoke_delivers_predict_probe(runner):
    runner.submit(_spec(job_type="smoke", job_id="j_20261021_0009"))
    wait_for(lambda: runner.poll("j_20261021_0009")["state"] == "done")
    v = runner.poll("j_20261021_0009")
    assert v["predict_probe"] == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]
    assert v["artifacts"] == []  # smoke writes no adapter


def test_merge_and_crosscheck_artifact_kinds(runner):
    runner.submit(_spec(job_type="merge", job_id="j_20261021_0010", epochs=1))
    wait_for(lambda: runner.poll("j_20261021_0010")["state"] == "done")
    assert runner.poll("j_20261021_0010")["artifacts"][0]["kind"] == "merged"

    runner.submit(_spec(job_type="crosscheck", job_id="j_20261021_0011", epochs=1))
    wait_for(lambda: runner.poll("j_20261021_0011")["state"] == "done")
    assert runner.poll("j_20261021_0011")["artifacts"][0]["kind"] == "adapter"


# -- submit validation & errors (the oracle's codes) -------------------------------------


def test_duplicate_submit_conflict(runner):
    runner.submit(_spec(job_id="j_20261021_0012"))
    with pytest.raises(RobovlaError) as ei:
        runner.submit(_spec(job_id="j_20261021_0012"))
    assert ei.value.code == "conflict"


def test_infer_spec_is_served_not_dispatched(runner):
    with pytest.raises(RobovlaError) as ei:
        runner.submit(_spec(job_type="infer", job_id="j_20261021_0013"))
    assert ei.value.code == "validation"


def test_invalid_spec_is_validation(runner):
    bad = _spec(job_id="j_20261021_0014")
    bad["job_id"] = "not-a-job-id"
    with pytest.raises(RobovlaError) as ei:
        runner.submit(bad)
    assert ei.value.code == "validation"


def test_spawn_failure_fails_the_job(tmp_path):
    def bad_argv(spec, jdir):
        return ["/nonexistent/robovla-job-xyz", "--spec", str(jdir / "spec.json")]

    r = JobRunner(tmp_path, argv_builder=bad_argv)
    r.submit(_spec(job_id="j_20261021_0015"))
    wait_for(lambda: r.poll("j_20261021_0015")["state"] == "failed")
    assert r.poll("j_20261021_0015")["error"].startswith("spawn:")


# -- tail / artifacts / registry ---------------------------------------------------------


def test_tail_line_offset_and_unknown_jobs(runner):
    runner.submit(_spec(job_id="j_20261021_0016"))
    wait_for(lambda: runner.poll("j_20261021_0016")["state"] == "done")
    full = runner.tail("j_20261021_0016")
    lines = full.splitlines()
    assert len(lines) >= 6
    assert runner.tail("j_20261021_0016", offset=3) == "\n".join(lines[3:]) + "\n"
    assert runner.fetch_artifacts("j_20261021_0016")[0]["kind"] == "adapter"

    for fn in (lambda: runner.poll("nope"), lambda: runner.tail("nope"),
               lambda: runner.fetch_artifacts("nope")):
        with pytest.raises(RobovlaError) as ei:
            fn()
        assert ei.value.code == "not_found"


def test_spawned_registry_tracks_live_pids(runner):
    runner.submit(_spec(seed=888, job_id="j_20261021_0017"))
    pids = wait_for(lambda: runner.spawned_pids())
    assert len(pids) == 1
    runner.kill("j_20261021_0017")
    assert runner.spawned_pids() == frozenset()
