"""ConsoleService SPOF rebuild (design §10): the in-memory queue is rebuilt
from the durable rows — at boot AND at every config reload. A console crash
(or a teacher node patch) mid-training must come back with the active job
still tracked and the queued jobs' pins intact.

Caught by the e2e class scenario (2026-10-23): ``Scheduler.add`` only
accepts QUEUED jobs ("new jobs must start queued"), but
``_build_scheduler`` re-adds the durable rows including running/finishing
ones — boot with a running job crashed the console, and no node patch could
ever land while a job was running. The rehydration seam is
``Scheduler.restore`` (same validation, minus the new-job state rule).
"""

from __future__ import annotations

import pathlib

import pytest

from robovla.core.config import parse_fleet
from robovla.core.spec import build_job_spec
from robovla.runner import FakeRunner
from robovla.service import ConsoleService
from robovla.storage import Storage

FLEET = """
version: 1
console:
  bind: 127.0.0.1
  port: 8501
  tls: false
  data_dir: /tmp/x
  session_ttl_h: 12
  limits:
    student: {max_running: 1, max_queued: 2, max_dataset_bytes: 2147483648,
              max_run_min: 180, run_now: true}
    admin_rate: "60/min"
  retention:
    run_logs_per_student: 20
    merged_export_days: 14
    backup: {enabled: true, at: "02:30", dest: /tmp/bk}
nodes:
  - id: testnode
    label: "test node"
    image: robovla-runner-cuda:2026.10
    workdir: /tmp/w
    agent: {lan_ip: "10.0.0.9", port: 8601, token_ref: testnode-agent}
    gpu: {vram_gb: 32, quant: bnb8, train_slots: 1, infer_slots: 2,
          budgets: {train_gb: 18, infer_gb: 11}}
    presets: {batch: [1, 4, 8]}
    priority: 0
    student_selectable: true
    enabled: true
"""


def _seed(tmp: pathlib.Path, state: str, node_id: str | None = None,
          pinned: bool = False) -> str:
    """One durable job row (valid spec) in the given state."""
    storage = Storage(tmp / "state.db")
    jid = "j_20261021_0001"
    spec = build_job_spec(
        job_id=jid, student="s07", job_type="train", dataset="s07/task/v1",
        node_id="testnode", node_agent="10.0.0.9:8601", budget_gb=18.0,
        batch=4, quant="bnb8").to_dict()
    storage.insert_job(jid, spec, state, 100.0, created_at="2026-10-21T09:00:00Z")
    if state != "queued" or pinned:
        storage.set_job(jid, state=state, node_id=node_id)
    storage.close()
    return jid


def _svc(tmp: pathlib.Path, fleet_yaml: str = FLEET) -> ConsoleService:
    return ConsoleService(
        tmp, parse_fleet(fleet_yaml), Storage(tmp / "state.db"),
        {"testnode": FakeRunner()}, clock=lambda: "2026-10-21T09:00:00Z",
        monotonic=lambda: 5000.0, tick_s=0.01)


def test_boot_with_running_job_rebuilds_the_queue(tmp_path):
    """Boot (SPOF recovery) with a running job: the console comes up and
    the job is tracked on its node (the agent keeps running it)."""
    jid = _seed(tmp_path, "running", node_id="testnode")
    svc = _svc(tmp_path)
    assert svc.scheduler.job(jid).state == "running"
    assert svc.scheduler.job(jid).node_id == "testnode"


def test_boot_with_queued_pinned_job_keeps_the_pin(tmp_path):
    jid = _seed(tmp_path, "queued", node_id="testnode", pinned=True)
    svc = _svc(tmp_path)
    j = svc.scheduler.job(jid)
    assert j.state == "queued"
    # the durable pin survives the rebuild (queued jobs keep their pin, §10)
    assert svc.storage.get_job(jid)["node_id"] == "testnode"


def test_reload_config_with_active_jobs(tmp_path):
    """A teacher node patch mid-training: the reload succeeds and the
    active jobs are unaffected (the docstring's promise)."""
    jid = _seed(tmp_path, "running", node_id="testnode")
    svc = _svc(tmp_path)
    assert svc.scheduler.job(jid).state == "running"

    fleet2 = parse_fleet(FLEET)  # an equivalent re-parse = a config reload
    svc.reload_config(fleet2, "teacher")
    assert svc.scheduler.job(jid).state == "running"
    assert svc.scheduler.job(jid).node_id == "testnode"


def test_add_still_rejects_active_jobs_at_submit_path(tmp_path):
    """The new-job validator is untouched: submit-path adds stay queued-only
    (the SPOF path is the only place active states are rehydrated)."""
    from robovla.core.errors import RobovlaError
    from robovla.core.scheduler import Job, Scheduler

    svc = _svc(tmp_path)
    with pytest.raises(RobovlaError) as e:
        svc.scheduler.add(Job(
            job_id="j_20261021_0099", student="s07", job_type="train",
            target_node=None, run_now=False, submitted_at=200.0,
            budget_gb=18.0, est_duration_s=3600.0, state="running"))
    assert e.value.code == "conflict"
