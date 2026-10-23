"""core.scheduler — FIFO admission with run-now, no preemption, priority +
config-aware first-fit (design §3, §10, §11 test matrix).

Pure: job times and node states are injected. The scheduler only *decides*;
the service seam dispatches via the runner and applies the resulting
``start``/``fail`` transitions.
"""

import pytest

from robovla.core.budgets import estimate_job_gb
from robovla.core.config import (
    AgentConfig,
    GpuBudgets,
    GpuConfig,
    NodeConfig,
    Presets,
)
from robovla.core.errors import RobovlaError
from robovla.core.scheduler import (
    Job,
    NodeState,
    Scheduler,
    default_duration_s,
)

# --- fixtures (no node names from fleet.yaml — hardware is config) ------------


def _node(
    nid="nodeA",
    priority=0,
    train_slots=1,
    infer_slots=2,
    vram=32.0,
    train_gb=18.0,
    infer_gb=11.0,
    batch=(1, 4, 8),
    enabled=True,
):
    return NodeConfig(
        id=nid,
        label=nid,
        image="robovla-runner-cuda:test",
        workdir="/w",
        agent=AgentConfig(lan_ip="10.0.0.99", port=8601, token_ref=f"{nid}-agent"),
        gpu=GpuConfig(
            vram_gb=vram,
            quant="bnb8",
            train_slots=train_slots,
            infer_slots=infer_slots,
            budgets=GpuBudgets(train_gb=train_gb, infer_gb=infer_gb),
        ),
        presets=Presets(batch=batch),
        priority=priority,
        student_selectable=False,
        enabled=enabled,
    )


def _states(**kw):
    out = {}
    for nid, (online, free) in kw.items():
        out[nid] = NodeState(online=online, free_vram_gb=free)
    return out


def _job(
    jid="j_20261021_0001",
    student="s07",
    job_type="train",
    target=None,
    run_now=False,
    at=100.0,
    budget=11.0,
    dur=3600.0,
):
    return Job(
        job_id=jid,
        student=student,
        job_type=job_type,
        target_node=target,
        run_now=run_now,
        submitted_at=at,
        budget_gb=budget,
        est_duration_s=dur,
    )


B1 = estimate_job_gb(18.0, 11.0, "bnb8", "train", 1, 8)  # 11 GB batch-1 job
B8 = estimate_job_gb(18.0, 11.0, "bnb8", "train", 8, 8)  # 18 GB full-batch job


class TestFifoAndRunNow:
    def test_fifo_order(self):
        s = Scheduler([_node()])
        s.add(_job(jid="j_20261021_0001", at=100.0))
        s.add(_job(jid="j_20261021_0002", at=200.0))
        admitted = s.tick(_states(nodeA=(True, 32.0)))  # 1 slot -> only the first
        assert [j.job_id for j in admitted] == ["j_20261021_0001"]

    def test_run_now_jumps_line_after_current_completion(self):
        s = Scheduler([_node()])
        s.add(_job(jid="j_20261021_0000", at=10.0, budget=B1))  # currently running
        s.transition("j_20261021_0000", "start", node_id="nodeA")
        s.add(_job(jid="j_20261021_0001", at=100.0))  # old FIFO job
        s.add(_job(jid="j_20261021_0002", at=200.0, run_now=True))
        s.add(_job(jid="j_20261021_0003", at=300.0))  # new FIFO job
        states = _states(nodeA=(True, 32.0))
        # slot full: run-now waits for the earliest completion (§3, no preemption)
        assert s.tick(states) == []
        # the current run finishes -> run-now is admitted to the front of the
        # line, ahead of both FIFO jobs regardless of submission order.
        s.transition("j_20261021_0000", "finish")
        s.transition("j_20261021_0000", "done")
        admitted = s.tick(states)
        assert [j.job_id for j in admitted] == ["j_20261021_0002"]

    def test_run_now_start_immediately_if_slot_free(self):
        s = Scheduler([_node()])
        s.add(_job(jid="j_20261021_0001", at=100.0))
        s.add(_job(jid="j_20261021_0002", at=200.0, run_now=True))
        admitted = s.tick(_states(nodeA=(True, 32.0)))
        assert [j.job_id for j in admitted] == ["j_20261021_0002"]


class TestNoPreemption:
    def test_running_job_holds_its_slot(self):
        s = Scheduler([_node()])
        running = _job(jid="j_20261021_0001", at=50.0)
        s.add(running)
        s.add(_job(jid="j_20261021_0002", at=100.0, run_now=True, budget=B1))
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        # slot full: run-now must wait for the current completion (§3)
        assert s.tick(_states(nodeA=(True, 32.0))) == []
        # the running job is untouched by the tick (no preemption)
        assert s.job("j_20261021_0001").state == "running"
        # after it finishes, run-now is admitted first
        s.transition("j_20261021_0001", "finish")
        s.transition("j_20261021_0001", "done")
        admitted = s.tick(_states(nodeA=(True, 32.0)))
        assert [j.job_id for j in admitted] == ["j_20261021_0002"]


class TestPriorityAndFit:
    def test_first_fit_is_priority_order(self):
        s = Scheduler([_node("nodeA", priority=0), _node("nodeB", priority=5)])
        s.add(_job(jid="j_20261021_0001", budget=B1))
        admitted = s.tick(_states(nodeA=(True, 32.0), nodeB=(True, 32.0)))
        assert len(admitted) == 1
        assert s.job("j_20261021_0001").node_id is None  # assigned at start
        # the admission decision must have picked nodeB (priority 5)
        assert s.admission_node("j_20261021_0001", _states(nodeA=(True, 32.0), nodeB=(True, 32.0))) == "nodeB"

    def test_config_aware_fit_16gb_batch1_vs_loaded_32gb(self):
        # 16 GB node (node3 profile): batch-1 job costs its full 14 GB base.
        # 32 GB node with the 27B assistant loaded: only 2.5 GB free.
        n16 = _node("nodeA", vram=16.0, train_gb=14.0, batch=(1,))
        n32 = _node("nodeB", priority=5, vram=32.0, train_gb=18.0)
        s = Scheduler([n16, n32])
        job = _job(jid="j_20261021_0001", budget=estimate_job_gb(14.0, 11.0, "bnb8", "train", 1, 1))
        s.add(job)
        states = _states(nodeA=(True, 16.0), nodeB=(True, 2.5))
        assert s.admission_node("j_20261021_0001", states) == "nodeA"
        admitted = s.tick(states)
        assert [j.job_id for j in admitted] == ["j_20261021_0001"]

    def test_two_batch1_slots_fit_on_32gb_not_two_full_batch(self):
        s = Scheduler([_node(train_slots=2)])
        s.add(_job(jid="j_20261021_0001", at=100.0, budget=B1))
        s.add(_job(jid="j_20261021_0002", at=200.0, budget=B1))
        s.add(_job(jid="j_20261021_0003", at=300.0, budget=B1))
        admitted = s.tick(_states(nodeA=(True, 32.0)))
        assert [j.job_id for j in admitted] == ["j_20261021_0001", "j_20261021_0002"]
        # and a full-batch job never gets two concurrent slots on 32 GB
        s2 = Scheduler([_node(train_slots=2)])
        s2.add(_job(jid="j_20261021_0011", at=100.0, budget=B8))
        s2.add(_job(jid="j_20261021_0012", at=200.0, budget=B8))
        admitted2 = s2.tick(_states(nodeA=(True, 32.0)))
        assert [j.job_id for j in admitted2] == ["j_20261021_0011"]


class TestSlotClasses:
    def test_infer_jobs_use_infer_slots(self):
        s = Scheduler([_node(train_slots=1, infer_slots=1)])
        s.add(_job(jid="j_20261021_0001", at=50.0, job_type="infer", budget=11.0))
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        s.add(_job(jid="j_20261021_0002", at=100.0, job_type="train", budget=B1))
        s.add(_job(jid="j_20261021_0003", at=200.0, job_type="infer", budget=11.0))
        admitted = s.tick(_states(nodeA=(True, 32.0)))
        # train slot free -> train admitted; infer slot taken -> 2nd infer waits
        assert [j.job_id for j in admitted] == ["j_20261021_0002"]

    def test_train_jobs_do_not_consume_infer_slots(self):
        s = Scheduler([_node(train_slots=1, infer_slots=2)])
        s.add(_job(jid="j_20261021_0001", at=50.0, budget=B1))
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        s.add(_job(jid="j_20261021_0002", at=100.0, job_type="infer", budget=11.0))
        admitted = s.tick(_states(nodeA=(True, 32.0)))
        assert [j.job_id for j in admitted] == ["j_20261021_0002"]


class TestPinningAndNodeHealth:
    def test_pinned_job_waits_on_its_node_only(self):
        s = Scheduler([_node("nodeA"), _node("nodeB", priority=5)])
        s.add(_job(jid="j_20261021_0001", target="nodeA", budget=B1))
        # nodeB (higher priority) is free, but the job is pinned to nodeA
        states = _states(nodeA=(True, 10.0), nodeB=(True, 32.0))
        assert s.tick(states) == []  # 10 < 11: nodeA's VRAM is short
        states_ok = _states(nodeA=(True, 16.0), nodeB=(True, 32.0))
        admitted = s.tick(states_ok)
        assert [j.job_id for j in admitted] == ["j_20261021_0001"]

    def test_pinned_to_offline_node_blocks(self):
        s = Scheduler([_node("nodeA")])
        s.add(_job(jid="j_20261021_0001", target="nodeA", budget=B1))
        blockers = s.admission_blockers("j_20261021_0001", _states(nodeA=(False, 0.0)))
        assert blockers == [
            {"node": "nodeA", "reason": "offline"},
        ]
        assert s.tick(_states(nodeA=(False, 0.0))) == []

    def test_pinned_to_disabled_node_never_admitted(self):
        s = Scheduler([_node("nodeA", enabled=False)])
        s.add(_job(jid="j_20261021_0001", target="nodeA", budget=B1))
        blockers = s.admission_blockers("j_20261021_0001", _states(nodeA=(True, 32.0)))
        assert blockers[0]["reason"] == "disabled"
        assert s.tick(_states(nodeA=(True, 32.0))) == []

    def test_budget_blocker_reason(self):
        s = Scheduler([_node("nodeA")])
        s.add(_job(jid="j_20261021_0001", budget=20.0))
        blockers = s.admission_blockers("j_20261021_0001", _states(nodeA=(True, 15.0)))
        assert blockers == [{"node": "nodeA", "reason": "budget"}]

    def test_busy_blocker_reason(self):
        s = Scheduler([_node("nodeA", train_slots=1)])
        s.add(_job(jid="j_20261021_0001", at=50.0, budget=B1))
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        s.add(_job(jid="j_20261021_0002", at=100.0, budget=B1))
        blockers = s.admission_blockers("j_20261021_0002", _states(nodeA=(True, 32.0)))
        assert blockers == [{"node": "nodeA", "reason": "busy"}]


class TestLifecycle:
    def test_kill_queued_and_running(self):
        s = Scheduler([_node()])
        s.add(_job(jid="j_20261021_0001", at=50.0))
        s.add(_job(jid="j_20261021_0002", at=100.0))
        s.transition("j_20261021_0002", "start", node_id="nodeA")
        s.transition("j_20261021_0001", "kill")
        assert s.job("j_20261021_0001").state == "killed"
        s.transition("j_20261021_0002", "kill")
        assert s.job("j_20261021_0002").state == "killed"
        # killed jobs are never admitted again
        assert s.tick(_states(nodeA=(True, 32.0))) == []

    def test_reschedule_changes_target_while_queued(self):
        s = Scheduler([_node("nodeA"), _node("nodeB", priority=5)])
        s.add(_job(jid="j_20261021_0001", target="nodeA", budget=B1))
        updated = s.set_target("j_20261021_0001", "nodeB")
        assert updated.target_node == "nodeB"
        updated = s.set_target("j_20261021_0001", None)  # clear the pin
        assert updated.target_node is None

    def test_reschedule_rejected_when_running(self):
        s = Scheduler([_node()])
        s.add(_job(jid="j_20261021_0001", at=50.0))
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        with pytest.raises(RobovlaError) as e:
            s.set_target("j_20261021_0001", None)
        assert e.value.code == "conflict"

    def test_start_requires_known_node(self):
        s = Scheduler([_node()])
        s.add(_job(jid="j_20261021_0001", at=50.0))
        with pytest.raises(RobovlaError) as e:
            s.transition("j_20261021_0001", "start", node_id="ghost")
        assert e.value.code == "validation"

    def test_add_duplicate_id_conflicts(self):
        s = Scheduler([_node()])
        s.add(_job(jid="j_20261021_0001"))
        with pytest.raises(RobovlaError) as e:
            s.add(_job(jid="j_20261021_0001"))
        assert e.value.code == "conflict"

    def test_add_invalid_job(self):
        s = Scheduler([_node()])
        with pytest.raises(RobovlaError):
            s.add(_job(jid="j_20261021_0001", job_type="retrain"))
        with pytest.raises(RobovlaError):
            s.add(_job(jid="j_20261021_0001", target="ghost"))
        with pytest.raises(RobovlaError):
            s.add(_job(jid="j_20261021_0001", budget=-1.0))

    def test_counts_per_student(self):
        s = Scheduler([_node()])
        s.add(_job(jid="j_20261021_0001", student="s01", at=50.0))
        s.add(_job(jid="j_20261021_0002", student="s01", at=60.0))
        s.add(_job(jid="j_20261021_0003", student="s02", at=70.0))
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        assert s.counts("s01") == {"running": 1, "queued": 1}
        assert s.counts("s02") == {"running": 0, "queued": 1}
        assert s.counts("s12") == {"running": 0, "queued": 0}


class TestPositionAndEta:
    def test_positions_on_single_node(self):
        s = Scheduler([_node(train_slots=1)])
        s.add(_job(jid="j_20261021_0001", at=50.0, dur=100.0))
        s.add(_job(jid="j_20261021_0002", at=60.0, dur=100.0))
        s.add(_job(jid="j_20261021_0003", at=70.0, dur=100.0, run_now=True))
        states = _states(nodeA=(True, 32.0))
        # run-now holds the head of the line
        assert s.position("j_20261021_0003", states) == 1
        assert s.position("j_20261021_0001", states) == 2
        assert s.position("j_20261021_0002", states) == 3

    def test_eta_sums_jobs_ahead_plus_own(self):
        s = Scheduler([_node(train_slots=1)])
        s.add(_job(jid="j_20261021_0001", at=50.0, dur=120.0))
        s.add(_job(jid="j_20261021_0002", at=60.0, dur=300.0))
        states = _states(nodeA=(True, 32.0))
        assert s.eta_s("j_20261021_0002", states) == pytest.approx(420.0)
        assert s.eta_s("j_20261021_0001", states) == pytest.approx(120.0)
        # running job: eta 0 (already started; progress is the stream's job)
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        assert s.eta_s("j_20261021_0001", states) == 0.0

    def test_eta_takes_best_eligible_node(self):
        s = Scheduler([_node("nodeA", priority=0), _node("nodeB", priority=5)])
        s.add(_job(jid="j_20261021_0002", at=40.0, dur=9999.0))  # running on nodeB
        s.transition("j_20261021_0002", "start", node_id="nodeB")
        s.add(_job(jid="j_20261021_0003", at=60.0, dur=60.0))
        states = _states(nodeA=(True, 32.0), nodeB=(True, 32.0))
        # nodeB: 9999 s of running work ahead -> nodeA (free, nothing ahead)
        # is the best eligible node -> eta is just the job's own duration.
        assert s.eta_s("j_20261021_0003", states) == pytest.approx(60.0)

    def test_eta_infinite_when_no_node_fits(self):
        s = Scheduler([_node("nodeA")])
        s.add(_job(jid="j_20261021_0001", budget=20.0, dur=60.0))
        assert s.eta_s("j_20261021_0001", _states(nodeA=(True, 5.0))) == float("inf")


class TestDurationDefaults:
    def test_smoke_and_selftest(self):
        assert default_duration_s("smoke") == pytest.approx(300.0)
        assert default_duration_s("gpu_selftest") == pytest.approx(900.0)

    def test_train_scales_with_batch_and_epochs(self):
        # base 90 min at batch 4 / 3 epochs (design §6: ~45 min–2 h/run)
        assert default_duration_s("train", batch=4, epochs=3) == pytest.approx(5400.0)
        assert default_duration_s("train", batch=8, epochs=3) == pytest.approx(2700.0)
        assert default_duration_s("train", batch=1, epochs=3) == pytest.approx(21600.0)
        assert default_duration_s("train", batch=4, epochs=6) == pytest.approx(10800.0)

    def test_crosscheck_is_half_dataset(self):
        # 50-episode deterministic subset (design §6) -> half the train time
        assert default_duration_s("crosscheck", batch=4, epochs=3) == pytest.approx(2700.0)

    def test_infer_has_no_duration(self):
        assert default_duration_s("infer") == 0.0

    def test_unknown_type_rejected(self):
        with pytest.raises(RobovlaError):
            default_duration_s("retrain")


class TestRestore:
    """SPOF rehydration (design §10): the in-memory queue is rebuilt from
    the durable rows at console boot and at every config reload — restored
    jobs may be RUNNING or FINISHING (a crash or a mid-training node patch
    must not drop the active job). ``restore`` is the rehydration seam;
    ``add`` stays the new-job validator (queued only)."""

    def _job(self, **over):
        from robovla.core.scheduler import Job

        base = dict(job_id="j_20261021_0001", student="s07", job_type="train",
                    target_node=None, run_now=False, submitted_at=100.0,
                    budget_gb=11.0, est_duration_s=3600.0)
        base.update(over)
        return Job(**base)

    def test_add_rejects_non_queued(self):
        s = Scheduler([_node()])
        with pytest.raises(RobovlaError) as e:
            s.add(self._job(state="running", node_id="nodeA"))
        assert e.value.code == "conflict"

    @pytest.mark.parametrize("state", ["queued", "running", "finishing"])
    def test_restore_accepts_every_active_state(self, state):
        s = Scheduler([_node()])
        j = s.restore(self._job(state=state, node_id="nodeA"))
        assert j.state == state
        assert s.job(j.job_id).state == state

    def test_restore_keeps_the_other_validations(self):
        s = Scheduler([_node()])
        with pytest.raises(RobovlaError) as e:
            s.restore(self._job(budget_gb=0.0))
        assert e.value.code == "validation"
        s.restore(self._job())
        with pytest.raises(RobovlaError) as e:
            s.restore(self._job())  # duplicate id
        assert e.value.code == "conflict"
        with pytest.raises(RobovlaError) as e:
            s.restore(self._job(job_id="j_20261021_0002", job_type="warp"))
        assert e.value.code == "validation"
