"""The queue scheduler: FIFO with run-now, no preemption, priority +
config-aware first-fit (design §3, §10; pure transition table in §10).

Pure core: job submission times and per-node states are *injected*; the
scheduler only decides. The service seam dispatches admitted jobs through
the runner client and applies the resulting ``start``/``fail`` transitions.

Admission (``tick``):
  * nodes are walked in first-fit order — ``priority`` higher first, ties keep
    fleet.yaml order (one rule, §3);
  * a job is admitted where it targets that node (auto = any) and the node has
    a free slot of the job's class (train|infer, one partition — DRY from
    :mod:`robovla.core.budgets`) and measured free VRAM >= the job's budget;
  * within a node, queue order is (run-now first, then FIFO by submission);
    a non-fitting job does not block smaller jobs behind it;
  * **no preemption** — running jobs are never displaced; ``run_now`` means
    "front of the line after the current completion" (§3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from .budgets import INFER_JOB_TYPES
from .config import NodeConfig
from .errors import RobovlaError
from .spec import JOB_TYPES
from .states import apply_transition

#: Display heuristics for ``eta_s`` / the dashboard (§7): base durations.
#: train is 90 min at batch 4 / 3 epochs, scaling inversely with batch and
#: linearly with epochs (design §6: ~45 min–2 h/run at batch 4–8); the
#: crosscheck runs the same recipe on a 50-episode deterministic subset.
_BASE_DURATIONS_S = {
    "smoke": 300.0,
    "train": 5400.0,
    "validate": 600.0,
    "merge": 900.0,
    "infer": 0.0,
    "crosscheck": 2700.0,
    "gpu_selftest": 900.0,
    "home_selftest": 900.0,
}

_TERMINAL_OR_RUNNING = ("running", "finishing")


def default_duration_s(job_type: str, batch: int | None = None, epochs: int | None = None) -> float:
    """Estimated run duration (display heuristic, not a guarantee).

    * train/crosscheck scale with ``batch`` (vs the base batch 4) and
      ``epochs`` (vs the base 3);
    * everything else is a flat base (infer serves until released -> 0).
    """
    if job_type not in JOB_TYPES:
        raise RobovlaError("validation", f"unknown job type: {job_type!r}")
    base = _BASE_DURATIONS_S[job_type]
    if job_type in ("train", "crosscheck"):
        if batch is None or epochs is None:
            raise RobovlaError("validation", f"{job_type} duration needs batch and epochs")
        if batch < 1 or epochs < 1:
            raise RobovlaError("validation", "batch and epochs must be >= 1")
        return base * (4 / batch) * (epochs / 3)
    return base


def slot_class(job_type: str) -> str:
    """Which per-node slot pool a job draws from (design §3): one partition."""
    return "infer" if job_type in INFER_JOB_TYPES else "train"


# --- data model ----------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """A queue entry. ``target_node`` None = auto (first fit); set = that node
    only (teacher pin or student-selected node, §9.3). ``node_id`` is the
    node assigned at ``start``. Times are the seam's monotonic clock."""

    job_id: str
    student: str
    job_type: str
    target_node: str | None
    run_now: bool
    submitted_at: float
    budget_gb: float
    est_duration_s: float
    state: str = "queued"
    node_id: str | None = None


@dataclass(frozen=True)
class NodeState:
    """One node's measured view, polled by the seam from the agent."""

    online: bool
    free_vram_gb: float


# --- scheduler -----------------------------------------------------------------


class Scheduler:
    def __init__(self, nodes: Sequence[NodeConfig]) -> None:
        if not nodes:
            raise RobovlaError("validation", "scheduler needs at least one node")
        self._nodes: tuple[NodeConfig, ...] = tuple(nodes)
        self._node_ids = {n.id for n in self._nodes}
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []  # submission order (FIFO)

    # -- accessors --

    def job(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise RobovlaError("not_found", f"unknown job: {job_id}") from None

    def jobs(self) -> tuple[Job, ...]:
        return tuple(self._jobs[jid] for jid in self._order)

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self._nodes)

    def _by_priority(self) -> list[NodeConfig]:
        """First-fit order: priority higher first; ties keep file order."""
        return sorted(self._nodes, key=lambda n: -n.priority)

    # -- queue mutations --

    def add(self, job: Job) -> Job:
        if not isinstance(job, Job):
            raise RobovlaError("validation", "add expects a Job")
        if not job.job_id or not isinstance(job.job_id, str):
            raise RobovlaError("validation", "job_id must be a non-empty string")
        if job.job_id in self._jobs:
            raise RobovlaError("conflict", f"duplicate job id: {job.job_id}")
        if not job.student:
            raise RobovlaError("validation", "student must be non-empty")
        if job.job_type not in JOB_TYPES:
            raise RobovlaError("validation", f"unknown job type: {job.job_type!r}")
        if job.target_node is not None and job.target_node not in self._node_ids:
            raise RobovlaError("validation", f"unknown node: {job.target_node!r}")
        if job.budget_gb <= 0:
            raise RobovlaError("validation", "budget_gb must be > 0")
        if job.est_duration_s < 0:
            raise RobovlaError("validation", "est_duration_s must be >= 0")
        if job.submitted_at < 0:
            raise RobovlaError("validation", "submitted_at must be >= 0")
        if job.state != "queued":
            raise RobovlaError("conflict", "new jobs must start queued")
        self._jobs[job.job_id] = job
        self._order.append(job.job_id)
        return job

    def transition(self, job_id: str, event: str, node_id: str | None = None) -> Job:
        """Apply a state-machine event (one table — :mod:`robovla.core.states`).

        ``start`` requires the node that will run the job (assigned to the job);
        the seam calls this after the runner confirms dispatch, and ``fail`` on
        dispatch failure / node death (§10: fail fast, auto-reschedule = new
        job with ``resume_from``).
        """
        job = self.job(job_id)
        if event == "start":
            if node_id is None or node_id not in self._node_ids:
                raise RobovlaError("validation", f"start needs a known node (got {node_id!r})")
        new_state = apply_transition(job.state, event)
        updated = replace(job, state=new_state, node_id=node_id if event == "start" else job.node_id)
        self._jobs[job_id] = updated
        return updated

    def set_target(self, job_id: str, node_id: str | None) -> Job:
        """Re-pin (or clear the pin on) a queued job — teacher reschedule (§8.3)."""
        job = self.job(job_id)
        if job.state != "queued":
            raise RobovlaError("conflict", f"job {job_id} is {job.state}; reschedule is queued-only")
        if node_id is not None and node_id not in self._node_ids:
            raise RobovlaError("validation", f"unknown node: {node_id!r}")
        apply_transition(job.state, "reschedule")  # validates the event
        updated = replace(job, target_node=node_id)
        self._jobs[job_id] = updated
        return updated

    def counts(self, student: str) -> dict[str, int]:
        """Per-student running/queued counts (caps are enforced by the service,
        §8.1 — an active inference slot counts as running: state-based)."""
        running = queued = 0
        for j in self._jobs.values():
            if j.student != student:
                continue
            if j.state in _TERMINAL_OR_RUNNING:
                running += 1
            elif j.state == "queued":
                queued += 1
        return {"running": running, "queued": queued}

    # -- admission --

    def _queue_key(self, j: Job) -> tuple[bool, float, str]:
        return (not j.run_now, j.submitted_at, j.job_id)

    def _targets_node(self, j: Job, node_id: str) -> bool:
        return j.target_node is None or j.target_node == node_id

    def _running_jobs(self, node_id: str, cls: str) -> list[Job]:
        return [
            j
            for j in self._jobs.values()
            if j.state in _TERMINAL_OR_RUNNING and j.node_id == node_id and slot_class(j.job_type) == cls
        ]

    def _free_slots(self, node: NodeConfig, cls: str) -> int:
        capacity = node.gpu.train_slots if cls == "train" else node.gpu.infer_slots
        return capacity - len(self._running_jobs(node.id, cls))

    def tick(self, node_states: dict[str, NodeState]) -> list[Job]:
        """Decide which queued jobs may start now (admission, not dispatch).

        Returns jobs in admission order; the seam dispatches each and applies
        ``start`` (or ``fail``). Within one tick, admitted jobs consume slots
        and VRAM for subsequent admissions (two-slot 32 GB scenario, §3).
        """
        admitted: list[Job] = []
        admitted_ids: set[str] = set()
        for node in self._by_priority():
            if not node.enabled:
                continue
            view = node_states.get(node.id)
            if view is None or not view.online:
                continue
            free_vram = view.free_vram_gb
            free = {"train": self._free_slots(node, "train"), "infer": self._free_slots(node, "infer")}
            candidates = sorted(
                (j for jid in self._order if (j := self._jobs[jid]).state == "queued" and self._targets_node(j, node.id)),
                key=self._queue_key,
            )
            for j in candidates:
                if j.job_id in admitted_ids:
                    continue
                cls = slot_class(j.job_type)
                if free[cls] <= 0 or free_vram < j.budget_gb:
                    continue  # this job doesn't fit here; smaller jobs may
                admitted.append(j)
                admitted_ids.add(j.job_id)
                free[cls] -= 1
                free_vram -= j.budget_gb
        return admitted

    def admission_node(self, job_id: str, node_states: dict[str, NodeState]) -> str | None:
        """The node (first-fit order) a job would be admitted to now, if any.

        A display heuristic: ignores queue position (jobs ahead on that node),
        which the position/eta helpers account for.
        """
        job = self.job(job_id)
        if job.state != "queued":
            return None
        for node in self._by_priority():
            if not self._targets_node(job, node.id) or not node.enabled:
                continue
            view = node_states.get(node.id)
            if view is None or not view.online:
                continue
            cls = slot_class(job.job_type)
            if self._free_slots(node, cls) <= 0:
                continue
            if view.free_vram_gb < job.budget_gb:
                continue
            return node.id
        return None

    def admission_blockers(self, job_id: str, node_states: dict[str, NodeState]) -> list[dict[str, Any]]:
        """Why a queued job cannot start now (feeds the ``waiting`` response,
        §9.3: reason busy vs budget). Empty list = admissible now."""
        job = self.job(job_id)
        if job.state != "queued":
            return []
        eligible = (
            [n for n in self._by_priority() if n.id == job.target_node]
            if job.target_node is not None
            else [n for n in self._by_priority()]
        )
        blockers: list[dict[str, Any]] = []
        for node in eligible:
            if not node.enabled:
                blockers.append({"node": node.id, "reason": "disabled"})
                continue
            view = node_states.get(node.id)
            if view is None or not view.online:
                blockers.append({"node": node.id, "reason": "offline"})
                continue
            cls = slot_class(job.job_type)
            if self._free_slots(node, cls) <= 0:
                blockers.append({"node": node.id, "reason": "busy"})
                continue
            if view.free_vram_gb < job.budget_gb:
                blockers.append({"node": node.id, "reason": "budget"})
                continue
            return []  # this node admits now
        return blockers

    # -- display: position + ETA --

    def position(self, job_id: str, node_states: dict[str, NodeState]) -> int:
        """1-based queue position; 0 for running/terminal jobs.

        For auto jobs this is the best (lowest) position across eligible
        nodes — the queue order that node would see (run-now first, then FIFO).
        """
        job = self.job(job_id)
        if job.state != "queued":
            return 0
        best = None
        for node in self._by_priority():
            if job.target_node is not None and node.id != job.target_node:
                continue
            ahead = sum(
                1
                for j in self._jobs.values()
                if j.state == "queued"
                and j.job_id != job_id
                and self._targets_node(j, node.id)
                and self._queue_key(j) < self._queue_key(job)
            )
            pos = ahead + 1
            best = pos if best is None else min(best, pos)
        return best if best is not None else 0

    def eta_s(self, job_id: str, node_states: dict[str, NodeState]) -> float:
        """Estimated seconds until the job finishes: queue wait + own duration.

        Heuristic (display only): running jobs on a node count their full
        duration (the seam knows real progress from the runner); a node whose
        free VRAM is below the job's budget offers no ETA (it needs the
        teacher to free VRAM — manual tenancy, §3). No admissible node
        -> ``inf`` (the response then says "waiting for a slot", §9.3).
        """
        job = self.job(job_id)
        if job.state != "queued":
            return 0.0
        best: float | None = None
        for node in self._by_priority():
            if job.target_node is not None and node.id != job.target_node:
                continue
            if not node.enabled:
                continue
            view = node_states.get(node.id)
            if view is None or not view.online or view.free_vram_gb < job.budget_gb:
                continue
            wait = sum(j.est_duration_s for j in self._running_jobs(node.id, slot_class(job.job_type)))
            wait += sum(
                j.est_duration_s
                for j in self._jobs.values()
                if j.state == "queued"
                and j.job_id != job_id
                and self._targets_node(j, node.id)
                and self._queue_key(j) < self._queue_key(job)
            )
            candidate = wait + job.est_duration_s
            best = candidate if best is None else min(best, candidate)
        return float("inf") if best is None else best
