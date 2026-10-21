"""ConsoleService — the service seam between the pure core and the I/O seams
(design §11: "the service seam dispatches admitted jobs through the runner").

Owns: the scheduler (rebuilt on startup/config reload), per-node runners,
measured node state (manual tenancy, §3), the SSE event bus, the dispatch
loop, lease expiry, GPU view, and audit stamps. Time is injected
(``clock`` -> RFC 3339, ``monotonic`` -> float) so tests drive ticks
deterministically; the async loop is for production.

Job *operations* (submit/kill/bump/requeue/complete, slots, robot predict)
live in :mod:`robovla.jobs_ops`; dataset prep in :mod:`robovla.datasets_ops`
— this file is the loop and the shared state.

Budget model (§3): a job's ``budget_gb`` is the VRAM estimate for its config
on its node. Pinned jobs use the pinned node's base budgets; auto jobs use the
*conservative* estimate (max across enabled nodes) so the core scheduler's
single per-job budget is safe on whatever node admits it. Dispatch re-derives
the budget on the admitted node; if VRAM dropped since admission the dispatch
is skipped and retried next tick (no preemption, ever).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .core.budgets import estimate_job_gb
from .core.config import FleetConfig, NodeConfig
from .core.errors import RobovlaError
from .core.norm_stats import NormStats
from .core.scheduler import Job, NodeState, Scheduler, default_duration_s
from .core.states import is_terminal
from .core.spec import validate_job_spec
from .core.ticket import Ticket, on_run_terminal
from .ratelimit import RateLimiter
from .runner import Runner
from .sse import EventBus
from .storage import Storage

#: §9.5: idle inference slot auto-release (the Pi watchdog stops the robot on
#: stale actions; the console frees the GPU).
LEASE_S = 300
MAX_UPLOADS_PER_STUDENT = 2  # §9.8: 2 concurrent uploads/student


def utcnow_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ticket_obj(t: dict[str, Any]) -> Ticket:
    """storage row -> core Ticket (one adapter, used for lifecycle calls)."""
    return Ticket(
        ticket_id=t["ticket_id"], owner=t["owner"], hypothesis=t["hypothesis"],
        config_summary=t["config_summary"], prediction=t["prediction"],
        linked_job=t.get("linked_job"), outcome=t.get("outcome"), state=t["state"],
        created_at=t["created_at"], updated_at=t["updated_at"])


def norm_stats_obj(d: dict[str, Any]) -> NormStats:
    return NormStats(min=tuple(d["min"]), max=tuple(d["max"]))


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class ConsoleService:
    def __init__(
        self,
        data_dir: str | Path,
        fleet: FleetConfig,
        storage: Storage,
        runners: dict[str, Runner],
        *,
        clock: Callable[[], str] | None = None,
        monotonic: Callable[[], float] | None = None,
        tick_s: float = 0.5,
        heartbeat_s: float = 15.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.fleet = fleet
        self.storage = storage
        self.runners = dict(runners)
        self.clock = clock or utcnow_rfc3339
        self._monotonic = monotonic or time.monotonic
        self.tick_s = tick_s
        self.heartbeat_s = heartbeat_s
        self.storage.set_session_ttl_h(fleet.console.session_ttl_h)
        self.fleet_path = None  # set by the app factory (admin config reload)
        self.bus = EventBus()
        self.limits = RateLimiter(now_ms=lambda: int(self._monotonic() * 1000))
        self._scheduler = self._build_scheduler()
        self._node_states: dict[str, NodeState] = {}
        self._procs: dict[str, list[dict]] = {}
        self._progress: dict[str, dict[str, Any]] = {}
        self._task: asyncio.Task | None = None
        self._started_at = self._monotonic()

    # -- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(self.tick_s)

    def _build_scheduler(self) -> Scheduler:
        """Seed the in-memory scheduler from durable state (SPOF: jobs survive
        a console rebuild, §10). Queued jobs keep their pin (node_id)."""
        s = Scheduler(self.fleet.nodes)
        for row in self.storage.list_jobs(state_in=("queued", "running", "finishing")):
            spec = row["spec"]
            s.add(Job(
                job_id=row["job_id"],
                student=spec["student"],
                job_type=spec["type"],
                target_node=row["node_id"] if row["state"] == "queued" else None,
                run_now=row["run_now"],
                submitted_at=row["submitted_at"],
                budget_gb=spec["limits"]["budget_gb"],
                est_duration_s=self.estimate(spec),
                state=row["state"],
                node_id=row["node_id"],
            ))
        return s

    def estimate(self, spec: dict[str, Any]) -> float:
        if spec["type"] in ("train", "crosscheck"):
            return default_duration_s(spec["type"], spec["optim"]["batch"],
                                      spec["optim"]["epochs"])
        return default_duration_s(spec["type"])

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    # -- audit / limits / maintenance -----------------------------------------------

    def audit(self, actor: str, action: str, target: str,
              before: dict | None = None, after: dict | None = None) -> None:
        self.storage.append_audit(self.clock(), actor, action, target, before, after)

    @property
    def maintenance(self) -> bool:
        return self.storage.get_kv("maintenance") == "on"

    def set_maintenance(self, on: bool, actor: str) -> None:
        before = self.maintenance
        self.storage.set_kv("maintenance", "on" if on else "off")
        self.audit(actor, "system.maintenance", "console", {"on": before}, {"on": on})

    def student_limits(self, user: dict[str, Any]) -> dict[str, Any]:
        """Per-student limits: config default unless the account overrides it
        (a stored ``None`` = use the fleet default, §8.1)."""
        d = self.fleet.console.limits.student

        def pick(key: str, default: Any) -> Any:
            v = user.get(key)
            return default if v is None else v

        out = {
            "max_running": pick("max_running", d.max_running),
            "max_queued": pick("max_queued", d.max_queued),
            "max_dataset_bytes": pick("max_dataset_bytes", d.max_dataset_bytes),
            "max_run_min": pick("max_run_min", d.max_run_min),
            "run_now": pick("run_now", d.run_now),
        }
        return {k: (bool(v) if k == "run_now" else v) for k, v in out.items()}

    def check_rate(self, kind: str, subject: str, limit: int, window_s: float) -> None:
        if not self.limits.allow(kind, subject, limit, int(window_s * 1000)):
            raise RobovlaError("rate_limited", f"too many {kind} requests; slow down")

    # -- node state (measured VRAM; tenancy is manual, §3) ---------------------------

    def refresh_nodes(self) -> None:
        states: dict[str, NodeState] = {}
        procs: dict[str, list[dict]] = {}
        for node in self.fleet.enabled_nodes():
            runner = self.runners.get(node.id)
            if runner is None:
                states[node.id] = NodeState(online=False, free_vram_gb=0.0)
                procs[node.id] = []
                continue
            try:
                v = runner.vram()
                states[node.id] = NodeState(online=True, free_vram_gb=float(v["free_gb"]))
                procs[node.id] = v.get("procs", [])
            except RobovlaError:
                states[node.id] = NodeState(online=False, free_vram_gb=0.0)
                procs[node.id] = []
        self._node_states = states
        self._procs = procs

    def node_state(self, node_id: str) -> NodeState:
        return self._node_states.get(node_id, NodeState(online=False, free_vram_gb=0.0))

    # -- the tick: admission -> dispatch -> poll -> leases -----------------------------

    async def tick(self) -> None:
        self.refresh_nodes()
        await self._dispatch_admitted()
        await self._poll_active()
        await self._expire_lease_holds()

    async def _dispatch_admitted(self) -> None:
        for job in self._scheduler.tick(self._node_states):
            await self._dispatch(job)

    async def _dispatch(self, job: Job) -> None:
        node_id = self._scheduler.admission_node(job.job_id, self._node_states)
        if node_id is None:
            return  # still queued (VRAM moved); retry next tick
        node = self.fleet_node(node_id)
        row = self.storage.get_job(job.job_id)
        try:
            spec = self.finalize_spec(row["spec"], node)
            runner = self.runners.get(node_id)
            if runner is None:
                return
            if job.job_type == "infer":
                # serving the adapter IS the job — nothing runs on the agent
                handle = await asyncio.to_thread(
                    runner.serve, row["run_id"], self.adapter_path(row["run_id"]))
                self.storage.set_job(job.job_id, slot_handle=handle)
            else:
                await asyncio.to_thread(runner.submit, spec.to_dict())
        except RobovlaError as e:
            if job.job_type == "infer":
                self.storage.set_job(job.job_id, slot_handle=None)
            self.audit("system", "dispatch.failed", job.job_id,
                       {"node": node_id}, {"error": e.message})
            self.bus.publish("class", "node",
                             {"node_id": node_id, "event": "dispatch_failed"})
            return  # stays queued; auto-reschedule on a fitting runner (§10)
        self.storage.set_job_spec(job.job_id, spec.to_dict())
        self._scheduler.transition(job.job_id, "start", node_id)
        self.storage.set_job(job.job_id, state="running", node_id=node_id)
        self.bus.publish(job.job_id, "state", {"state": "running", "node_id": node_id})
        self.bus.publish("class", "queue", {"job_id": job.job_id, "event": "started",
                                            "node_id": node_id})

    def finalize_spec(self, spec: dict[str, Any], node: NodeConfig):
        """Dispatch finalizes node/agent and re-derives the budget (§3)."""
        d = dict(spec)
        d["node"] = {"id": node.id, "agent": f"{node.agent.lan_ip}:{node.agent.port}"}
        budget = self.estimate_on(d["type"], d.get("optim", {}).get("batch"), node)
        if self.node_state(node.id).free_vram_gb < budget:
            raise RobovlaError("budget_unfit",
                               f"{node.id} free VRAM below {budget:g} GB at dispatch")
        d["limits"] = {**spec["limits"], "budget_gb": budget}
        return validate_job_spec(d)

    # -- polling running jobs ----------------------------------------------------------

    async def _poll_active(self) -> None:
        for row in self.storage.list_jobs(state_in=("running", "finishing")):
            if row["spec"]["type"] == "infer":
                continue  # served adapters have no agent job to poll (lease only)
            runner = self.runners.get(row["node_id"] or "")
            if runner is None:
                continue
            try:
                view = await asyncio.to_thread(runner.poll, row["job_id"])
            except RobovlaError as e:
                await self.terminate(row["job_id"], "failed", reason=e.message)  # §10 fail fast
                continue
            await self._apply_view(row, view)

    async def _apply_view(self, row: dict[str, Any], view: dict[str, Any]) -> None:
        job_id, state = row["job_id"], view["state"]
        if state == "running":
            for m in view.get("metrics", []):
                self.bus.publish(job_id, "metrics", m)
            if view.get("metrics"):
                self._progress[job_id] = {**self._progress.get(job_id, {}),
                                          **view["metrics"][-1]}
            for line in view.get("log", []):
                self.bus.publish(job_id, "log", {"line": line})
            if view.get("val_accuracy") is not None:
                self.bus.publish(job_id, "val", {"accuracy": view["val_accuracy"]})
            if view.get("predict_probe"):
                self._store_probe(job_id, view["predict_probe"])
            if view.get("artifacts"):
                self.store_artifacts(job_id, view["artifacts"])
            return
        if state in ("finishing", "done"):
            if state == "finishing" and row["state"] == "running":
                self._scheduler.transition(job_id, "finish")
                self.storage.set_job(job_id, state="finishing")
            if state == "done":
                self.store_artifacts(job_id, view.get("artifacts", []))
                if view.get("val_accuracy") is not None:
                    self.bus.publish(job_id, "val", {"accuracy": view["val_accuracy"]})
                if view.get("predict_probe"):
                    self._store_probe(job_id, view["predict_probe"])
                await self.terminate(job_id, "done")
            return
        await self.terminate(job_id, "failed",
                             reason=view.get("error") or "agent reported failure")

    def _store_probe(self, job_id: str, probe: list[float]) -> None:
        self.storage.set_job(job_id, predict_probe=probe)
        self.bus.publish(job_id, "predict", {"action_norm": probe})

    def store_artifacts(self, job_id: str, arts: list[dict[str, Any]]) -> None:
        for a in arts:
            self.storage.add_artifact(job_id, a["kind"], a["path"], a.get("nbytes", 0),
                                      at=self.clock())
        self.bus.publish(job_id, "artifacts", {"artifacts": arts})
        self.bus.publish("class", "artifacts", {"job_id": job_id, "count": len(arts)})

    async def terminate(self, job_id: str, final: str, reason: str | None = None) -> None:
        row = self.storage.get_job(job_id)
        if row is None or is_terminal(row["state"]):
            return
        event = "done" if final == "done" else "fail"
        try:
            if row["state"] == "running":
                self._scheduler.transition(job_id, "finish")
            self._scheduler.transition(job_id, event)
        except RobovlaError:
            return  # already terminal (race with kill)
        self.storage.set_job(job_id, state=final, terminal_at=self.clock())
        self._progress.pop(job_id, None)
        self.release_ticket_on_terminal(row.get("ticket_id"), job_id, final)
        data: dict[str, Any] = {"state": final}
        if reason:
            data["reason"] = reason
        self.bus.publish(job_id, "state", data)
        self.bus.publish("class", "queue", {"job_id": job_id, "event": final})

    def release_ticket_on_terminal(self, ticket_id: str | None, job_id: str,
                                   final: str) -> None:
        """The link releases when the linked run reaches a terminal state
        (v0.5.1: the ticket is retained for the record, re-linkable)."""
        if not ticket_id:
            return
        t = self.storage.get_ticket(ticket_id)
        if not t:
            return
        released = on_run_terminal(ticket_obj(t), job_id, final, self.clock())
        if released.state != t["state"] or released.linked_job != t.get("linked_job"):
            self.storage.save_ticket({**t, "linked_job": released.linked_job,
                                      "state": released.state,
                                      "updated_at": released.updated_at})

    # -- leases / GPU view / probes -------------------------------------------------------

    async def _expire_lease_holds(self) -> None:
        now = _dt(self.clock())
        for row in self.storage.list_jobs(state_in=("running",)):
            if row["spec"]["type"] != "infer" or not row.get("last_used_at"):
                continue
            if (now - _dt(row["last_used_at"])).total_seconds() > LEASE_S:
                self._release_infer_job(row)

    def _release_infer_job(self, row: dict[str, Any]) -> None:
        """Release a served inference slot: agent handle, scheduler state,
        storage row, class feed. No-op if already terminal (kill race)."""
        runner = self.runners.get(row["node_id"] or "")
        if runner is not None and row.get("slot_handle"):
            try:
                runner.release(row["slot_handle"])
            except RobovlaError as e:
                self.audit("system", "slot.release_failed", row["job_id"],
                           None, {"error": e.message})
        try:
            self._scheduler.transition(row["job_id"], "kill")
        except RobovlaError:
            return  # already terminal (an explicit kill won the race)
        self.storage.set_job(row["job_id"], state="killed",
                             terminal_at=self.clock(), slot_handle=None)
        self.bus.publish(row["job_id"], "state",
                         {"state": "killed", "reason": "released"})
        self.bus.publish("class", "slots", {"job_id": row["job_id"],
                                            "event": "released"})

    # -- helpers shared with jobs_ops / datasets_ops ---------------------------------------

    def fleet_node(self, node_id: str) -> NodeConfig:
        n = self.fleet.nodes_by_id().get(node_id)
        if n is None:
            raise RobovlaError("not_found", f"unknown node: {node_id}")
        return n

    def enabled_nodes_priority(self) -> list[NodeConfig]:
        return [n for n in self.fleet.nodes_by_priority() if n.enabled]

    def estimate_on(self, job_type: str, batch: int | None, node: NodeConfig) -> float:
        max_batch = max(node.presets.batch) if node.presets.batch else 1
        return estimate_job_gb(node.gpu.budgets.train_gb, node.gpu.budgets.infer_gb,
                               node.gpu.quant, job_type, batch, max_batch)

    def budget_for(self, job_type: str, batch: int | None,
                   node: NodeConfig | None = None) -> float:
        """Admission budget (§3): node-specific, or the conservative max
        across enabled nodes for auto selection."""
        if node is not None:
            return self.estimate_on(job_type, batch, node)
        return max((self.estimate_on(job_type, batch, n)
                    for n in self.enabled_nodes_priority()), default=0.0)

    def new_job_id(self) -> str:
        day = self.clock()[:10].replace("-", "")
        seq = self.storage.jobs_created_on(self.clock()[:10]) + 1
        return f"j_{day}_{seq:04d}"

    def adapter_path(self, run_id: str) -> str:
        arts = [a for a in self.storage.artifacts_for(run_id) if a["kind"] == "adapter"]
        if not arts:
            raise RobovlaError("not_found", f"run {run_id!r} has no adapter artifact")
        return arts[0]["path"]

    def own_dataset(self, user: dict[str, Any], dataset_id: str | None) -> dict[str, Any]:
        if not dataset_id:
            raise RobovlaError("validation", "dataset_id is required")
        ds = self.storage.get_dataset(dataset_id)
        if ds is None:
            raise RobovlaError("not_found", f"unknown dataset: {dataset_id}")
        if user["role"] != "teacher" and ds["owner"] != user["user_id"]:
            raise RobovlaError("forbidden", "not your dataset")
        return ds

    def progress(self, job_id: str) -> dict[str, Any]:
        return dict(self._progress.get(job_id, {}))

    # -- config hot reload (§10: validate -> apply; queued jobs revalidated) --------------

    def reload_config(self, new_fleet: FleetConfig, actor: str,
                      changes: list | None = None) -> None:
        from .core.config import diff_fleet
        if changes is None:
            changes = diff_fleet(self.fleet, new_fleet)
        self.fleet = new_fleet
        self.storage.set_session_ttl_h(new_fleet.console.session_ttl_h)
        self._scheduler = self._build_scheduler()
        for c in changes:
            self.audit(actor, "config.change", c.scope, {c.field: c.before},
                       {c.field: c.after})
        self.bus.publish("class", "config", {"changes": [c.to_dict() for c in changes]})
