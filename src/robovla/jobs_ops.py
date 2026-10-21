"""Job operations on the ConsoleService (design §9.3/§9.5): submit, kill,
bump, requeue, manual completion, inference-slot claims, and the robot
predict proxy. Thin functions over service + core; all validation errors are
``RobovlaError`` with §9.6 codes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .core.errors import RobovlaError
from .core.scheduler import Job
from .core.spec import build_job_spec, validate_job_spec
from .core.norm_stats import denormalize
from .service import LEASE_S, ConsoleService, _dt, norm_stats_obj

_OVERRIDABLE = ("rank", "alpha", "lr", "epochs", "image_aug", "seed")
_SUBMIT_TYPES = ("smoke", "train", "validate", "merge", "crosscheck")


def require_ticket(svc: ConsoleService, user: dict[str, Any],
                   ticket_id: str | None) -> str:
    """train: ticket present, own, not yet linked (400 / 409 rows, §11)."""
    if not ticket_id:
        raise RobovlaError("validation", "train requires an experiment ticket (§9.3)")
    t = svc.storage.get_ticket(ticket_id)
    if t is None or t["owner"] != user["user_id"]:
        raise RobovlaError("not_found", "ticket not found (must be yours)")
    if t["state"] == "linked":
        raise RobovlaError("conflict",
                           f"ticket {ticket_id} is already linked to {t['linked_job']}")
    return ticket_id


def resolve_node(svc: ConsoleService, user: dict[str, Any], node_id: str) -> str:
    """'auto' or a specific node (student-selectable gating, §8.2/§9.3)."""
    if node_id == "auto":
        return "auto"
    node = svc.fleet_node(node_id)
    if not node.enabled:
        raise RobovlaError("conflict", f"node {node_id!r} is disabled")
    if user["role"] != "teacher" and not node.student_selectable:
        raise RobovlaError("forbidden", f"node {node_id!r} is not student-selectable (§8.2)")
    return node_id


def check_batch(svc: ConsoleService, batch: int, node_id: str) -> None:
    """Batch must be in the node's presets (auto: at least one enabled node)."""
    if node_id != "auto":
        presets = svc.fleet_node(node_id).presets.batch
        if batch not in presets:
            raise RobovlaError("budget_unfit",
                               f"batch {batch} not allowed on {node_id} (presets {list(presets)})")
        return
    if not any(batch in n.presets.batch for n in svc.enabled_nodes_priority()):
        raise RobovlaError("budget_unfit", f"batch {batch} fits no enabled node's presets")


def submit_run(svc: ConsoleService, user: dict[str, Any], payload: dict[str, Any],
               idempotency_key: str | None = None) -> dict[str, Any]:
    if svc.maintenance:
        raise RobovlaError("conflict", "console is in maintenance mode (read-only)")
    if idempotency_key:
        existing = svc.storage.job_by_idempotency(idempotency_key, svc.clock())
        if existing:
            position = svc.scheduler.position(existing["job_id"], _node_states(svc))
            return {"job_id": existing["job_id"], "position": position, "eta_s": 0.0,
                    "deduped": True}
    job_type = payload.get("job_type")
    if job_type not in _SUBMIT_TYPES:
        raise RobovlaError("validation",
                           "job_type must be smoke|train|validate|merge|crosscheck")
    ds = svc.own_dataset(user, payload.get("dataset_id"))
    if ds["status"] != "ready":
        raise RobovlaError("dataset_not_ready",
                           f"dataset {ds['dataset_id']!r} is {ds['status']}")
    ticket_id = require_ticket(svc, user, payload.get("ticket_id")) if job_type == "train" else None
    node_id = resolve_node(svc, user, payload.get("node_id", "auto"))
    limits = svc.student_limits(user)
    counts = svc.scheduler.counts(user["user_id"])
    if counts["running"] >= limits["max_running"]:
        raise RobovlaError("limit_exceeded",
                           "max running reached (an active inference slot counts)")
    if counts["queued"] >= limits["max_queued"]:
        raise RobovlaError("limit_exceeded", "max queued jobs reached")

    cfg = payload.get("config", {})
    batch = int(cfg.get("batch", 4))
    check_batch(svc, batch, node_id)
    run_now = bool(payload.get("run_now", False))
    if run_now and user["role"] not in ("teacher",) and not limits["run_now"]:
        raise RobovlaError("forbidden", "run-now is not enabled for your account")

    first = svc.enabled_nodes_priority()
    if not first:
        raise RobovlaError("conflict", "no enabled nodes in the fleet")
    pinned = svc.fleet_node(node_id) if node_id != "auto" else None
    anchor = pinned or first[0]  # display node until dispatch finalizes (§3)
    budget = svc.budget_for(job_type, batch, pinned)
    max_min = min(int(cfg.get("max_run_min", limits["max_run_min"])), limits["max_run_min"])
    overrides = {k: cfg[k] for k in _OVERRIDABLE if k in cfg}
    job_id = svc.new_job_id()
    spec = build_job_spec(
        job_id=job_id, student=user["user_id"], job_type=job_type,
        dataset=ds["dataset_id"], node_id=anchor.id,
        node_agent=f"{anchor.agent.lan_ip}:{anchor.agent.port}",
        budget_gb=budget, batch=batch, quant=anchor.gpu.quant,
        overrides=overrides, max_min=max_min, idempotency_key=idempotency_key,
    )
    svc.storage.insert_job(
        job_id, spec.to_dict(), "queued", svc._monotonic(),
        run_now=run_now, ticket_id=ticket_id,
        idempotency_key=idempotency_key, created_at=svc.clock())
    svc.scheduler.add(Job(
        job_id=job_id, student=user["user_id"], job_type=job_type,
        target_node=node_id if node_id != "auto" else None,
        run_now=run_now, submitted_at=svc._monotonic(), budget_gb=budget,
        est_duration_s=svc.estimate(spec.to_dict())))
    if job_type == "train" and ticket_id:
        t = svc.storage.get_ticket(ticket_id)
        svc.storage.save_ticket({**t, "linked_job": job_id, "state": "linked",
                                 "updated_at": svc.clock()})
    states = _node_states(svc)
    blockers = svc.scheduler.admission_blockers(job_id, states)
    position = svc.scheduler.position(job_id, states)
    eta = svc.scheduler.eta_s(job_id, states)
    resp: dict[str, Any] = {
        "job_id": job_id, "position": position,
        "eta_s": None if eta == float("inf") else round(eta),
    }
    if blockers:
        resp["waiting"] = "slot"
        resp["blockers"] = blockers
    svc.bus.publish("class", "queue", {"job_id": job_id, "event": "submitted",
                                       "student": user["user_id"]})
    svc.audit(user["user_id"], "run.submit", job_id, None,
              {"job_type": job_type, "dataset": ds["dataset_id"], "node": node_id})
    return resp


def _node_states(svc: ConsoleService) -> dict:
    """Scheduler node-state view from the service's last refresh (§3)."""
    return {n.id: svc.node_state(n.id) for n in svc.fleet.enabled_nodes()}


async def kill_job(svc: ConsoleService, job_id: str, actor: str) -> dict[str, Any]:
    row = svc.storage.get_job(job_id)
    if row is None:
        raise RobovlaError("not_found", f"unknown job: {job_id}")
    if row["state"] in ("running", "finishing"):
        runner = svc.runners.get(row["node_id"] or "")
        if runner is not None:
            try:
                if row["spec"]["type"] == "infer":
                    if row.get("slot_handle"):
                        await asyncio.to_thread(runner.release, row["slot_handle"])
                else:
                    await asyncio.to_thread(runner.kill, job_id)
            except RobovlaError as e:
                svc.audit(actor, "kill.agent_failed", job_id, None, {"error": e.message})
        svc.scheduler.transition(job_id, "kill")
        svc.storage.set_job(job_id, state="killed", terminal_at=svc.clock(),
                            slot_handle=None)
    else:  # queued — no agent involved
        svc.scheduler.transition(job_id, "kill")
        svc.storage.set_job(job_id, state="killed", terminal_at=svc.clock())
    svc.release_ticket_on_terminal(row.get("ticket_id"), job_id, "killed")
    svc.bus.publish(job_id, "state", {"state": "killed"})
    svc.bus.publish("class", "queue", {"job_id": job_id, "event": "killed"})
    return svc.storage.get_job(job_id)


def bump(svc: ConsoleService, job_id: str, actor: str) -> dict[str, Any]:
    """Run-now: front of the line after the current completion (§3, no
    preemption). Teacher only (routed via RBAC)."""
    row = svc.storage.get_job(job_id)
    if row is None:
        raise RobovlaError("not_found", f"unknown job: {job_id}")
    svc.scheduler.run_now(job_id)  # validates queued-only
    svc.audit(actor, "job.bump", job_id, None, {"run_now": True})
    svc.bus.publish("class", "queue", {"job_id": job_id, "event": "bumped"})
    return svc.storage.get_job(job_id)


def requeue(svc: ConsoleService, job_id: str, actor: str,
            node_id: str | None) -> dict[str, Any]:
    """Pin (or clear the pin on) a queued job; budget re-derived for the pin."""
    row = svc.storage.get_job(job_id)
    if row is None:
        raise RobovlaError("not_found", f"unknown job: {job_id}")
    if row["state"] != "queued":
        raise RobovlaError("conflict", "requeue applies to queued jobs")
    node = None
    if node_id is not None:
        node = svc.fleet_node(node_id)
        if not node.enabled:
            raise RobovlaError("conflict", f"node {node_id!r} is disabled")
    svc.scheduler.set_target(job_id, node_id)
    if node is not None:
        spec = dict(row["spec"])
        spec["limits"] = {**spec["limits"],
                          "budget_gb": svc.estimate_on(
                              spec["type"], spec.get("optim", {}).get("batch"), node)}
        svc.storage.set_job_spec(job_id, spec)
    svc.storage.set_job(job_id, node_id=node_id)  # durable pin (SPOF, §10)
    svc.audit(actor, "job.requeue", job_id, {"node": row["node_id"]}, {"node": node_id})
    svc.bus.publish("class", "queue", {"job_id": job_id, "event": "requeued",
                                       "node_id": node_id})
    return svc.storage.get_job(job_id)


def complete(svc: ConsoleService, job_id: str, actor: str, note: str) -> dict[str, Any]:
    """Manual completion: work done outside the console (§8.3)."""
    row = svc.storage.get_job(job_id)
    if row is None:
        raise RobovlaError("not_found", f"unknown job: {job_id}")
    if row["state"] in ("done", "failed", "killed"):
        raise RobovlaError("conflict", f"job {job_id} is {row['state']}")
    svc.storage.set_job(job_id, state="done", terminal_at=svc.clock(), note=note)
    jobs = [j for j in svc.scheduler.jobs() if j.job_id != job_id]
    fresh = type(svc.scheduler)(svc.fleet.nodes)
    for j in jobs:
        fresh.add(j)
    svc._scheduler = fresh
    svc.bus.publish(job_id, "state", {"state": "done", "manual": True})
    svc.bus.publish("class", "queue", {"job_id": job_id, "event": "done"})
    svc.audit(actor, "job.complete", job_id, {"state": row["state"]},
              {"state": "done", "note": note})
    return svc.storage.get_job(job_id)


def annotate(svc: ConsoleService, job_id: str, actor: str, note: str) -> dict[str, Any]:
    row = svc.storage.get_job(job_id)
    if row is None:
        raise RobovlaError("not_found", f"unknown job: {job_id}")
    svc.storage.set_job(job_id, note=note)
    svc.audit(actor, "job.note", job_id, {"note": row["note"]}, {"note": note})
    return svc.storage.get_job(job_id)


# -- inference slots (§9.5) -------------------------------------------------------------


def claim_slot(svc: ConsoleService, user: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Create an infer job for a done run (auto first-fit; no preemption)."""
    if svc.maintenance:
        raise RobovlaError("conflict", "console is in maintenance mode (read-only)")
    row = svc.storage.get_job(run_id)
    if row is None:
        raise RobovlaError("not_found", f"unknown run: {run_id}")
    if user["role"] != "teacher" and row["spec"]["student"] != user["user_id"]:
        raise RobovlaError("forbidden", "not your run")
    if row["state"] != "done":
        raise RobovlaError("conflict", f"run {run_id} is {row['state']}; only done runs serve")
    svc.adapter_path(run_id)  # must exist
    limits = svc.student_limits(user)
    counts = svc.scheduler.counts(user["user_id"])
    if counts["running"] >= limits["max_running"]:
        raise RobovlaError("limit_exceeded",
                           "max running reached (an active inference slot counts)")
    budget = svc.budget_for("infer", None)
    job_id = svc.new_job_id()
    spec = dict(row["spec"])
    spec["job_id"] = job_id  # the infer job IS a new job id (not the run's)
    spec["type"] = "infer"
    spec["limits"] = {**spec["limits"], "budget_gb": budget}
    spec = validate_job_spec(spec)
    svc.storage.insert_job(job_id, spec.to_dict(), "queued", svc._monotonic(),
                           created_at=svc.clock(), run_id=run_id)
    svc.scheduler.add(Job(job_id=job_id, student=user["user_id"], job_type="infer",
                          target_node=None, run_now=False,
                          submitted_at=svc._monotonic(), budget_gb=budget,
                          est_duration_s=0.0))
    svc.audit(user["user_id"], "slot.claim", job_id, None, {"run_id": run_id})
    svc.bus.publish("class", "slots", {"job_id": job_id, "event": "claimed",
                                       "student": user["user_id"]})
    return {"job_id": job_id, "run_id": run_id}


def slot_board(svc: ConsoleService) -> list[dict[str, Any]]:
    out = []
    for row in svc.storage.list_jobs(state_in=("queued", "running")):
        if row["spec"]["type"] != "infer":
            continue
        out.append({
            "job_id": row["job_id"], "student": row["spec"]["student"],
            "run_id": row["run_id"], "state": row["state"], "node_id": row["node_id"],
            "position": svc.scheduler.position(row["job_id"], _node_states(svc))
            if row["state"] == "queued" else 0,
        })
    return out


def robot_predict(svc: ConsoleService, user: dict[str, Any], image_b64: str,
                  prompt: str) -> dict[str, Any]:
    """The ONLY robot-facing path (§9.5): token + active-slot lease check,
    proxy to the node agent, un-normalize with the dataset's stats."""
    if not image_b64:
        raise RobovlaError("validation", "image (base64) is required")
    if not prompt:
        raise RobovlaError("validation", "prompt is required")
    mine = [j for j in svc.storage.list_jobs(state_in=("running",))
            if j["spec"]["type"] == "infer"
            and j["spec"]["student"] == user["user_id"] and j.get("slot_handle")]
    if not mine:
        raise RobovlaError("forbidden", "no active inference slot for this token")
    job = mine[0]
    now = svc.clock()
    if job.get("last_used_at"):
        idle = (_dt(now) - _dt(job["last_used_at"])).total_seconds()
        if idle > LEASE_S:
            row = svc.storage.get_job(job["job_id"])
            svc._release_infer_job(row)
            raise RobovlaError("conflict", "slot lease expired (idle > 300 s)")
    runner = svc.runners.get(job["node_id"] or "")
    if runner is None:
        raise RobovlaError("node_offline", "slot node is offline")
    out = runner.predict(job["slot_handle"], image_b64, prompt)
    norm = list(out["action_norm"])
    if len(norm) != 7:
        raise RobovlaError("internal", "agent returned a non-7-dim action")
    robot = norm
    ds = svc.storage.get_dataset(job["spec"]["dataset"])
    if ds and ds.get("norm_stats"):
        robot = list(denormalize(norm, norm_stats_obj(ds["norm_stats"])))
    svc.storage.set_job(job["job_id"], last_used_at=now)
    return {"action_norm": norm, "action_robot": robot,
            "latency_ms": out.get("latency_ms", 0.0)}
