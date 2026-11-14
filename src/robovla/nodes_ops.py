"""Node-ops on the ConsoleService (design §3/§8.2/§8.5): measured VRAM view
with orphan alerts, the join probe, and the agent-scoped VRAM purge. Admin
surface only — every action here is audited.
"""

from __future__ import annotations

from typing import Any

from .core.errors import RobovlaError
from .core.scheduler import default_duration_s
from .core.spec import model_family
from .service import ConsoleService


def gpu_view(svc: ConsoleService) -> dict[str, Any]:
    """Per-node measured VRAM + holding processes + orphan alert (§3, §8.5).
    Orphans = agent-spawned procs while the node has no running job
    (agent reaps its jobs; a leftover = un-reaped orphan)."""
    out: dict[str, Any] = {}
    for node in svc.fleet.enabled_nodes():
        st = svc.node_state(node.id)
        procs = svc._procs.get(node.id, [])
        running = any(j.node_id == node.id and j.state in ("running", "finishing")
                      for j in svc.scheduler.jobs())
        orphans = [p for p in procs if p.get("agent_spawned") and not running]
        out[node.id] = {
            "online": st.online, "free_gb": st.free_vram_gb,
            "vram_gb": node.gpu.vram_gb, "procs": procs,
            "orphan_alert": bool(orphans),
            "orphan_pids": [p.get("pid") for p in orphans],
        }
    return out


def endpoint_board(svc: ConsoleService, *, job_type: str = "train",
                   batch: int | None = None, epochs: int | None = None,
                   is_teacher: bool = False) -> list[dict[str, Any]]:
    """The training-endpoint board (§7.3): one row per node with measured
    load, both queues (pinned vs auto), and the wait a job of the caller's
    shape would face there.

    Students use this to choose which queue to join; the choice itself is
    still the submit-time ``node_id`` gate of §9.3 (``student_selectable``,
    §8.2) — this view tells the truth about why a node is not pickable
    rather than hiding it.
    """
    states = {n.id: svc.node_state(n.id) for n in svc.fleet.enabled_nodes()}
    out = []
    for node in svc.fleet.nodes_by_priority():
        st = svc.node_state(node.id)
        q = svc.scheduler.node_queue(node.id)
        # why this endpoint is not pickable right now, in plain words (§7.3):
        # busy is a WAIT (eta_start_s), never a refusal
        reasons: list[str] = []
        if not node.enabled:
            reasons.append("disabled")
        elif not st.online:
            reasons.append("offline")
        elif not is_teacher and not node.student_selectable:
            reasons.append("teacher has not opened this node to students")
        elif batch is not None and batch not in node.presets.batch:
            reasons.append(f"batch {batch} is not in this node's presets")
        # the caller's job shape, priced on THIS node
        est_run_s = None
        if job_type in ("train", "crosscheck") and batch and epochs:
            est_run_s = round(default_duration_s(job_type, batch, epochs))
        wait = (svc.scheduler.wait_for_new_s(
            node.id, job_type, family=model_family(node.model), node_states=states)
            if job_type else None)
        eta_start_s = round(wait) if wait is not None else None
        out.append({
            "id": node.id, "label": node.label, "model": node.model,
            "quant": node.gpu.quant, "vram_gb": node.gpu.vram_gb,
            "free_gb": round(st.free_vram_gb, 2),
            "used_gb": round(max(0.0, node.gpu.vram_gb - st.free_vram_gb), 2),
            "used_pct": round(100.0 * (1 - st.free_vram_gb / node.gpu.vram_gb), 1)
            if node.gpu.vram_gb else 0.0,
            "online": st.online, "enabled": node.enabled,
            "priority": node.priority,
            "batch_presets": list(node.presets.batch),
            "train_slots": node.gpu.train_slots, "train_used": q["train_used"],
            "infer_slots": node.gpu.infer_slots, "infer_used": q["infer_used"],
            "running": q["running"], "queued": q["queued"],
            "auto_queued": q["auto_queued"],
            "student_selectable": node.student_selectable,
            "selectable": not reasons,
            "reasons": reasons,
            "est_run_s": est_run_s,
            "eta_start_s": eta_start_s,
            # teacher-only: the agent endpoint (debugging, §8.2)
            "agent": (f"{node.agent.lan_ip}:{node.agent.port}") if is_teacher else None,
        })
    return out


def node_probe(svc: ConsoleService, node_id: str) -> dict[str, Any]:
    """Join probe (§8.2): agent reachable, image tag matches, GPU visible,
    disk >= 50 GB, model dir present."""
    node = svc.fleet_node(node_id)
    runner = svc.runners.get(node_id)
    if runner is None:
        return {"node_id": node_id, "ok": False, "reason": "no agent configured"}
    try:
        health = runner.health()
    except RobovlaError as e:
        return {"node_id": node_id, "ok": False, "reason": e.message}
    return {
        "node_id": node_id,
        "ok": bool(health.get("ok"))
        and health.get("image") == node.image
        and float(health.get("disk_free_gb", 0)) >= 50
        and bool(health.get("model_dir")),
        "image": health.get("image"), "expected_image": node.image,
        "gpu": health.get("gpu"), "disk_free_gb": health.get("disk_free_gb"),
        "model_dir": health.get("model_dir"),
    }


def purge_vram(svc: ConsoleService, node_id: str, pids: list[int],
               actor: str) -> dict[str, Any]:
    """Agent-scoped orphan purge (teacher-confirmed, §8.2). Host-level
    processes are never touched. Audited once here (single call site)."""
    svc.fleet_node(node_id)
    runner = svc.runners.get(node_id)
    if runner is None:
        raise RobovlaError("node_offline", f"no agent for {node_id}")
    agent_pids = {p["pid"] for p in svc._procs.get(node_id, [])
                  if p.get("agent_spawned")}
    bad = [p for p in pids if p not in agent_pids]
    if bad:
        raise RobovlaError("forbidden", f"pids {bad} are not agent-scoped (§8.2)")
    runner.purge_procs(pids)
    svc.audit(actor, "node.purge_vram", node_id, None, {"pids": pids})
    return {"node_id": node_id, "purged": pids}
