"""Node-ops on the ConsoleService (design §3/§8.2/§8.5): measured VRAM view
with orphan alerts, the join probe, and the agent-scoped VRAM purge. Admin
surface only — every action here is audited.
"""

from __future__ import annotations

from typing import Any

from .core.errors import RobovlaError
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
