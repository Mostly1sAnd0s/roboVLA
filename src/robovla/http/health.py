"""/api/health — LAN-public (classroom display, §8.5): service uptime,
per-node state (online, free VRAM, queue depth), maintenance flag."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .auth import svc

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health(request: Request) -> dict:
    s = svc(request)
    depth = {n: 0 for n in s.fleet.nodes_by_id()}
    for j in s.scheduler.jobs():
        if j.state == "queued" and j.target_node:
            depth[j.target_node] = depth.get(j.target_node, 0) + 1
    return {
        "service": {
            "up": True,
            "uptime_s": round(s._monotonic() - s._started_at, 1),
            "maintenance": s.maintenance,
        },
        "nodes": [
            {"id": n.id, "label": n.label, "enabled": n.enabled,
             "online": s.node_state(n.id).online,
             "free_gb": round(s.node_state(n.id).free_vram_gb, 2),
             "vram_gb": n.gpu.vram_gb,
             "queue_depth": depth.get(n.id, 0)}
            for n in s.fleet.nodes
        ],
    }
