"""Admin API (design §9.7): every teacher control from §8, audited.

Role = teacher (``require_admin`` = role + 60/min rate limit in one
dependency); every action writes the append-only audit log (actor,
timestamp, action, target, before->after). The node registry *is* the fleet
config (§10): node endpoints rebuild + revalidate + hot-reload the fleet;
node *host* lifecycle (container restart, image update, GPU reset) is
deliberately NOT here — host-side teacher one-liners (§8.2).
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Request

from ..core.config import FleetConfig, NodeConfig, diff_fleet, parse_fleet
from ..core.errors import RobovlaError
from .auth import require_admin, svc
from .. import backup, jobs_ops, nodes_ops

router = APIRouter(prefix="/api/admin", tags=["admin"])


# -- fleet config <-> raw YAML (one mapping direction each, no drift) ------------


def _node_raw(n: NodeConfig) -> dict:
    return {
        "id": n.id, "label": n.label, "image": n.image, "workdir": n.workdir,
        "agent": {"lan_ip": n.agent.lan_ip, "port": n.agent.port,
                  "token_ref": n.agent.token_ref},
        "gpu": {"vram_gb": n.gpu.vram_gb, "quant": n.gpu.quant,
                "train_slots": n.gpu.train_slots, "infer_slots": n.gpu.infer_slots,
                "budgets": {"train_gb": n.gpu.budgets.train_gb,
                            "infer_gb": n.gpu.budgets.infer_gb}},
        "presets": {"batch": list(n.presets.batch)},
        "priority": n.priority,
        "student_selectable": n.student_selectable,
        "enabled": n.enabled,
    }


def _fleet_raw(fleet: FleetConfig) -> dict:
    c = fleet.console
    return {
        "version": fleet.version,
        "console": {
            "bind": c.bind, "port": c.port, "tls": c.tls, "data_dir": c.data_dir,
            "session_ttl_h": c.session_ttl_h,
            "limits": {"student": {"max_running": c.limits.student.max_running,
                                   "max_queued": c.limits.student.max_queued,
                                   "max_dataset_bytes": c.limits.student.max_dataset_bytes,
                                   "max_run_min": c.limits.student.max_run_min,
                                   "run_now": c.limits.student.run_now},
                       "admin_rate": c.limits.admin_rate},
            "retention": {"run_logs_per_student": c.retention.run_logs_per_student,
                          "merged_export_days": c.retention.merged_export_days,
                          "backup": {"enabled": c.retention.backup.enabled,
                                     "at": c.retention.backup.at,
                                     "dest": c.retention.backup.dest}},
        },
        "nodes": [_node_raw(n) for n in fleet.nodes],
    }


def _apply_fleet(s, raw: dict, actor: str) -> FleetConfig:
    new = parse_fleet(yaml.safe_dump(raw))
    changes = diff_fleet(s.fleet, new)
    _write_fleet_file(s, raw)
    s.reload_config(new, actor, changes)
    return new


def _write_fleet_file(s, raw: dict) -> None:
    if s.fleet_path:
        Path(s.fleet_path).write_text(yaml.safe_dump(raw, sort_keys=False))


def _new_agent_token(s, node_id: str) -> str:
    value = "agent_" + secrets.token_urlsafe(24)
    s.storage.set_agent_token(node_id, hash_password(value),
                              token_lookup_key(value), at=s.clock())
    return value


def _patch_fleet_node(s, node_id: str, patch: dict, actor: str) -> dict:
    raw = _fleet_raw(s.fleet)
    nodes = {n["id"]: n for n in raw["nodes"]}
    if node_id not in nodes:
        raise RobovlaError("not_found", f"unknown node: {node_id}")
    for k, v in patch.items():
        if k in ("gpu", "agent", "presets") and isinstance(v, dict):
            nodes[node_id][k] = {**nodes[node_id].get(k, {}), **v}
        else:
            nodes[node_id][k] = v
    _apply_fleet(s, raw, actor)
    return {"node_id": node_id}


# -- nodes (§8.2) -------------------------------------------------------------------


@router.get("/nodes")
async def list_nodes(request: Request, user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    view = nodes_ops.gpu_view(s)
    return {"nodes": [
        {**_node_raw(n), "online": view.get(n.id, {}).get("online"),
         "free_gb": view.get(n.id, {}).get("free_gb"),
         "orphan_alert": view.get(n.id, {}).get("orphan_alert"),
         "orphan_pids": view.get(n.id, {}).get("orphan_pids", []),
         "queue_depth": sum(1 for j in s.scheduler.jobs()
                            if j.state == "queued" and j.target_node == n.id)}
        for n in s.fleet.nodes]}


@router.post("/nodes")
async def add_node(request: Request, user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    body = await request.json()
    nid = body.get("id", "")
    if not nid:
        raise RobovlaError("validation", "node id is required")
    raw = _fleet_raw(s.fleet)
    if any(n["id"] == nid for n in raw["nodes"]):
        raise RobovlaError("conflict", f"node {nid!r} exists")
    raw["nodes"].append(_node_raw_default(nid, body))
    new = _apply_fleet(s, raw, user["user_id"])
    token = _new_agent_token(s, nid)
    # agent endpoint must be reachable in the service (HttpRunner in prod)
    node = new.nodes_by_id()[nid]
    if nid not in s.runners:
        from ..runner import HttpRunner
        s.runners[nid] = HttpRunner(f"http://{node.agent.lan_ip}:{node.agent.port}",
                                    token)
    s.audit(user["user_id"], "node.add", nid, None, _node_raw(node))
    return {"node_id": nid, "agent_token": token, "shown_once": True,
            "enabled": node.enabled}


def _node_raw_default(nid: str, body: dict) -> dict:
    gpu = body.get("gpu", {})
    budgets = gpu.get("budgets", {})
    return {
        "id": nid, "label": body.get("label", nid),
        "image": body.get("image", "robovla-runner-cuda:2026.10"),
        "workdir": body.get("workdir", "/home/adam/robovla"),
        "agent": {"lan_ip": body.get("lan_ip", "127.0.0.1"),
                  "port": int(body.get("port", 8601)),
                  "token_ref": f"{nid}-agent"},
        "gpu": {"vram_gb": float(gpu.get("vram_gb", 32)),
                "quant": gpu.get("quant", "bnb8"),
                "train_slots": int(gpu.get("train_slots", 1)),
                "infer_slots": int(gpu.get("infer_slots", 1)),
                "budgets": {"train_gb": float(budgets.get("train_gb", 18)),
                            "infer_gb": float(budgets.get("infer_gb", 11))}},
        "presets": {"batch": body.get("batch", [1, 4, 8])},
        "priority": int(body.get("priority", 0)),
        "student_selectable": bool(body.get("student_selectable", False)),
        "enabled": bool(body.get("enabled", False)),
    }


@router.patch("/nodes/{nid}")
async def patch_node(nid: str, request: Request,
                     user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    body = await request.json()
    out = _patch_fleet_node(s, nid, body, user["user_id"])
    s.audit(user["user_id"], "node.patch", nid, None, body)
    return out


@router.delete("/nodes/{nid}")
async def remove_node(nid: str, request: Request,
                      user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    if nid not in s.fleet.nodes_by_id():
        raise RobovlaError("not_found", f"unknown node: {nid}")
    raw = _fleet_raw(s.fleet)
    raw["nodes"] = [n for n in raw["nodes"] if n["id"] != nid]
    _apply_fleet(s, raw, user["user_id"])
    s.audit(user["user_id"], "node.remove", nid)
    return {"ok": True}


@router.post("/nodes/{nid}/probe")
async def probe_node(nid: str, request: Request,
                     user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    out = nodes_ops.node_probe(s, nid)
    s.audit(user["user_id"], "node.probe", nid, None, {"ok": out.get("ok")})
    return out


@router.post("/nodes/{nid}/purge-vram")
async def purge_node_vram(nid: str, request: Request,
                          user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    body = await request.json()
    # audited once inside nodes_ops.purge_vram (single call site)
    return nodes_ops.purge_vram(s, nid, body.get("pids", []), user["user_id"])


@router.get("/nodes/{nid}/logs")
async def node_logs(nid: str, request: Request,
                    user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    runner = s.runners.get(nid)
    if runner is None:
        raise RobovlaError("node_offline", f"no agent configured for {nid}")
    try:
        log = runner.agent_log(500)
    except RobovlaError as e:
        raise e
    return {"node_id": nid, "log": log}


# -- jobs (§8.3) --------------------------------------------------------------------


@router.get("/jobs")
async def list_jobs(request: Request, user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    return {"jobs": [
        {"job_id": r["job_id"], "student": r["spec"]["student"],
         "job_type": r["spec"]["type"], "state": r["state"],
         "node_id": r["node_id"], "submitted_at": r["submitted_at"]}
        for r in s.storage.list_jobs()]}


@router.post("/jobs/{job_id}/bump")
async def job_bump(job_id: str, request: Request,
                   user: dict = Depends(require_admin)) -> dict:
    return jobs_ops.bump(svc(request), job_id, user["user_id"])


@router.post("/jobs/{job_id}/kill")
async def job_kill(job_id: str, request: Request,
                   user: dict = Depends(require_admin)) -> dict:
    return await jobs_ops.kill_job(svc(request), job_id, user["user_id"])


@router.post("/jobs/{job_id}/requeue")
async def job_requeue(job_id: str, request: Request,
                      user: dict = Depends(require_admin)) -> dict:
    body = await request.json()
    return jobs_ops.requeue(svc(request), job_id, user["user_id"], body.get("node_id"))


@router.post("/jobs/{job_id}/complete")
async def job_complete(job_id: str, request: Request,
                       user: dict = Depends(require_admin)) -> dict:
    body = await request.json()
    note = body.get("note", "")
    if not note:
        raise RobovlaError("validation", "note is required (manual completion, §8.3)")
    return jobs_ops.complete(svc(request), job_id, user["user_id"], note)


@router.post("/jobs/{job_id}/note")
async def job_note(job_id: str, request: Request,
                   user: dict = Depends(require_admin)) -> dict:
    body = await request.json()
    return jobs_ops.annotate(svc(request), job_id, user["user_id"],
                             body.get("note", ""))


# -- datasets / artifacts (§8.4) --------------------------------------------------------


@router.delete("/datasets/{dataset_id:path}")
async def purge_dataset(dataset_id: str, request: Request,
                        user: dict = Depends(require_admin)) -> dict:
    """Two-step purge: ?confirm=true required; runs that used it keep their
    artifacts (run config is self-contained)."""
    s = svc(request)
    if request.query_params.get("confirm") != "true":
        raise RobovlaError("validation", "two-step confirm: retry with ?confirm=true")
    ds = s.storage.get_dataset(dataset_id)
    if ds is None:
        raise RobovlaError("not_found", f"unknown dataset: {dataset_id}")
    import shutil
    root = Path(s.data_dir) / "datasets" / dataset_id.replace("/", "__")
    if root.exists():
        shutil.rmtree(root)
    _delete_dataset_row(s, dataset_id)
    s.audit(user["user_id"], "dataset.purge", dataset_id,
            {"status": ds["status"]}, None)
    return {"ok": True, "dataset_id": dataset_id}


def _delete_dataset_row(s, dataset_id: str) -> None:
    s.storage.set_dataset(dataset_id, status="purged", at=s.clock())


@router.post("/datasets/{dataset_id:path}/reject")
async def reject_dataset(dataset_id: str, request: Request,
                         user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    ds = s.storage.get_dataset(dataset_id)
    if ds is None:
        raise RobovlaError("not_found", f"unknown dataset: {dataset_id}")
    body = await request.json()
    reason = body.get("reason", "rejected by teacher")
    s.storage.set_dataset(dataset_id, status="rejected", rejected_reason=reason,
                          at=s.clock())
    s.audit(user["user_id"], "dataset.reject", dataset_id,
            {"status": ds["status"]}, {"reason": reason})
    return {"dataset_id": dataset_id, "status": "rejected", "reason": reason}


@router.get("/artifacts")
async def list_artifacts(request: Request, user: dict = Depends(require_admin)) -> dict:
    return {"artifacts": svc(request).storage.list_artifacts()}


@router.delete("/artifacts/{job_id}/{kind}")
async def delete_artifact(job_id: str, kind: str, request: Request,
                          user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    arts = [a for a in s.storage.artifacts_for(job_id) if a["kind"] == kind]
    if not arts:
        raise RobovlaError("not_found", f"no {kind!r} artifact for {job_id}")
    p = Path(s.data_dir) / arts[0]["path"]
    if p.exists():
        p.unlink()
    _drop_artifact(s, job_id, kind)
    s.audit(user["user_id"], "artifact.purge", job_id, {"kind": kind}, None)
    return {"ok": True}


def _drop_artifact(s, job_id: str, kind: str) -> None:
    s.storage._db.execute("DELETE FROM artifacts WHERE job_id = ? AND kind = ?",
                          (job_id, kind))


# -- system (§8.5) -----------------------------------------------------------------------


@router.get("/audit")
async def audit_log(request: Request, since: str | None = None,
                    user: dict = Depends(require_admin)) -> dict:
    return {"entries": svc(request).storage.audit_entries(since=since)}


@router.get("/config")
async def config_view(request: Request, user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    raw = _fleet_raw(s.fleet)
    return {"yaml": yaml.safe_dump(raw, sort_keys=False), "config": raw}


@router.post("/config/reload")
async def config_reload(request: Request, user: dict = Depends(require_admin)) -> dict:
    """Hot reload (§10/§8.5): re-read fleet.yaml -> validate -> diff -> apply;
    active jobs unaffected, queued jobs revalidated."""
    s = svc(request)
    if s.fleet_path is None or not Path(s.fleet_path).exists():
        raise RobovlaError("not_found", "fleet.yaml not configured on this console")
    new = parse_fleet(Path(s.fleet_path).read_text())
    changes = diff_fleet(s.fleet, new)
    s.reload_config(new, user["user_id"], changes)
    return {"ok": True, "changes": [c.to_dict() for c in changes]}


@router.post("/maintenance")
async def maintenance(request: Request, user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    body = await request.json()
    on = bool(body.get("on"))
    s.set_maintenance(on, user["user_id"])
    return {"maintenance": on}


@router.post("/backups/trigger")
async def backup_trigger(request: Request, user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    dest = s.fleet.console.retention.backup.dest
    out = _make_backup_bundle(s, dest)
    s.audit(user["user_id"], "backup.trigger", dest,
            None, {"size_bytes": out["size_bytes"], "sha256": out["sha256"][:16]})
    return out


@router.get("/backups")
async def backups(request: Request, user: dict = Depends(require_admin)) -> dict:
    return {"backups": svc(request).storage.list_backups()}


def _make_backup_bundle(s, dest: str) -> dict:
    """Backup bundle (§8.4): the shared console seam (robovla.backup).

    DRY with the CLI: one bundle builder, WAL-consistent DB snapshot
    (online backup API), ``backups`` table row via the record hook.
    """
    return backup.make_backup_bundle(
        s.data_dir, dest, clock=s.clock,
        record=lambda i: s.storage.add_backup(
            s.clock(), i["path"], i["size_bytes"], i["sha256"], i["status"]))
