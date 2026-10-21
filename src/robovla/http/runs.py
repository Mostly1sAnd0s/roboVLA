"""Runs (design §9.3): submit (ticket-gated for train, idempotent, capped),
queue view, kill/bump/requeue/complete, logs tail, artifacts, linked ticket.

Ownership: students see/act on their own runs; teacher sees all; cross-tenant
access is ``forbidden`` (test matrix row). Queue/queue-ETA feed the class
board for every role.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from .. import jobs_ops
from ..core.errors import RobovlaError
from ..core.states import TERMINAL_STATES
from .auth import ROLE_STUDENT, ROLE_TEACHER, ROLE_OBSERVER, current_user, require, svc

router = APIRouter(prefix="/api/runs", tags=["runs"])
_ANY = (ROLE_STUDENT, ROLE_TEACHER, ROLE_OBSERVER)


def _run_view(s, row: dict) -> dict:
    view = {
        "job_id": row["job_id"], "student": row["spec"]["student"],
        "job_type": row["spec"]["type"], "dataset": row["spec"]["dataset"],
        "state": row["state"], "node_id": row["node_id"],
        "submitted_at": row["submitted_at"], "created_at": row["created_at"],
        "terminal_at": row.get("terminal_at"), "note": row.get("note"),
        "ticket_id": row.get("ticket_id"), "run_id": row.get("run_id"),
        "run_now": row["run_now"],
        "config": {
            "lora": row["spec"]["lora"], "optim": row["spec"]["optim"],
            "quant": row["spec"]["quant"], "seed": row["spec"]["seed"],
            "image_aug": row["spec"]["image_aug"],
            "prompt_template": row["spec"]["prompt_template"],
            "budget_gb": row["spec"]["limits"]["budget_gb"],
        },
    }
    if row["state"] not in TERMINAL_STATES:
        states = {n.id: s.node_state(n.id) for n in s.fleet.enabled_nodes()}
        view["position"] = s.scheduler.position(row["job_id"], states)
        eta = s.scheduler.eta_s(row["job_id"], states)
        view["eta_s"] = None if eta == float("inf") else round(eta)
        view["progress"] = s.progress(row["job_id"])
    if row.get("predict_probe") is not None:
        view["predict_probe"] = row["predict_probe"]
    return view


def _own_or_teacher(s, user: dict, job_id: str) -> dict:
    row = s.storage.get_job(job_id)
    if row is None:
        raise RobovlaError("not_found", f"unknown job: {job_id}")
    if user["role"] not in (ROLE_TEACHER, ROLE_OBSERVER) \
            and row["spec"]["student"] != user["user_id"]:
        raise RobovlaError("forbidden", "not your run")
    return row


@router.post("")
async def submit(request: Request, user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER))) -> dict:
    s = svc(request)
    s.check_rate("submit", user["user_id"], 10, 60)
    body = await request.json()
    return jobs_ops.submit_run(s, user, body,
                               idempotency_key=request.headers.get("Idempotency-Key"))


@router.get("")
async def list_runs(request: Request, user: dict = Depends(current_user)) -> dict:
    s = svc(request)
    student = None if user["role"] in (ROLE_TEACHER, ROLE_OBSERVER) else user["user_id"]
    rows = s.storage.list_jobs(student=student)
    return {"runs": [_run_view(s, r) for r in rows]}


@router.get("/queue")
async def queue(request: Request, user: dict = Depends(current_user)) -> dict:
    """All nodes, all students: position + ETA (feeds the class board)."""
    s = svc(request)
    states = {n.id: s.node_state(n.id) for n in s.fleet.enabled_nodes()}
    rows = s.storage.list_jobs(state_in=("queued", "running", "finishing"))
    out = []
    for r in rows:
        v = _run_view(s, r)
        blockers = s.scheduler.admission_blockers(r["job_id"], states) \
            if r["state"] == "queued" else []
        if blockers:
            v["waiting"] = "slot"
            v["blockers"] = blockers
        out.append(v)
    return {"queue": out}


@router.get("/{job_id}")
async def get_run(job_id: str, request: Request, user: dict = Depends(current_user)) -> dict:
    s = svc(request)
    row = _own_or_teacher(s, user, job_id)
    return _run_view(s, row)


@router.get("/{job_id}/ticket")
async def run_ticket(job_id: str, request: Request, user: dict = Depends(current_user)) -> dict:
    s = svc(request)
    row = _own_or_teacher(s, user, job_id)
    if not row.get("ticket_id"):
        raise RobovlaError("not_found", f"run {job_id} has no linked ticket")
    t = s.storage.get_ticket(row["ticket_id"])
    return {"job_id": job_id, "ticket": t, "outcome": t.get("outcome")}


@router.get("/{job_id}/logs")
async def run_logs(job_id: str, request: Request,
                   offset: int = 0, user: dict = Depends(current_user)) -> dict:
    s = svc(request)
    row = _own_or_teacher(s, user, job_id)
    runner = s.runners.get(row["node_id"] or "")
    if runner is None or not row["node_id"]:
        raise RobovlaError("node_offline", "node is offline (or not yet dispatched)")
    try:
        log = await asyncio.to_thread(runner.tail, job_id, offset)
    except RobovlaError as e:
        raise e
    return {"job_id": job_id, "offset": offset, "log": log}


@router.get("/{job_id}/artifacts")
async def run_artifacts(job_id: str, request: Request,
                        user: dict = Depends(current_user)) -> dict:
    s = svc(request)
    row = _own_or_teacher(s, user, job_id)
    arts = s.storage.artifacts_for(job_id)
    from pathlib import Path
    available = [a for a in arts if (s.data_dir / a["path"]).exists()]
    return {"job_id": job_id, "artifacts": arts,
            "downloadable": [a["kind"] for a in available],
            "run_config": row["spec"]}


@router.get("/{job_id}/artifacts/{kind}")
async def download_artifact(job_id: str, kind: str, request: Request,
                            user: dict = Depends(current_user)) -> FileResponse:
    s = svc(request)
    _own_or_teacher(s, user, job_id)
    from pathlib import Path
    arts = [a for a in s.storage.artifacts_for(job_id) if a["kind"] == kind]
    if not arts:
        raise RobovlaError("not_found", f"no {kind!r} artifact for {job_id}")
    p = s.data_dir / arts[0]["path"]
    if not p.exists():
        raise RobovlaError("not_found", "artifact file is not on this host (node-local)")
    return FileResponse(p)


@router.post("/{job_id}/kill")
async def kill(job_id: str, request: Request, user: dict = Depends(current_user)) -> dict:
    s = svc(request)
    row = _own_or_teacher(s, user, job_id)
    if user["role"] != ROLE_TEACHER and row["state"] not in ("queued",):
        raise RobovlaError("forbidden", "students may kill only their queued jobs "
                                        "(teacher kills running jobs, §8.3)")
    return await jobs_ops.kill_job(s, job_id, user["user_id"])


@router.post("/{job_id}/bump")
async def bump(job_id: str, request: Request,
               user: dict = Depends(require(ROLE_TEACHER))) -> dict:
    return jobs_ops.bump(svc(request), job_id, user["user_id"])


@router.post("/{job_id}/requeue")
async def requeue(job_id: str, request: Request,
                  user: dict = Depends(require(ROLE_TEACHER))) -> dict:
    s = svc(request)
    _own_or_teacher(s, user, job_id)
    body = await request.json()
    return jobs_ops.requeue(s, job_id, user["user_id"], body.get("node_id"))


@router.post("/{job_id}/complete")
async def complete(job_id: str, request: Request,
                   user: dict = Depends(require(ROLE_TEACHER))) -> dict:
    s = svc(request)
    _own_or_teacher(s, user, job_id)
    body = await request.json()
    note = body.get("note", "")
    if not note:
        raise RobovlaError("validation", "note is required (manual completion, §8.3)")
    return jobs_ops.complete(s, job_id, user["user_id"], note)

