"""Inference slots + the robot path (design §9.5) and the agent bundle pull
(§9.9 — the ONE non-cookie console route).

Slot claims create ``infer`` jobs (auto first-fit, no preemption): a free
slot + VRAM -> 200 with the stable console endpoint; otherwise 202 queued
with position/ETA. Robots authenticate with the per-student Bearer token and
get un-normalized actions; the node agent port is console-only.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import Response

from .. import datasets_ops, jobs_ops
from ..core.errors import RobovlaError
from ..service import LEASE_S
from .auth import (ROLE_STUDENT, ROLE_TEACHER, agent_token_ok, current_user,
                   require, svc, user_from_bearer)

router = APIRouter(prefix="/api/infer", tags=["infer"])
robot_router = APIRouter(prefix="/api/robot", tags=["robot"])

ROBOT_ENDPOINT = "/api/robot/predict"
_RATE_PER_S = 5  # ~5 req/s per student (§9.5)


@router.post("/slots/claim")
async def claim(request: Request, user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER))) -> dict:
    s = svc(request)
    body = await request.json()
    run_id = body.get("run_id", "")
    created = jobs_ops.claim_slot(s, user, run_id)
    job_id = created["job_id"]
    # the tick loop serves the slot; one immediate tick keeps claims snappy
    await s.tick()
    row = s.storage.get_job(job_id)
    states = {n.id: s.node_state(n.id) for n in s.fleet.enabled_nodes()}
    if row["state"] == "running":
        return {
            "slot": row.get("slot_handle"), "node_id": row["node_id"],
            "job_id": job_id, "run_id": run_id,
            # the stable endpoint for the life of the course (§9.5) — robots
            # talk only to the console; <host> is this console's LAN address.
            "endpoint": f"http://<console-host>:{s.fleet.console.port}{ROBOT_ENDPOINT}",
            "lease_s": LEASE_S,
        }
    eta = s.scheduler.eta_s(job_id, states)
    return {
        "queued": True, "job_id": job_id, "run_id": run_id,
        "position": s.scheduler.position(job_id, states),
        "eta_s": None if eta == float("inf") else round(eta),
    }


@router.get("/slots")
async def slots(request: Request, user: dict = Depends(current_user)) -> dict:
    return {"slots": jobs_ops.slot_board(svc(request))}


@router.post("/slots/{job_id}/release")
async def release(job_id: str, request: Request, user: dict = Depends(current_user)) -> dict:
    s = svc(request)
    row = s.storage.get_job(job_id)
    if row is None or row["spec"]["type"] != "infer":
        raise RobovlaError("not_found", f"unknown inference slot: {job_id}")
    if user["role"] != ROLE_TEACHER and row["spec"]["student"] != user["user_id"]:
        raise RobovlaError("forbidden", "not your slot")
    return await jobs_ops.kill_job(s, job_id, user["user_id"])


# -- robot endpoint (Bearer student token; the ONLY robot-facing route) ---------


@robot_router.post("/predict")
async def robot_predict(request: Request,
                        authorization: str = Header(default="")) -> dict:
    s = svc(request)
    if not authorization.lower().startswith("bearer "):
        raise RobovlaError("auth", "missing Bearer token")
    token = authorization[7:].strip()
    user = user_from_bearer(s, token)
    s.check_rate("robot_predict", user["user_id"], _RATE_PER_S, 1.0)
    body = await request.json()
    return jobs_ops.robot_predict(s, user, body.get("image", ""), body.get("prompt", ""))


# -- agent bundle pull (§9.9: same per-node agent token; the only non-cookie) ---


agent_router = APIRouter(prefix="/api/agent", tags=["agent"])


@agent_router.get("/datasets/{dataset_id:path}/bundle")
async def dataset_bundle(dataset_id: str, request: Request,
                         authorization: str = Header(default="")) -> Response:
    """The agent pulls the prepared dataset tar at dispatch. The agent
    presents the per-node pre-shared token; the console verifies it against
    the argon2 hash in its DB (which node? by the dataset's owner node set —
    here: any node whose token matches; agent tokens are per-node)."""
    s = svc(request)
    if not authorization.lower().startswith("bearer "):
        raise RobovlaError("auth", "missing Bearer agent token")
    token = authorization[7:].strip()
    node_id = None
    for n in s.fleet.nodes:
        if agent_token_ok(s, n.id, token):
            node_id = n.id
            break
    if node_id is None:
        raise RobovlaError("auth", "invalid agent token")
    ds = s.storage.get_dataset(dataset_id)
    if ds is None or ds["status"] != "ready":
        raise RobovlaError("dataset_not_ready",
                           f"dataset {dataset_id!r} is not ready to pull")
    blob, sha = datasets_ops.bundle(s, dataset_id)
    s.audit(node_id, "dataset.pull", dataset_id, None, {"sha256": sha[:16]})
    return Response(content=blob, media_type="application/gzip",
                    headers={"X-Dataset-Sha256": sha, "Content-Length": str(len(blob))})
