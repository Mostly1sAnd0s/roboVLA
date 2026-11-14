"""Browser-side robot operation (§9.10 extension): config, teleop, stop, record.

The robot never listens, so nothing here talks to the Pi. A browser call writes
state; the robot picks it up on its next ``POST /api/robot/telemetry`` reply.
That is the price of the invariant, and it buys one logged, rate-limited door.

Authorization: the owner of the robot, or any teacher. Stop is the exception in
one direction only — a teacher may stop every robot at once.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from .. import jobs_ops
from ..core.robot_control import TELEOP_LIMITS, control_for
from .auth import ROLE_STUDENT, ROLE_TEACHER, require, svc

router = APIRouter(prefix="/api/robot", tags=["robot-ui"])


def _target(user: dict[str, Any], body: dict[str, Any]) -> str:
    """Whose robot: the caller's, or the named one if the caller is a teacher."""
    who = str(body.get("student") or user["user_id"])
    if user["role"] != ROLE_TEACHER and who != user["user_id"]:
        return user["user_id"]          # never escalate; ops re-check anyway
    return who


@router.get("/config")
async def get_config(request: Request, student: str = "",
                     user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER))) -> dict:
    s = svc(request)
    who = student or user["user_id"]
    if user["role"] != ROLE_TEACHER and who != user["user_id"]:
        who = user["user_id"]
    return {"student": who, "config": jobs_ops.robot_config(s, who),
            "limits": {k: list(v) for k, v in TELEOP_LIMITS.items()}}


@router.post("/config")
async def post_config(request: Request,
                      user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER))) -> dict:
    s = svc(request)
    body = await request.json()
    who = _target(user, body)
    cfg = jobs_ops.set_robot_config(s, user, who, body.get("config") or {})
    return {"student": who, "config": cfg}


@router.get("/status")
async def status(request: Request, student: str = "",
                 user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER))) -> dict:
    """Everything one drive session needs, from one call: the last frame and
    state, what the ladder currently says, and the robot's config."""
    s = svc(request)
    who = student or user["user_id"]
    if user["role"] != ROLE_TEACHER and who != user["user_id"]:
        who = user["user_id"]
    view = s.robot_view.view(who)
    ctl = jobs_ops.robot_ctl(s, who)
    return {"student": who, "view": view, "age_s": s.robot_view.age_s(who),
            "control": control_for(ctl, s.clock()), "config": jobs_ops.robot_config(s, who),
            "record_acked": ctl.get("record_acked") or 0}


@router.post("/teleop")
async def teleop(request: Request,
                 user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER))) -> dict:
    """One human command, alive for ``hold_s``. The browser refreshes it while
    a key or stick is held and sends zeros the moment it lets go or loses focus."""
    s = svc(request)
    body = await request.json()
    s.check_rate(f"robot_teleop:{user['user_id']}", user["user_id"], 20, 1.0)
    return jobs_ops.queue_teleop(s, user, _target(user, body),
                                 body.get("vx"), body.get("vy"), body.get("wz"),
                                 body.get("hold_s"))


@router.post("/stop")
async def stop(request: Request,
               user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER))) -> dict:
    """Latch the stop (spacebar, big red button) or clear it. Teacher may
    broadcast to every robot; nobody but the owner or a teacher clears one."""
    s = svc(request)
    body = await request.json()
    return jobs_ops.queue_stop(s, user, _target(user, body),
                               clear=bool(body.get("clear")),
                               broadcast=bool(body.get("broadcast")),
                               reason=str(body.get("reason") or ""))


@router.post("/record")
async def record(request: Request,
                 user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER))) -> dict:
    s = svc(request)
    body = await request.json()
    return jobs_ops.queue_record(s, user, _target(user, body),
                                 str(body.get("action") or ""),
                                 str(body.get("task") or ""),
                                 str(body.get("note") or ""))
