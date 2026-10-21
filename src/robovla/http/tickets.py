"""Experiment tickets (design §9.3): standalone resource created *before*
run submission — hypothesis -> config -> prediction; outcome at completion.
Editable only while unlinked (core enforces); the link releases when the
linked run reaches a terminal state (service, v0.5.1).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..core.errors import RobovlaError
from ..core.ticket import create_ticket, set_outcome, update_ticket
from ..service import ticket_obj
from .auth import ROLE_STUDENT, ROLE_TEACHER, require, svc

router = APIRouter(prefix="/api/tickets", tags=["tickets"])
_ROLES = (ROLE_STUDENT, ROLE_TEACHER)


def _view(t: dict) -> dict:
    return {k: t.get(k) for k in
            ("ticket_id", "owner", "hypothesis", "config_summary", "prediction",
             "linked_job", "outcome", "state", "created_at", "updated_at")}


@router.post("")
async def create(request: Request, user: dict = Depends(require(*_ROLES))) -> dict:
    s = svc(request)
    body = await request.json()
    n = int(s.storage.get_kv(f"ticket_seq_{user['user_id']}") or 0) + 1
    s.storage.set_kv(f"ticket_seq_{user['user_id']}", str(n))
    tid = f"t_{user['user_id']}_{n:04d}"
    t = create_ticket(tid, user["user_id"], body, at=s.clock())
    s.storage.save_ticket(t.to_dict())
    s.audit(user["user_id"], "ticket.create", tid)
    return _view(s.storage.get_ticket(tid))


@router.get("")
async def list_tickets(request: Request, user: dict = Depends(require(*_ROLES))) -> dict:
    s = svc(request)
    owner = None if user["role"] == ROLE_TEACHER else user["user_id"]
    return {"tickets": [_view(t) for t in s.storage.list_tickets(owner=owner)]}


@router.get("/{ticket_id}")
async def get_ticket(ticket_id: str, request: Request,
                     user: dict = Depends(require(*_ROLES))) -> dict:
    s = svc(request)
    t = s.storage.get_ticket(ticket_id)
    if t is None:
        raise RobovlaError("not_found", f"unknown ticket: {ticket_id}")
    if user["role"] != ROLE_TEACHER and t["owner"] != user["user_id"]:
        raise RobovlaError("forbidden", "not your ticket")
    return _view(t)


@router.put("/{ticket_id}")
async def update(request: Request, ticket_id: str,
                 user: dict = Depends(require(*_ROLES))) -> dict:
    s = svc(request)
    t = s.storage.get_ticket(ticket_id)
    if t is None:
        raise RobovlaError("not_found", f"unknown ticket: {ticket_id}")
    if user["role"] != ROLE_TEACHER and t["owner"] != user["user_id"]:
        raise RobovlaError("forbidden", "not your ticket")
    body = await request.json()
    updated = update_ticket(ticket_obj(t), body, at=s.clock())  # unlinked-only
    s.storage.save_ticket(updated.to_dict())
    s.audit(user["user_id"], "ticket.update", ticket_id)
    return _view(s.storage.get_ticket(ticket_id))


@router.put("/{ticket_id}/outcome")
async def set_ticket_outcome(ticket_id: str, request: Request,
                             user: dict = Depends(require(*_ROLES))) -> dict:
    s = svc(request)
    t = s.storage.get_ticket(ticket_id)
    if t is None:
        raise RobovlaError("not_found", f"unknown ticket: {ticket_id}")
    if user["role"] != ROLE_TEACHER and t["owner"] != user["user_id"]:
        raise RobovlaError("forbidden", "not your ticket")
    body = await request.json()
    updated = set_outcome(ticket_obj(t), body.get("outcome", ""), at=s.clock())
    s.storage.save_ticket(updated.to_dict())
    s.audit(user["user_id"], "ticket.outcome", ticket_id)
    return _view(s.storage.get_ticket(ticket_id))
