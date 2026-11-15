"""Admin: student/teacher accounts (design §8.1 / §9.7 users).

Create/import (inactive until activated), activate/deactivate, per-student
limit overrides, out-of-band password reset (one-time code for the
classroom — no email), session list/revoke, Pi-token generate/rotate/revoke.
Every action audited; teacher + 60/min via require_admin.
"""

from __future__ import annotations

import csv
import io
import json
import secrets

from fastapi import APIRouter, Depends, Request

from ..core.errors import RobovlaError
from .auth import hash_password, new_token_value, require_admin, svc, token_lookup_key

router = APIRouter(prefix="/api/admin/users", tags=["admin"])


@router.get("")
async def list_users(request: Request, user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    # password_min travels with the list so the page never carries a copy of
    # the rule that would silently rot next to PASSWORD_MIN
    return {"users": [
        {k: u[k] for k in ("user_id", "role", "active", "display_name", "email",
                           "created_at", "max_running", "max_queued",
                           "max_dataset_bytes", "max_run_min", "run_now")}
        for u in s.storage.list_users()], "password_min": PASSWORD_MIN}


@router.post("")
async def create_user(request: Request, user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    body = await request.json()
    uid = body.get("user_id", "")
    if not uid or s.storage.get_user(uid):
        raise RobovlaError("conflict", f"unknown user_id {uid!r} or it exists")
    password = body.get("password") or secrets.token_urlsafe(12)
    u = s.storage.create_user(
        uid, body.get("role", "student"), hash_password(password),
        display_name=body.get("display_name", ""), email=body.get("email", ""),
        active=bool(body.get("active", False)), at=s.clock())
    s.audit(user["user_id"], "user.create", uid, None,
            {"role": u["role"], "active": u["active"]})
    return {**{k: u[k] for k in ("user_id", "role", "active", "display_name")},
            "password": password, "shown_once": True}


@router.post("/import")
async def import_users(request: Request, user: dict = Depends(require_admin)) -> dict:
    """CSV roster import (id,name,email,role,active) — created inactive."""
    s = svc(request)
    body = await request.json()
    created: list[str] = []
    reader = csv.DictReader(io.StringIO(body.get("csv", "")))
    for row in reader:
        uid = (row.get("id") or "").strip()
        if not uid or s.storage.get_user(uid):
            continue
        s.storage.create_user(
            uid, (row.get("role") or "student").strip() or "student",
            hash_password(secrets.token_urlsafe(12)),
            display_name=(row.get("name") or "").strip(),
            email=(row.get("email") or "").strip(),
            active=str(row.get("active", "false")).lower() in ("1", "true", "yes"),
            at=s.clock())
        s.audit(user["user_id"], "user.import", uid)
        created.append(uid)
    return {"created": created}


@router.patch("/{uid}")
async def patch_user(uid: str, request: Request,
                     user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    before = s.storage.get_user(uid)
    if before is None:
        raise RobovlaError("not_found", f"unknown user: {uid}")
    body = await request.json()
    allowed = {k: v for k, v in body.items()
               if k in ("active", "role", "display_name", "email",
                        "max_running", "max_queued", "max_dataset_bytes",
                        "max_run_min", "run_now")}
    if not allowed:
        raise RobovlaError("validation", "no updatable fields in body")
    after = s.storage.set_user(uid, at=s.clock(), **allowed)
    s.audit(user["user_id"], "user.patch", uid,
            {k: before.get(k) for k in allowed}, {k: after.get(k) for k in allowed})
    return {k: after.get(k) for k in ("user_id", "role", "active", "display_name")}


#: Admin-set passwords are typed on a keyboard by an operator, not generated;
#: 10 characters is the floor that keeps "123456" out of a classroom roster.
PASSWORD_MIN = 10


@router.post("/{uid}/reset-password")
async def reset_password(uid: str, request: Request,
                         user: dict = Depends(require_admin)) -> dict:
    """Out-of-band reset (§8.1), for any account including another teacher.

    No body, or no ``password`` field: a generated one-time password — the
    classroom paper flow, returned once.

    ``{"password": "..."}``: set that exact value. Bringing a seat back means
    typing a credential you already chose, not copying a random string from one
    device to another. A caller-supplied password is NEVER echoed back (it would
    land in browser history and logs); only generated values are returned.

    The old password stops working at this call. Existing sessions are left
    alone — revoking them is the separate, visible action on the same panel, so
    resetting your own password does not log you out mid-task.
    """
    s = svc(request)
    if s.storage.get_user(uid) is None:
        raise RobovlaError("not_found", f"unknown user: {uid}")
    raw = await request.body()
    body = json.loads(raw) if raw else {}
    if not isinstance(body, dict):
        raise RobovlaError("validation", "body must be an object")
    supplied = body.get("password")
    if supplied is None:
        password, generated = secrets.token_urlsafe(10), True
    else:
        password, generated = str(supplied), False
        if len(password) < PASSWORD_MIN:
            raise RobovlaError("validation",
                               f"password must be at least {PASSWORD_MIN} characters")
    s.storage.set_password_hash(uid, hash_password(password), at=s.clock())
    # the audit row records WHO reset WHO and which flavour, never the value
    s.audit(user["user_id"], "user.reset_password", uid, None,
            {"generated": generated})
    out = {"user_id": uid, "generated": generated}
    if generated:
        out["password"] = password
        out["shown_once"] = True
    return out


@router.get("/{uid}/sessions")
async def user_sessions(uid: str, request: Request,
                        user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    return {"sessions": s.storage.sessions_for_user(uid, s.clock())}


@router.delete("/{uid}/sessions/{sid}")
async def revoke_session(uid: str, sid: str, request: Request,
                         user: dict = Depends(require_admin)) -> dict:
    s = svc(request)
    s.storage.delete_session(sid)
    s.audit(user["user_id"], "user.revoke_session", uid, {"session_id": sid})
    return {"ok": True}


@router.post("/{uid}/tokens/{action}")
async def user_token(uid: str, action: str, request: Request,
                     user: dict = Depends(require_admin)) -> dict:
    """Admin Pi-token ops: generate | rotate | revoke (§8.1)."""
    s = svc(request)
    if s.storage.get_user(uid) is None:
        raise RobovlaError("not_found", f"unknown user: {uid}")
    active = [t for t in s.storage.list_tokens(uid) if t["revoked_at"] is None]
    if action == "generate" and active:
        raise RobovlaError("conflict", "active token exists; rotate or revoke")
    if action in ("rotate", "revoke") and not active:
        raise RobovlaError("conflict", "no active token")
    if action == "revoke":
        s.storage.revoke_token(active[-1]["token_id"], at=s.clock())
        s.audit(user["user_id"], "user.token_revoke", uid,
                {"token_id": active[-1]["token_id"]}, None)
        return {"ok": True}
    if action == "rotate":
        s.storage.revoke_token(active[-1]["token_id"], at=s.clock())
    value = new_token_value()
    tid = s.storage.create_token(uid, hash_password(value),
                                 token_lookup_key(value), at=s.clock())
    s.audit(user["user_id"], f"user.token_{action}", uid, None, {"token_id": tid})
    return {"token_id": tid, "token": value, "shown_once": True}
