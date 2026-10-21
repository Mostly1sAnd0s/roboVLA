"""Auth: cookie sessions (12 h sliding), argon2id passwords, per-student Pi
tokens, and the ONE RBAC dependency (design §9.1/§9.8: "Enforced by one
FastAPI dependency — never per-route").

Roles: teacher (everything) · student (own data, class board, slot claims) ·
observer (read-only board + GPU state). Cross-tenant ownership checks happen
in the service (``own_dataset`` -> ``forbidden``); this module only gates
roles. Pi tokens: value shown once; at rest an argon2 hash (authoritative) +
a sha256 lookup key (O(1) candidate find — the hash is still verified before
the token is trusted, §9.8).
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from argon2 import PasswordHasher
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..core.errors import RobovlaError
from ..service import ConsoleService

COOKIE = "robovla_session"
_hasher = PasswordHasher()

ROLE_STUDENT = "student"
ROLE_TEACHER = "teacher"
ROLE_OBSERVER = "observer"


def svc(request: Request) -> ConsoleService:
    return request.app.state.svc


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(hashed: str, plain: str) -> bool:
    try:
        return _hasher.verify(hashed, plain)
    except Exception:
        return False


def token_lookup_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def new_token_value() -> str:
    """Pi token value (shown once; argon2 hash + sha256 key stored, §9.8)."""
    return "pi_" + secrets.token_urlsafe(32)


def current_user(request: Request) -> dict[str, Any]:
    """Session-cookie identity; 401 ``auth`` when absent/expired/inactive."""
    s = svc(request)
    sid = request.cookies.get(COOKIE)
    if not sid:
        raise RobovlaError("auth", "not logged in")
    session = s.storage.get_session(sid, s.clock())
    if session is None:
        raise RobovlaError("auth", "session expired")
    user = s.storage.get_user(session["user_id"])
    if user is None or not user["active"]:
        raise RobovlaError("auth", "account is not active")
    return user


def optional_user(request: Request) -> dict[str, Any] | None:
    sid = request.cookies.get(COOKIE)
    if not sid:
        return None
    s = svc(request)
    session = s.storage.get_session(sid, s.clock())
    if session is None:
        return None
    user = s.storage.get_user(session["user_id"])
    return user if user and user["active"] else None


def require(*roles: str) -> Any:
    """The ONE RBAC dependency: role gate (403 ``forbidden``)."""

    def dep(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
        if roles and user["role"] not in roles:
            raise RobovlaError("forbidden", "role may not perform this action")
        return user

    return dep


def require_admin(request: Request,
                  user: dict[str, Any] = Depends(require(ROLE_TEACHER))) -> dict[str, Any]:
    """Teacher + the admin rate limit (60/min, §9.7) — one dependency."""
    s = svc(request)
    s.check_rate("admin", user["user_id"], s.fleet.console.limits.admin_rate_per_min, 60)
    return user


def user_from_bearer(s: ConsoleService, token: str) -> dict[str, Any]:
    """Bearer Pi token -> active user; ``auth`` on any miss (§9.5)."""
    row = s.storage.token_by_lookup(token_lookup_key(token))
    if row is None or not verify_password(row["token_hash"], token):
        raise RobovlaError("auth", "invalid robot token")
    user = s.storage.get_user(row["user_id"])
    if user is None or not user["active"]:
        raise RobovlaError("auth", "account is not active")
    return user


def agent_token_ok(s: ConsoleService, node_id: str, token: str) -> bool:
    """Agent-token verification (the bundle pull, §9.9): argon2 check of the
    per-node pre-shared token against the DB hash."""
    stored = s.storage.agent_token_by_lookup(node_id, token_lookup_key(token))
    if stored is None:
        return False
    return verify_password(stored, token)


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login")
async def login(request: Request) -> dict[str, Any]:
    s = svc(request)
    s.check_rate("login", request.client.host if request.client else "?", 5, 60)
    body = await request.json()
    username, password = str(body.get("username", "")), str(body.get("password", ""))
    user = s.storage.get_user(username)
    if user is None or not verify_password(user["password_hash"], password):
        s.audit(username or "?", "auth.login_failure", username)
        raise RobovlaError("auth", "invalid username or password")
    if not user["active"]:
        s.audit(username, "auth.login_rejected", username)
        raise RobovlaError("auth", "account is not active")
    sid = s.storage.create_session(user["user_id"], s.clock(),
                                   s.fleet.console.session_ttl_h)
    s.audit(user["user_id"], "auth.login", user["user_id"])
    resp = JSONResponse({"user_id": user["user_id"], "role": user["role"],
                         "display_name": user["display_name"]})
    resp.set_cookie(COOKIE, sid, httponly=True, samesite="lax", path="/")
    return resp


@router.post("/logout")
async def logout(request: Request,
                 user: dict[str, Any] = Depends(current_user)) -> JSONResponse:
    s = svc(request)
    s.storage.delete_session(request.cookies.get(COOKIE, ""))
    s.audit(user["user_id"], "auth.logout", user["user_id"])
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


@router.get("/me")
async def me(request: Request,
             user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    s = svc(request)
    limits = s.student_limits(user)
    tokens = [
        {"token_id": t["token_id"], "created_at": t["created_at"],
         "revoked_at": t["revoked_at"]}
        for t in s.storage.list_tokens(user["user_id"])
    ]
    return {
        "user_id": user["user_id"], "role": user["role"],
        "display_name": user["display_name"], "email": user["email"],
        "active": user["active"], "limits": limits, "tokens": tokens,
    }


@router.post("/token")
async def token_action(request: Request,
                       user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    """Own Pi token: {action: generate|rotate|revoke} (value shown once)."""
    s = svc(request)
    body = await request.json()
    action = body.get("action")
    active = [t for t in s.storage.list_tokens(user["user_id"])
              if t["revoked_at"] is None]
    if action == "generate":
        if active:
            raise RobovlaError("conflict", "a token is already active; use rotate or revoke")
        value = new_token_value()
        tid = s.storage.create_token(user["user_id"], hash_password(value),
                                     token_lookup_key(value), at=s.clock())
        s.audit(user["user_id"], "token.generate", user["user_id"])
        return {"token_id": tid, "token": value, "shown_once": True}
    if action == "rotate":
        if not active:
            raise RobovlaError("conflict", "no active token to rotate")
        s.storage.revoke_token(active[-1]["token_id"], at=s.clock())
        value = new_token_value()
        tid = s.storage.create_token(user["user_id"], hash_password(value),
                                     token_lookup_key(value), at=s.clock())
        s.audit(user["user_id"], "token.rotate", user["user_id"],
                {"token_id": active[-1]["token_id"]}, {"token_id": tid})
        return {"token_id": tid, "token": value, "shown_once": True}
    if action == "revoke":
        if not active:
            raise RobovlaError("conflict", "no active token to revoke")
        s.storage.revoke_token(active[-1]["token_id"], at=s.clock())
        s.audit(user["user_id"], "token.revoke", user["user_id"],
                {"token_id": active[-1]["token_id"]}, None)
        return {"ok": True}
    raise RobovlaError("validation", "action must be generate|rotate|revoke")

