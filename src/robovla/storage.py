"""SQLite storage seam (design §11: "I/O only at seams"; DRY: the only SQL in
the codebase lives here + the DDL in ``storage_schema.py``).

WAL mode (readers never block the scheduler, §10). Rows are plain dicts;
domain rules live in ``robovla.core``; timestamps are caller-supplied RFC 3339
(service owns the clock). Tables (design §8/§9): users, sessions, tokens (Pi),
agent_tokens, datasets, chunks, tickets, jobs, artifacts, audit (append-only),
idempotency, backups, kv.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta

from .storage_schema import SCHEMA as _SCHEMA

#: Idempotency dedup window (design §9: 24 h).
IDEMPOTENCY_TTL_H = 24

_T = "2026-10-21T00:00:00Z"  # default stand-in; real rows always carry caller-supplied time


def _now_rfc3339() -> str:
    from datetime import datetime, timezone  # seam: only place the clock may leak

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class Storage:
    """All console state in one SQLite file (WAL). Sync API; the service wraps
    calls in ``asyncio.to_thread`` where latency matters (LAN, small DB)."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False)
        # single console process; access is serialized by the GIL, and WAL
        # keeps readers from blocking the scheduler (design §10)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)
    def close(self) -> None:
        self._db.close()
    # -- users ---------------------------------------------------------------
    def create_user(
        self,
        user_id: str,
        role: str,
        password_hash: str,
        *,
        display_name: str = "",
        email: str = "",
        active: bool = False,
        at: str = _T,
    ) -> dict[str, Any]:
        self._db.execute(
            "INSERT INTO users (user_id, role, active, password_hash, display_name, email, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, role, int(active), password_hash, display_name, email, at),
        )
        return self.get_user(user_id)
    def get_user(self, user_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return _user_dict(row) if row else None
    def list_users(self) -> list[dict[str, Any]]:
        return [_user_dict(r) for r in self._db.execute("SELECT * FROM users ORDER BY user_id")]
    def set_user(self, user_id: str, at: str = _T, **fields: Any) -> dict[str, Any]:
        allowed = {"role", "active", "display_name", "email",
                   "max_running", "max_queued", "max_dataset_bytes", "max_run_min", "run_now"}
        bad = [k for k in fields if k not in allowed]
        if bad:
            raise ValueError(f"unknown user field(s): {', '.join(sorted(bad))}")
        for k in ("active", "run_now"):
            if k in fields:
                fields[k] = int(bool(fields[k]))
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._db.execute(
            f"UPDATE users SET {sets} WHERE user_id = ?",
            (*fields.values(), user_id),
        )
        user = self.get_user(user_id)
        if user is None:
            raise KeyError(user_id)
        return user
    def set_password_hash(self, user_id: str, password_hash: str, at: str = _T) -> None:
        self._db.execute(
            "UPDATE users SET password_hash = ? WHERE user_id = ?",
            (password_hash, user_id),
        )
    # -- sessions (12 h sliding, §9.1) -----------------------------------------
    def create_session(self, user_id: str, created_at: str, ttl_h: float) -> str:
        session_id = f"ss_{uuid.uuid4().hex[:16]}"
        expires = datetime.fromisoformat(created_at.replace("Z", "+00:00")) + timedelta(hours=ttl_h)
        self._db.execute(
            "INSERT INTO sessions (session_id, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (session_id, user_id, created_at, expires.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
        return session_id
    def get_session(self, session_id: str, now: str) -> dict[str, Any] | None:
        """Valid (unexpired) session or None; sliding refresh of ``expires_at``."""
        row = self._db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] <= now:
            self.delete_session(session_id)
            return None
        new_exp = datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(
            hours=self._SESSION_TTL_H
        )
        new_exp_str = new_exp.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._db.execute(
            "UPDATE sessions SET expires_at = ? WHERE session_id = ?",
            (new_exp_str, session_id),
        )
        # report the post-slide expiry (the API promise: 12 h sliding)
        return {"session_id": row["session_id"], "user_id": row["user_id"],
                "created_at": row["created_at"], "expires_at": new_exp_str}
    # Session TTL is a global console setting; injected here so storage needs
    # no config dependency (the service sets it once at startup, §10).
    _SESSION_TTL_H: float = 12.0
    def set_session_ttl_h(self, hours: float) -> None:
        self._SESSION_TTL_H = float(hours)
    def delete_session(self, session_id: str) -> None:
        self._db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    def sessions_for_user(self, user_id: str, now: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND expires_at > ? ORDER BY created_at",
            (user_id, now),
        ).fetchall()
        return [dict(r) for r in rows]
    # -- Pi tokens (shown once, hashed at rest, §9.8) ---------------------------
    def create_token(self, user_id: str, token_hash: str, token_lookup: str,
                     at: str = _T) -> str:
        token_id = f"tk_{uuid.uuid4().hex[:12]}"
        self._db.execute(
            "INSERT INTO tokens (token_id, user_id, token_hash, token_lookup, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (token_id, user_id, token_hash, token_lookup, at),
        )
        return token_id
    def token_by_lookup(self, token_lookup: str) -> dict[str, Any] | None:
        """O(1) find by the sha256 lookup key (argon2 hash stays authoritative)."""
        row = self._db.execute(
            "SELECT * FROM tokens WHERE token_lookup = ? AND revoked_at IS NULL",
            (token_lookup,),
        ).fetchone()
        return dict(row) if row else None
    def list_tokens(self, user_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM tokens WHERE user_id = ? ORDER BY created_at", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    def revoke_token(self, token_id: str, at: str = _T) -> bool:
        cur = self._db.execute(
            "UPDATE tokens SET revoked_at = ? WHERE token_id = ? AND revoked_at IS NULL",
            (at, token_id),
        )
        return cur.rowcount > 0
    # -- per-node agent tokens (§9.8: hash in DB for audit/rotation) -------------
    def set_agent_token(self, node_id: str, token_hash: str, token_lookup: str,
                        at: str = _T) -> None:
        self._db.execute(
            "INSERT INTO agent_tokens (node_id, token_hash, token_lookup, created_at, rotated_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT(node_id) DO UPDATE SET"
            " token_hash = excluded.token_hash, token_lookup = excluded.token_lookup,"
            " rotated_at = excluded.rotated_at",
            (node_id, token_hash, token_lookup, at, at),
        )
    def get_agent_token_hash(self, node_id: str) -> str | None:
        row = self._db.execute(
            "SELECT token_hash FROM agent_tokens WHERE node_id = ?", (node_id,)
        ).fetchone()
        return row["token_hash"] if row else None

    def agent_token_by_lookup(self, node_id: str, token_lookup: str) -> str | None:
        row = self._db.execute(
            "SELECT token_hash FROM agent_tokens WHERE node_id = ? AND token_lookup = ?",
            (node_id, token_lookup),
        ).fetchone()
        return row["token_hash"] if row else None
    # -- datasets -----------------------------------------------------------------
    def create_dataset(
        self,
        dataset_id: str,
        owner: str,
        status: str,
        at: str = _T,
        total_bytes: int = 0,
    ) -> dict[str, Any]:
        self._db.execute(
            "INSERT INTO datasets (dataset_id, owner, status, total_bytes, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (dataset_id, owner, status, total_bytes, at, at),
        )
        return self.get_dataset(dataset_id)
    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["qa"] = _loads(d.pop("qa_json"))
        d["norm_stats"] = _loads(d.pop("norm_stats_json"))
        d["split"] = _loads(d.pop("split_json"))
        return d
    def list_datasets(self, owner: str | None = None) -> list[dict[str, Any]]:
        if owner is None:
            rows = self._db.execute("SELECT * FROM datasets ORDER BY dataset_id").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM datasets WHERE owner = ? ORDER BY dataset_id", (owner,)
            ).fetchall()
        return [self.get_dataset(r["dataset_id"]) for r in rows]
    _DATASET_FIELDS = {"status", "total_bytes", "qa_json", "norm_stats_json",
                       "split_json", "rejected_reason", "updated_at"}
    def set_dataset(self, dataset_id: str, at: str | None = None, **fields: Any) -> dict[str, Any]:
        # ``at`` is sugar for updated_at; ``qa``/``norm_stats``/``split`` are
        # accepted as parsed objects (serialized here, single place).
        fields["updated_at"] = at or _now_rfc3339()
        for k in ("qa", "norm_stats", "split"):
            if k in fields:
                fields[k + "_json"] = _dumps(fields.pop(k))
        bad = [k for k in fields if k not in self._DATASET_FIELDS]
        if bad:
            raise ValueError(f"unknown dataset field(s): {', '.join(sorted(bad))}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        cur = self._db.execute(
            f"UPDATE datasets SET {sets} WHERE dataset_id = ?", (*fields.values(), dataset_id)
        )
        if cur.rowcount == 0:
            raise KeyError(dataset_id)
        return self.get_dataset(dataset_id)
    def add_chunk(self, dataset_id: str, chunk_index: int, sha256: str, nbytes: int) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO chunks (dataset_id, chunk_index, sha256, nbytes)"
            " VALUES (?, ?, ?, ?)",
            (dataset_id, chunk_index, sha256, nbytes),
        )
        self._db.execute(
            "UPDATE datasets SET total_bytes = (SELECT COALESCE(SUM(nbytes), 0) FROM chunks"
            " WHERE dataset_id = ?) WHERE dataset_id = ?",
            (dataset_id, dataset_id),
        )
    def chunks(self, dataset_id: str) -> dict[int, dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM chunks WHERE dataset_id = ? ORDER BY chunk_index", (dataset_id,)
        ).fetchall()
        return {r["chunk_index"]: dict(r) for r in rows}
    # -- tickets (core owns the rules; storage stores rows, §11) ------------------
    _TICKET_FIELDS = {"owner", "hypothesis", "config_summary", "prediction",
                      "linked_job", "outcome", "state", "updated_at"}
    def save_ticket(self, ticket: dict[str, Any]) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO tickets (ticket_id, owner, hypothesis, config_summary,"
            " prediction, linked_job, outcome, state, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticket["ticket_id"], ticket["owner"], ticket["hypothesis"],
                ticket["config_summary"], ticket["prediction"], ticket["linked_job"],
                ticket["outcome"], ticket["state"],
                ticket.get("created_at", _T), ticket["updated_at"],
            ),
        )
    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        return dict(row) if row else None
    def list_tickets(self, owner: str | None = None) -> list[dict[str, Any]]:
        if owner is None:
            rows = self._db.execute("SELECT * FROM tickets ORDER BY created_at").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM tickets WHERE owner = ? ORDER BY created_at", (owner,)
            ).fetchall()
        return [dict(r) for r in rows]
    def ticket_by_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM tickets WHERE linked_job = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None
    # -- jobs ---------------------------------------------------------------------
    def insert_job(
        self,
        job_id: str,
        spec: dict[str, Any],
        state: str,
        submitted_at: float,
        *,
        node_id: str | None = None,
        run_now: bool = False,
        ticket_id: str | None = None,
        idempotency_key: str | None = None,
        created_at: str = _T,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._db.execute(
            "INSERT INTO jobs (job_id, spec_json, state, node_id, submitted_at, run_now,"
            " ticket_id, idempotency_key, created_at, run_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, json.dumps(spec, sort_keys=True), state, node_id, submitted_at,
             int(run_now), ticket_id, idempotency_key, created_at, run_id),
        )
        if idempotency_key is not None:
            self.register_idempotency(idempotency_key, job_id, created_at)
        return self.get_job(job_id)
    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["spec"] = json.loads(d.pop("spec_json"))
        d["run_now"] = bool(d["run_now"])
        d["predict_probe"] = _loads(d.pop("predict_probe_json", None))
        return d
    def list_jobs(self, student: str | None = None, state_in: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT job_id FROM jobs WHERE 1=1", []
        if student is not None:
            q += " AND json_extract(spec_json, '$.student') = ?"
            args.append(student)
        if state_in is not None:
            q += " AND state IN (%s)" % ",".join("?" * len(state_in))
            args.extend(state_in)
        q += " ORDER BY submitted_at, job_id"
        rows = self._db.execute(q, args).fetchall()
        return [self.get_job(r["job_id"]) for r in rows]
    _JOB_FIELDS = {"state", "node_id", "ticket_id", "terminal_at", "note",
                  "slot_handle", "last_used_at", "predict_probe", "run_id"}
    def set_job(self, job_id: str, predict_probe: list | None = None, **fields: Any) -> dict[str, Any]:
        # ``predict_probe`` is accepted as a 7-dim list (smoke job, §6) and
        # stored as JSON; ``slot_handle``/``last_used_at`` are plain columns.
        if predict_probe is not None:
            fields["predict_probe_json"] = _dumps(predict_probe)
        bad = [k for k in fields if k not in self._JOB_FIELDS]
        if bad:
            raise ValueError(f"unknown job field(s): {', '.join(sorted(bad))}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        cur = self._db.execute(f"UPDATE jobs SET {sets} WHERE job_id = ?", (*fields.values(), job_id))
        if cur.rowcount == 0:
            raise KeyError(job_id)
        return self.get_job(job_id)
    def set_job_spec(self, job_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Replace the stored JobSpec (dispatch finalizes node/agent/budget)."""
        cur = self._db.execute(
            "UPDATE jobs SET spec_json = ? WHERE job_id = ?",
            (json.dumps(spec, sort_keys=True), job_id),
        )
        if cur.rowcount == 0:
            raise KeyError(job_id)
        return self.get_job(job_id)
    def jobs_created_on(self, day: str) -> int:
        return self._db.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE substr(created_at, 1, 10) = ?", (day,)
        ).fetchone()["n"]
    # -- idempotency (24 h dedup, §9) --------------------------------------------
    def register_idempotency(self, key: str, job_id: str, created_at: str) -> None:
        expires = datetime.fromisoformat(created_at.replace("Z", "+00:00")) + timedelta(
            hours=IDEMPOTENCY_TTL_H
        )
        self._db.execute(
            "INSERT OR IGNORE INTO idempotency (key, job_id, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (key, job_id, created_at, expires.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    def job_by_idempotency(self, key: str, now: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT job_id FROM idempotency WHERE key = ? AND expires_at > ?", (key, now)
        ).fetchone()
        return self.get_job(row["job_id"]) if row else None
    # -- artifacts -------------------------------------------------------------------
    def add_artifact(self, job_id: str, kind: str, path: str, nbytes: int, at: str = _T) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO artifacts (job_id, kind, path, nbytes, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (job_id, kind, path, nbytes, at),
        )
    def artifacts_for(self, job_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM artifacts WHERE job_id = ? ORDER BY kind", (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    def list_artifacts(self, job_id: str | None = None) -> list[dict[str, Any]]:
        if job_id is None:
            rows = self._db.execute("SELECT * FROM artifacts ORDER BY job_id, kind").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM artifacts WHERE job_id = ? ORDER BY kind", (job_id,)
            ).fetchall()
        return [dict(r) for r in rows]
    # -- audit (append-only, §8) ------------------------------------------------------
    def append_audit(
        self,
        at: str,
        actor: str,
        action: str,
        target: str,
        before: dict | None = None,
        after: dict | None = None,
    ) -> int:
        cur = self._db.execute(
            "INSERT INTO audit (at, actor, action, target, before_json, after_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (at, actor, action, target, _dumps(before), _dumps(after)),
        )
        return int(cur.lastrowid)
    def audit_entries(
        self,
        since: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        target: str | None = None,
    ) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM audit WHERE 1=1", []
        if since is not None:
            q += " AND at >= ?"
            args.append(since)
        if actor is not None:
            q += " AND actor = ?"
            args.append(actor)
        if action is not None:
            q += " AND action = ?"
            args.append(action)
        if target is not None:
            q += " AND target = ?"
            args.append(target)
        q += " ORDER BY seq"
        return [dict(r) for r in self._db.execute(q, args).fetchall()]
    # -- backups / kv -------------------------------------------------------------------
    def add_backup(self, at: str, dest: str, size_bytes: int, sha256: str, status: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO backups (at, dest, size_bytes, sha256, status) VALUES (?, ?, ?, ?, ?)",
            (at, dest, size_bytes, sha256, status),
        )
    def list_backups(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT * FROM backups ORDER BY at DESC")]
    def get_kv(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None
    def set_kv(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
def _user_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["active"] = bool(d["active"])
    for k in ("run_now",):
        d[k] = bool(d[k]) if d[k] is not None else None
    return d
