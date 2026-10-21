"""Append-only audit log reducer (design §8: all admin actions + auth events).

Every entry is ``(actor, timestamp, action, target, before->after JSON)`` —
append-only, admin-readable, JSON-exportable (§8.1). Pure: timestamps are
*injected* RFC 3339 UTC strings (the seam owns the wall clock); this module
never reads time, so unit tests run deterministically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .errors import RobovlaError

#: Strict RFC 3339 (``T`` separator, explicit offset) — what the API promises.
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$"
)


def parse_rfc3339(value: str) -> datetime:
    """Parse a strict RFC 3339 timestamp -> aware datetime (UTC-normalized)."""
    if not isinstance(value, str) or not _RFC3339_RE.fullmatch(value):
        raise RobovlaError("validation", f"timestamp {value!r} must be RFC 3339 (e.g. 2026-10-21T09:00:00Z)")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise RobovlaError("validation", f"timestamp {value!r} is not a valid date/time") from e


@dataclass(frozen=True)
class AuditEntry:
    """One immutable audit record."""

    at: str
    actor: str
    action: str
    target: str
    before: dict | None
    after: dict | None

    def to_dict(self) -> dict:
        return {
            "at": self.at,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "before": self.before,
            "after": self.after,
        }


def _check_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RobovlaError("validation", f"{name}: must be a non-empty string")
    return value


def _check_snapshot(value: object, name: str) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RobovlaError("validation", f"{name}: must be a JSON object or null")
    return value


class AuditLog:
    """Append-only log of audit entries.

    Immutability: entries are frozen dataclasses; ``entries`` is a tuple that
    is re-bound on every record (nothing mutates in place).
    """

    def __init__(self) -> None:
        self._entries: tuple[AuditEntry, ...] = ()

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def record(
        self,
        at: str,
        actor: str,
        action: str,
        target: str,
        before: dict | None = None,
        after: dict | None = None,
    ) -> AuditEntry:
        """Append one entry (validates everything; the seam stamps ``at``)."""
        parse_rfc3339(at)
        entry = AuditEntry(
            at=at,
            actor=_check_str(actor, "actor"),
            action=_check_str(action, "action"),
            target=_check_str(target, "target"),
            before=_check_snapshot(before, "before"),
            after=_check_snapshot(after, "after"),
        )
        self._entries = self._entries + (entry,)
        return entry

    def filter(
        self,
        since: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        target: str | None = None,
    ) -> list[AuditEntry]:
        """Entries matching all given criteria; ``since`` is inclusive."""
        since_dt = parse_rfc3339(since) if since is not None else None
        out = []
        for e in self._entries:
            if since_dt is not None and parse_rfc3339(e.at) < since_dt:
                continue
            if actor is not None and e.actor != actor:
                continue
            if action is not None and e.action != action:
                continue
            if target is not None and e.target != target:
                continue
            out.append(e)
        return out

    def to_list(self) -> list[dict]:
        """JSON-ready snapshot (a copy — later records don't affect it)."""
        return [e.to_dict() for e in self._entries]
