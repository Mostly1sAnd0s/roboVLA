"""Experiment-ticket schema + link lifecycle (design §9.3, v0.5.1).

Tickets are a standalone resource created *before* run submission:
hypothesis -> config summary -> prediction; the outcome is recorded at run
end. Lifecycle (states are explicit; ``linked_job`` alone cannot tell
"never linked" from "released"):

    unlinked --link--> linked --run terminal (done|failed|killed)--> released
    released --link--> linked            (re-linkable; retained for the record)

* editable only while ``unlinked`` — after that it is the class record;
* the link **releases** when the linked run reaches a terminal state
  (v0.5.1); the ticket stays, re-linkable or replaced by a new ticket;
* only one run can be linked at a time (``conflict`` otherwise — the
  "409 if already linked" row of the test matrix).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .errors import RobovlaError
from .states import is_terminal

MAX_FIELD_LEN = 2000

_FIELDS = ("hypothesis", "config_summary", "prediction")
_STATES = ("unlinked", "linked", "released")


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    owner: str
    hypothesis: str
    config_summary: str
    prediction: str
    linked_job: str | None
    outcome: str | None
    state: str  # "unlinked" | "linked" | "released"
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "owner": self.owner,
            "hypothesis": self.hypothesis,
            "config_summary": self.config_summary,
            "prediction": self.prediction,
            "linked_job": self.linked_job,
            "outcome": self.outcome,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _fail(message: str) -> None:
    raise RobovlaError("validation", message)


def _conflict(message: str) -> None:
    raise RobovlaError("conflict", message)


def _check_field(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{name}: must be a non-empty string")
    if len(value) > MAX_FIELD_LEN:
        _fail(f"{name}: at most {MAX_FIELD_LEN} characters")
    return value


def _check_id(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{what}: must be a non-empty string")
    return value


def create_ticket(ticket_id: str, owner: str, payload: dict, at: str) -> Ticket:
    """Create a ticket from ``{hypothesis, config_summary, prediction}``."""
    _check_id(ticket_id, "ticket_id")
    _check_id(owner, "owner")
    if not isinstance(payload, dict):
        _fail("payload: must be an object")
    extra = [k for k in payload if k not in _FIELDS]
    if extra:
        _fail(f"payload: unknown key(s) {', '.join(sorted(extra))}")
    values = {}
    for field in _FIELDS:
        if field not in payload:
            _fail(f"payload: missing key {field!r}")
        values[field] = _check_field(field, payload[field])
    return Ticket(
        ticket_id=ticket_id,
        owner=owner,
        linked_job=None,
        outcome=None,
        state="unlinked",
        created_at=at,
        updated_at=at,
        **values,
    )


def update_ticket(t: Ticket, payload: dict, at: str) -> Ticket:
    """Partial update of the three fields — only while ``unlinked``."""
    if t.state != "unlinked":
        _conflict(f"ticket {t.ticket_id} is {t.state}; it is the record and cannot be edited")
    if not isinstance(payload, dict) or not payload:
        _fail("payload: must be a non-empty object")
    extra = [k for k in payload if k not in _FIELDS]
    if extra:
        _fail(f"payload: unknown key(s) {', '.join(sorted(extra))}")
    updates = {k: _check_field(k, v) for k, v in payload.items()}
    return replace(t, **updates, updated_at=at)


def link_ticket(t: Ticket, job_id: str, at: str) -> Ticket:
    """Link the ticket to a run — the run becomes its experiment run (§9.3)."""
    if t.state == "linked":
        _conflict(f"ticket {t.ticket_id} is already linked to {t.linked_job}")
    _check_id(job_id, "job_id")
    return replace(t, linked_job=job_id, state="linked", updated_at=at)


def on_run_terminal(t: Ticket, job_id: str, state: str, at: str) -> Ticket:
    """Release the link when the linked run reaches a terminal state.

    Non-terminal states, or a different job id, leave the ticket untouched.
    ``state`` must be a known job state (terminal check is one definition,
    DRY — :func:`robovla.core.states.is_terminal`).
    """
    if not is_terminal(state):
        return t
    if t.state != "linked" or t.linked_job != job_id:
        return t
    return replace(t, linked_job=None, state="released", updated_at=at)


def set_outcome(t: Ticket, outcome: str, at: str) -> Ticket:
    """Record the outcome (completed at run end, §9.3).

    Allowed in any state — a student may draft it while linked and revise it
    after release; the ticket itself stays the record.
    """
    if not isinstance(outcome, str) or not outcome.strip():
        _fail("outcome: must be a non-empty string")
    if len(outcome) > MAX_FIELD_LEN:
        _fail(f"outcome: at most {MAX_FIELD_LEN} characters")
    return replace(t, outcome=outcome, updated_at=at)
