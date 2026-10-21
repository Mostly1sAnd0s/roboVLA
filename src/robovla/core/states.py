"""The job state machine — a pure transition table (design §10).

    queued -> running -> finishing -> done | failed | killed

Plus: ``kill`` from ``queued``/``running`` and ``reschedule`` (re-pin) while
``queued``. Terminal states are absorbing. There is **no preemption**: nothing
ever moves a ``running`` job backwards (design §3).

This module is the single source of truth for job states and transitions;
``scheduler`` and ``ticket`` both consume it (DRY: one table, no drift).
No I/O, no clock — the caller stamps audit entries with injected time.
"""

from __future__ import annotations

from .errors import RobovlaError

#: Every job state (design §10 / §9.4 ``state`` event values).
STATES = frozenset({"queued", "running", "finishing", "done", "failed", "killed"})

#: Terminal states — a job leaving these is a bug, not a transition.
TERMINAL_STATES = frozenset({"done", "failed", "killed"})

#: The transition table: state -> {event -> next state}. Anything not listed
#: here is illegal and raises ``conflict``. Exhaustive by construction: the
#: unit tests assert every (state, event) pair is either legal or rejected.
TRANSITIONS: dict[str, dict[str, str]] = {
    "queued": {"start": "running", "kill": "killed", "reschedule": "queued"},
    "running": {"finish": "finishing", "kill": "killed", "fail": "failed"},
    "finishing": {"done": "done", "fail": "failed"},
    "done": {},
    "failed": {},
    "killed": {},
}

#: All events the table knows (union of keys).
EVENTS = frozenset({event for table in TRANSITIONS.values() for event in table})


def is_terminal(state: str) -> bool:
    """True if ``state`` is ``done``/``failed``/``killed``."""
    if state not in STATES:
        raise RobovlaError("validation", f"unknown job state: {state!r}")
    return state in TERMINAL_STATES


def apply_transition(state: str, event: str) -> str:
    """Return the state after ``event``, or raise.

    * unknown state -> ``validation``
    * unknown event -> ``validation``
    * illegal transition -> ``conflict`` (message names the pair)
    """
    if state not in STATES:
        raise RobovlaError("validation", f"unknown job state: {state!r}")
    if event not in EVENTS:
        raise RobovlaError("validation", f"unknown job event: {event!r}")
    nxt = TRANSITIONS[state].get(event)
    if nxt is None:
        raise RobovlaError("conflict", f"illegal transition {state!r} + {event!r}")
    return nxt
