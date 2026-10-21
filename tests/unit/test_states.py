"""core.states — the job state machine (design §10): a pure transition table.

queued -> running -> finishing -> done|failed|killed  (+ kill/reschedule).
"""

import pytest

from robovla.core.errors import RobovlaError
from robovla.core.states import (
    EVENTS,
    STATES,
    TERMINAL_STATES,
    apply_transition,
    is_terminal,
)


class TestStates:
    def test_states(self):
        assert set(STATES) == {"queued", "running", "finishing", "done", "failed", "killed"}

    def test_terminal_states(self):
        assert TERMINAL_STATES == frozenset({"done", "failed", "killed"})

    def test_is_terminal(self):
        for s in ("done", "failed", "killed"):
            assert is_terminal(s)
        for s in ("queued", "running", "finishing"):
            assert not is_terminal(s)

    def test_unknown_state_rejected(self):
        with pytest.raises(RobovlaError) as e:
            is_terminal("paused")
        assert e.value.code == "validation"


class TestTransitionTable:
    def test_happy_path(self):
        s = "queued"
        for event, expected in (
            ("start", "running"),
            ("finish", "finishing"),
            ("done", "done"),
        ):
            s = apply_transition(s, event)
            assert s == expected

    def test_failure_paths(self):
        assert apply_transition("running", "fail") == "failed"
        assert apply_transition("finishing", "fail") == "failed"

    def test_kill_from_queued_and_running(self):
        assert apply_transition("queued", "kill") == "killed"
        assert apply_transition("running", "kill") == "killed"

    def test_reschedule_stays_queued(self):
        assert apply_transition("queued", "reschedule") == "queued"

    def test_terminal_is_absorbing(self):
        for s in ("done", "failed", "killed"):
            for event in EVENTS:
                with pytest.raises(RobovlaError) as e:
                    apply_transition(s, event)
                assert e.value.code == "conflict"

    def test_invalid_transitions_rejected(self):
        # no preemption: nothing moves a running job backwards
        for s, event in (
            ("running", "start"),
            ("queued", "finish"),
            ("queued", "done"),
            ("finishing", "kill"),
            ("finishing", "start"),
            ("running", "reschedule"),
        ):
            with pytest.raises(RobovlaError) as e:
                apply_transition(s, event)
            assert e.value.code == "conflict"

    def test_unknown_event_rejected(self):
        with pytest.raises(RobovlaError) as e:
            apply_transition("queued", "teleport")
        assert e.value.code == "validation"

    def test_table_is_exhaustive_over_events(self):
        # every (state, event) pair is either legal or explicitly illegal —
        # no silent KeyError.
        for s in STATES:
            for event in EVENTS:
                try:
                    nxt = apply_transition(s, event)
                except RobovlaError:
                    continue
                assert nxt in STATES
