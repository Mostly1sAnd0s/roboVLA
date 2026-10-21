"""core.ticket — experiment-ticket schema + link lifecycle (design §9.3,
v0.5.1).

Tickets are standalone resources created *before* run submission (hypothesis
-> config -> prediction). The link releases when the linked run reaches a
terminal state (done|failed|killed) — the ticket is retained for the record,
re-linkable, or replaced by a new ticket.
"""

import pytest

from robovla.core.errors import RobovlaError
from robovla.core.ticket import (
    MAX_FIELD_LEN,
    Ticket,
    create_ticket,
    link_ticket,
    on_run_terminal,
    set_outcome,
    update_ticket,
)

T0 = "2026-10-21T12:00:00Z"
T1 = "2026-10-21T13:00:00Z"
PAYLOAD = {
    "hypothesis": "rank 32 with lr 5e-4 beats rank 16 on stop-at-target",
    "config_summary": "qlora 8-bit, rank 32, alpha 32, lr 5e-4, 3 epochs, batch 4",
    "prediction": "held-out accuracy >= 0.6",
}


def _ticket(**kw):
    kw.setdefault("ticket_id", "t_0001")
    kw.setdefault("owner", "s07")
    kw.setdefault("payload", PAYLOAD)
    kw.setdefault("at", T0)
    return create_ticket(**kw)


class TestCreate:
    def test_create(self):
        t = _ticket()
        assert t.ticket_id == "t_0001"
        assert t.owner == "s07"
        assert t.hypothesis == PAYLOAD["hypothesis"]
        assert t.config_summary == PAYLOAD["config_summary"]
        assert t.prediction == PAYLOAD["prediction"]
        assert t.linked_job is None
        assert t.outcome is None
        assert t.created_at == T0 and t.updated_at == T0

    @pytest.mark.parametrize("field", ["hypothesis", "config_summary", "prediction"])
    def test_missing_field_rejected(self, field):
        p = {k: v for k, v in PAYLOAD.items() if k != field}
        with pytest.raises(RobovlaError) as e:
            create_ticket(ticket_id="t_0002", owner="s07", payload=p, at=T0)
        assert e.value.code == "validation"
        assert field in e.value.message

    @pytest.mark.parametrize("field", ["hypothesis", "config_summary", "prediction"])
    def test_empty_field_rejected(self, field):
        p = {**PAYLOAD, field: ""}
        with pytest.raises(RobovlaError):
            create_ticket(ticket_id="t_0002", owner="s07", payload=p, at=T0)

    def test_too_long_rejected(self):
        assert MAX_FIELD_LEN == 2000
        with pytest.raises(RobovlaError):
            create_ticket(
                ticket_id="t_0002", owner="s07",
                payload={**PAYLOAD, "hypothesis": "x" * 2001}, at=T0,
            )

    def test_unknown_payload_key_rejected(self):
        with pytest.raises(RobovlaError):
            create_ticket(
                ticket_id="t_0002", owner="s07",
                payload={**PAYLOAD, "outcome": "early"}, at=T0,
            )

    def test_owner_must_be_non_empty(self):
        with pytest.raises(RobovlaError):
            create_ticket(ticket_id="t_0002", owner="", payload=PAYLOAD, at=T0)


class TestUpdate:
    def test_partial_update_unlinked(self):
        t = _ticket()
        t2 = update_ticket(t, {"hypothesis": "revised hypothesis"}, at=T1)
        assert t2.hypothesis == "revised hypothesis"
        assert t2.prediction == PAYLOAD["prediction"]  # untouched
        assert t2.updated_at == T1 and t2.created_at == T0
        # original unchanged (frozen)
        assert t.hypothesis == PAYLOAD["hypothesis"]

    def test_update_requires_at_least_one_field(self):
        with pytest.raises(RobovlaError):
            update_ticket(_ticket(), {}, at=T1)

    def test_update_rejected_when_linked(self):
        t = link_ticket(_ticket(), "j_20261021_0001", at=T0)
        with pytest.raises(RobovlaError) as e:
            update_ticket(t, {"hypothesis": "change"}, at=T1)
        assert e.value.code == "conflict"

    def test_update_rejected_after_release(self):
        # retained for the record: released tickets are not editable
        t = on_run_terminal(link_ticket(_ticket(), "j_20261021_0001", at=T0),
                            "j_20261021_0001", "done", at=T1)
        assert t.linked_job is None
        with pytest.raises(RobovlaError) as e:
            update_ticket(t, {"hypothesis": "change"}, at=T1)
        assert e.value.code == "conflict"

    def test_update_unknown_field_rejected(self):
        with pytest.raises(RobovlaError):
            update_ticket(_ticket(), {"rank": 64}, at=T1)


class TestLinkRelease:
    def test_link(self):
        t = link_ticket(_ticket(), "j_20261021_0001", at=T0)
        assert t.linked_job == "j_20261021_0001"
        assert t.updated_at == T0

    def test_link_twice_conflicts(self):
        t = link_ticket(_ticket(), "j_20261021_0001", at=T0)
        with pytest.raises(RobovlaError) as e:
            link_ticket(t, "j_20261021_0002", at=T1)
        assert e.value.code == "conflict"

    @pytest.mark.parametrize("state", ["done", "failed", "killed"])
    def test_terminal_state_releases(self, state):
        t = link_ticket(_ticket(), "j_20261021_0001", at=T0)
        released = on_run_terminal(t, "j_20261021_0001", state, at=T1)
        assert released.linked_job is None

    def test_non_terminal_state_keeps_link(self):
        t = link_ticket(_ticket(), "j_20261021_0001", at=T0)
        for state in ("running", "finishing"):
            still = on_run_terminal(t, "j_20261021_0001", state, at=T1)
            assert still.linked_job == "j_20261021_0001"

    def test_other_run_terminal_keeps_link(self):
        t = link_ticket(_ticket(), "j_20261021_0001", at=T0)
        still = on_run_terminal(t, "j_20261021_9999", "done", at=T1)
        assert still.linked_job == "j_20261021_0001"

    def test_unlinked_ticket_ignores_terminal_event(self):
        t = _ticket()
        assert on_run_terminal(t, "j_20261021_0001", "done", at=T1) == t

    def test_relink_after_release(self):
        t = link_ticket(_ticket(), "j_20261021_0001", at=T0)
        released = on_run_terminal(t, "j_20261021_0001", "failed", at=T1)
        relinked = link_ticket(released, "j_20261021_0002", at=T1)
        assert relinked.linked_job == "j_20261021_0002"

    def test_bad_job_id_rejected(self):
        with pytest.raises(RobovlaError):
            link_ticket(_ticket(), "", at=T0)


class TestOutcome:
    def test_set_outcome(self):
        t = link_ticket(_ticket(), "j_20261021_0001", at=T0)
        done = set_outcome(t, "held-out accuracy 0.62 — prediction confirmed", at=T1)
        assert done.outcome == "held-out accuracy 0.62 — prediction confirmed"

    def test_outcome_overwritable(self):
        t = set_outcome(_ticket(), "first draft", at=T0)
        t2 = set_outcome(t, "revised", at=T1)
        assert t2.outcome == "revised"

    def test_empty_outcome_rejected(self):
        with pytest.raises(RobovlaError):
            set_outcome(_ticket(), "  ", at=T0)
