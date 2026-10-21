"""core.audit — the append-only audit reducer (design §8).

All admin actions + auth events are written here: (actor, timestamp, action,
target, before->after JSON). Pure: timestamps are *injected* RFC 3339 UTC
strings; the seam owns the wall clock.
"""

import json

import pytest

from robovla.core.audit import AuditEntry, AuditLog
from robovla.core.errors import RobovlaError


def _log() -> AuditLog:
    log = AuditLog()
    log.record("2026-10-21T09:00:00Z", "teacher", "node.enable", "node2",
               before={"enabled": False}, after={"enabled": True})
    log.record("2026-10-21T09:05:00Z", "teacher", "user.deactivate", "s03",
               before={"active": True}, after={"active": False})
    log.record("2026-10-21T10:00:00Z", "s07", "auth.login", "s07")
    return log


class TestRecord:
    def test_entry_fields(self):
        log = AuditLog()
        e = log.record("2026-10-21T09:00:00Z", "teacher", "job.kill", "j_1",
                       before={"state": "queued"}, after={"state": "killed"})
        assert isinstance(e, AuditEntry)
        assert (e.actor, e.action, e.target) == ("teacher", "job.kill", "j_1")
        assert e.before == {"state": "queued"} and e.after == {"state": "killed"}
        assert log.entries[-1] is e

    def test_before_after_optional(self):
        e = AuditLog().record("2026-10-21T09:00:00Z", "s07", "auth.login", "s07")
        assert e.before is None and e.after is None

    @pytest.mark.parametrize("bad_at", ["not-a-time", "2026-13-01T00:00:00Z", "", "2026-10-21 09:00:00"])
    def test_bad_timestamp_rejected(self, bad_at):
        with pytest.raises(RobovlaError) as e:
            AuditLog().record(bad_at, "teacher", "x", "y")
        assert e.value.code == "validation"

    @pytest.mark.parametrize("field", ["actor", "action", "target"])
    def test_empty_field_rejected(self, field):
        with pytest.raises(RobovlaError):
            AuditLog().record("2026-10-21T09:00:00Z", "", "a", "t") if field == "actor" \
                else AuditLog().record("2026-10-21T09:00:00Z", "a", "", "t") if field == "action" \
                else AuditLog().record("2026-10-21T09:00:00Z", "a", "b", "")

    @pytest.mark.parametrize("value", ["not-a-dict", ["x"], 42, None])
    def test_non_dict_snapshot_rejected(self, value):
        # None is the explicit "no snapshot" case; anything non-dict is a bug.
        if value is None:
            e = AuditLog().record("2026-10-21T09:00:00Z", "a", "b", "c", before=None)
            assert e.before is None
        else:
            with pytest.raises(RobovlaError):
                AuditLog().record("2026-10-21T09:00:00Z", "a", "b", "c", before=value)


class TestAppendOnly:
    def test_grows(self):
        log = _log()
        assert len(log) == 3
        log.record("2026-10-21T11:00:00Z", "teacher", "job.bump", "j_2")
        assert len(log) == 4

    def test_entries_are_immutable(self):
        log = _log()
        with pytest.raises(Exception):
            log.entries[0].actor = "someone-else"

    def test_list_view_is_detached(self):
        log = _log()
        entries = log.to_list()
        log.record("2026-10-21T11:00:00Z", "teacher", "job.bump", "j_2")
        assert len(entries) == 3


class TestFilter:
    def test_since_inclusive(self):
        log = _log()
        got = log.filter(since="2026-10-21T09:05:00Z")
        assert [e.action for e in got] == ["user.deactivate", "auth.login"]

    def test_actor_action_target(self):
        log = _log()
        assert [e.action for e in log.filter(actor="s07")] == ["auth.login"]
        assert [e.target for e in log.filter(action="node.enable")] == ["node2"]
        assert [e.actor for e in log.filter(target="s03")] == ["teacher"]

    def test_filters_combine(self):
        log = _log()
        assert log.filter(actor="teacher", action="user.deactivate") == [log.entries[1]]

    def test_since_rejects_bad_timestamp(self):
        with pytest.raises(RobovlaError):
            _log().filter(since="yesterday")

    def test_empty_log_filters(self):
        assert AuditLog().filter(since="2026-10-21T09:00:00Z") == []


class TestExport:
    def test_to_list_is_json(self):
        log = _log()
        payload = log.to_list()
        assert isinstance(payload, list) and len(payload) == 3
        text = json.dumps(payload)
        assert "node.enable" in text
        assert "before" in payload[0] and "after" in payload[0]
        assert "before" not in payload[2] or payload[2]["before"] is None
