"""storage — SQLite seam (design §11: I/O only at seams; single SQL module).

Covers: users/limits, sessions (expiry + sliding TTL), Pi tokens (hashed at
rest, revoke), agent tokens, datasets + chunked-upload accounting, tickets,
jobs + 24 h idempotency window, artifacts, audit append/filter, backups, kv.
"""

from __future__ import annotations

import pytest

from robovla.storage import Storage

T0 = "2026-10-21T09:00:00Z"
T1 = "2026-10-21T09:30:00Z"


@pytest.fixture
def store(tmp_path) -> Storage:
    s = Storage(tmp_path / "console" / "state.db")
    yield s
    s.close()


# -- users --------------------------------------------------------------------


def test_create_and_get_user(store):
    u = store.create_user("s01", "student", "hash-x", display_name="Sam", at=T0)
    assert u["user_id"] == "s01"
    assert u["role"] == "student"
    assert u["active"] is False
    assert u["max_running"] is None  # None = config default


def test_set_user_limits_override_and_fields(store):
    store.create_user("s01", "student", "h", at=T0)
    u = store.set_user("s01", at=T1, max_running=2, max_queued=1, active=True, role="teacher")
    assert u["max_running"] == 2 and u["max_queued"] == 1
    assert u["active"] is True and u["role"] == "teacher"
    with pytest.raises(ValueError):
        store.set_user("s01", password_hash="bad")  # not a field


def test_users_inactive_until_activated_and_list(store):
    store.create_user("s01", "student", "h", at=T0)
    store.create_user("s02", "teacher", "h", active=True, at=T0)
    users = {u["user_id"]: u for u in store.list_users()}
    assert users["s01"]["active"] is False
    assert users["s02"]["active"] is True
    assert set(users) == {"s01", "s02"}


# -- sessions -----------------------------------------------------------------


def test_session_expiry_and_sliding_ttl(store):
    store.create_user("s01", "student", "h", at=T0)
    sid = store.create_session("s01", T0, ttl_h=12)
    # created 09:00 -> expires 21:00
    s = store.get_session(sid, T1)  # 09:30: valid, slides expiry to 21:30
    assert s is not None and s["user_id"] == "s01"
    assert store.get_session(sid, "2026-10-21T21:15:00Z")["expires_at"] == "2026-10-22T09:15:00Z"
    # idle > 12 h past the last slide -> expired and deleted
    assert store.get_session(sid, "2026-10-22T09:16:00Z") is None
    assert store.get_session(sid, "2026-10-22T09:17:00Z") is None
    assert store.sessions_for_user("s01", T1) == []
    store.close()


def test_sessions_for_user_lists_active_only(store):
    store.create_user("s01", "student", "h", at=T0)
    sid = store.create_session("s01", T0, ttl_h=12)
    rows = store.sessions_for_user("s01", T1)
    assert [r["session_id"] for r in rows] == [sid]
    store.delete_session(sid)
    assert store.sessions_for_user("s01", T1) == []


# -- Pi tokens ------------------------------------------------------------------


def test_token_shown_once_hashed_at_rest_and_revocable(store):
    store.create_user("s07", "student", "h", at=T0)
    tid = store.create_token("s07", "hash-abc", at=T0)
    row = store.token_by_hash("hash-abc")
    assert row is not None and row["user_id"] == "s07" and row["token_id"] == tid
    assert store.token_by_hash("wrong") is None
    assert store.revoke_token(tid, at=T1) is True
    assert store.token_by_hash("hash-abc") is None  # revoked tokens stop working
    assert store.revoke_token(tid, at=T1) is False  # already revoked
    toks = store.list_tokens("s07")
    assert len(toks) == 1 and toks[0]["revoked_at"] == T1


# -- agent tokens -----------------------------------------------------------------


def test_agent_token_set_rotate_and_lookup(store):
    store.set_agent_token("pai1", "hash-1", at=T0)
    assert store.get_agent_token_hash("pai1") == "hash-1"
    store.set_agent_token("pai1", "hash-2", at=T1)  # rotation upserts
    assert store.get_agent_token_hash("pai1") == "hash-2"
    assert store.get_agent_token_hash("node2") is None


# -- datasets + chunks -------------------------------------------------------------


def test_dataset_lifecycle_and_chunk_accounting(store):
    ds = store.create_dataset("s07/task-stop-at-target/v2", "s07", "uploading", at=T0)
    assert ds["status"] == "uploading" and ds["total_bytes"] == 0
    store.add_chunk("s07/task-stop-at-target/v2", 0, "sha-a", 1000)
    store.add_chunk("s07/task-stop-at-target/v2", 1, "sha-b", 500)
    ds = store.get_dataset("s07/task-stop-at-target/v2")
    assert ds["total_bytes"] == 1500
    assert store.chunks("s07/task-stop-at-target/v2")[1]["sha256"] == "sha-b"
    store.set_dataset("s07/task-stop-at-target/v2", status="ready",
                      qa={"status": "ready"}, at=T1)
    ds = store.get_dataset("s07/task-stop-at-target/v2")
    assert ds["status"] == "ready" and ds["qa"] == {"status": "ready"}
    assert store.list_datasets(owner="s07")[0]["owner"] == "s07"
    assert store.list_datasets()[0]["dataset_id"].startswith("s07/")


def test_set_dataset_unknown_raises(store):
    with pytest.raises(KeyError):
        store.set_dataset("s07/nope/v1", status="ready")


# -- tickets -----------------------------------------------------------------------


def test_ticket_row_roundtrip_and_by_job(store):
    t = {
        "ticket_id": "t_001", "owner": "s07", "hypothesis": "h", "config_summary": "c",
        "prediction": "p", "linked_job": None, "outcome": None, "state": "unlinked",
        "created_at": T0, "updated_at": T0,
    }
    store.save_ticket(t)
    got = store.get_ticket("t_001")
    assert got["state"] == "unlinked" and got["linked_job"] is None
    store.save_ticket({**t, "state": "linked", "linked_job": "j_20261021_0001", "updated_at": T1})
    assert store.ticket_by_job("j_20261021_0001")["ticket_id"] == "t_001"
    assert [x["ticket_id"] for x in store.list_tickets(owner="s07")] == ["t_001"]


# -- jobs + idempotency -------------------------------------------------------------


SPEC = {"student": "s07", "type": "train", "job_id": "j_20261021_0001"}


def test_job_insert_get_list_and_terminal_fields(store):
    store.insert_job("j_20261021_0001", SPEC, "queued", 1000.0,
                     ticket_id="t_001", created_at=T0)
    j = store.get_job("j_20261021_0001")
    assert j["spec"] == SPEC and j["state"] == "queued" and j["run_now"] is False
    j = store.set_job("j_20261021_0001", state="running", node_id="pai1")
    assert j["node_id"] == "pai1"
    j = store.set_job("j_20261021_0001", state="done", terminal_at=T1)
    assert j["terminal_at"] == T1
    assert [x["job_id"] for x in store.list_jobs(student="s07")] == ["j_20261021_0001"]
    assert store.list_jobs(student="s01") == []
    assert [x["job_id"] for x in store.list_jobs(state_in=("done",))] == ["j_20261021_0001"]


def test_idempotency_24h_window(store):
    store.insert_job("j_20261021_0001", SPEC, "queued", 1000.0,
                     idempotency_key="key-1", created_at=T0)
    assert store.job_by_idempotency("key-1", T1)["job_id"] == "j_20261021_0001"
    # after 24 h the key expires -> a new submission is not deduped
    assert store.job_by_idempotency("key-1", "2026-10-22T10:00:00Z") is None
    assert store.job_by_idempotency("other", T1) is None


# -- artifacts / audit / backups / kv -------------------------------------------------


def test_artifacts(store):
    store.insert_job("j_20261021_0001", SPEC, "done", 1000.0, created_at=T0)
    store.add_artifact("j_20261021_0001", "adapter", "artifacts/j_.../adapter.safetensors", 123, at=T1)
    store.add_artifact("j_20261021_0001", "run_config", "artifacts/j_.../run.json", 77, at=T1)
    arts = store.artifacts_for("j_20261021_0001")
    assert {a["kind"] for a in arts} == {"adapter", "run_config"}
    assert len(store.list_artifacts()) == 2


def test_audit_append_only_and_filter(store):
    store.append_audit(T0, "teacher", "user.create", "s01", after={"role": "student"})
    store.append_audit(T1, "teacher", "user.activate", "s01", after={"active": True})
    rows = store.audit_entries()
    assert len(rows) == 2 and rows[0]["at"] == T0
    assert store.audit_entries(since=T1)[0]["action"] == "user.activate"
    assert store.audit_entries(actor="teacher", action="user.create")[0]["target"] == "s01"


def test_backups_and_kv(store):
    store.add_backup(T0, "/home/adam/robovla-backups/2026-10-21T02:30", 1234, "abc", "ok")
    assert store.list_backups()[0]["status"] == "ok"
    store.set_kv("maintenance", "off")
    assert store.get_kv("maintenance") == "off"
    store.set_kv("maintenance", "on")
    assert store.get_kv("maintenance") == "on"
    assert store.get_kv("missing") is None
