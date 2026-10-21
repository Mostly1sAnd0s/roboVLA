"""The shipped roster.csv through the REAL admin import path (§8.1/§9.7):
12 students created **inactive**; the teacher activates one (out-of-band
password reset) and it can log in. End-to-end proof the shipped file
imports through the shipped routes.

One teacher login is reused throughout: the login rate limit is 5/min/IP
(§9.8) and the TestClient shares one IP — and a later student login
overwrites the shared client's session cookie, so all admin calls happen
before it.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROSTER = REPO / "roster.csv"
STUDENT_IDS = tuple(f"s{i:02d}" for i in range(1, 13))


def test_shipped_roster_imports_inactive_and_activation_works(env):
    t = env.as_teacher()  # one teacher session for the whole test
    # the harness pre-seeds s01/s07 (other integration tests use them); the
    # import route skips existing users (idempotent roster import)
    r = t.post("/api/admin/users/import", json={"csv": ROSTER.read_text()})
    assert r.status_code == 200, r.text
    expected = [uid for uid in STUDENT_IDS if uid not in ("s01", "s07")]
    assert r.json()["created"] == expected

    r = t.get("/api/admin/users")
    users = {u["user_id"]: u for u in r.json()["users"]}
    assert all(users[uid]["role"] == "student" for uid in STUDENT_IDS)
    assert "s00" not in users  # s00 is reserved for system jobs
    # every roster account the import CREATED started inactive (§8.1)
    assert all(users[uid]["active"] is False for uid in expected)

    # an imported student cannot log in while inactive (§8.1)
    r = env.client.post("/api/auth/login",
                        json={"username": "s02", "password": "anything"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "auth"

    # teacher activates s01 (per-student, not bulk) + out-of-band password
    # (classroom, no email)
    r = t.patch("/api/admin/users/s01", json={"active": True})
    assert r.status_code == 200, r.text
    assert r.json()["active"] is True
    r = t.get("/api/admin/users")
    users = {u["user_id"]: u for u in r.json()["users"]}
    assert users["s01"]["active"] is True
    assert all(users[uid]["active"] is False for uid in expected)
    r = t.post("/api/admin/users/s01/reset-password")
    assert r.status_code == 200, r.text
    pw = r.json()["password"]

    # last: the student login (its session cookie overwrites the shared
    # client, so no admin calls after this one)
    r = env.client.post("/api/auth/login", json={"username": "s01",
                                                 "password": pw})
    assert r.status_code == 200, r.text
    assert r.json()["user_id"] == "s01"
