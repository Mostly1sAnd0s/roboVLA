"""Admin password reset (§8.1): generated one-time, or an exact value set by
the operator, for any account — including another teacher."""

from __future__ import annotations

from robovla.http.auth import verify_password


def _hash(env, uid):
    row = env.svc.storage._db.execute(
        "SELECT password_hash FROM users WHERE user_id = ?", (uid,)).fetchone()
    return row[0]


class TestSetPassword:
    def test_a_teacher_sets_an_exact_password_for_a_student(self, env):
        env.as_teacher()
        r = env.client.post("/api/admin/users/s01/reset-password",
                            json={"password": "chalkboard-2026"})
        assert r.status_code == 200, r.text
        assert "password" not in r.json()          # never echoed back
        assert verify_password(_hash(env, "s01"), "chalkboard-2026")

    def test_the_new_password_works_and_the_old_one_does_not(self, env):
        env.as_teacher()
        env.client.post("/api/admin/users/s01/reset-password",
                        json={"password": "chalkboard-2026"})
        env.client.post("/api/auth/logout")
        assert env.client.post("/api/auth/login", json={
            "username": "s01", "password": "chalkboard-2026"}).status_code == 200
        assert env.client.post("/api/auth/login", json={
            "username": "s01", "password": "stu1123"}).status_code == 401

    def test_a_teacher_may_reset_another_teacher(self, env):
        from robovla.http.auth import hash_password
        env.svc.storage.create_user("teacher2", "teacher", hash_password("partner12345"),
                                    display_name="Mr T2", active=True,
                                    at="2026-10-21T09:00:00Z")
        env.as_teacher()
        r = env.client.post("/api/admin/users/teacher2/reset-password",
                            json={"password": "staffroom-2026"})
        assert r.status_code == 200, r.text
        assert verify_password(_hash(env, "teacher2"), "staffroom-2026")

    def test_short_passwords_are_refused(self, env):
        env.as_teacher()
        r = env.client.post("/api/admin/users/s01/reset-password", json={"password": "short"})
        assert r.status_code == 400 and "10" in r.text
        assert not verify_password(_hash(env, "s01"), "short")   # untouched

    def test_generated_flow_is_unchanged_and_shown_once(self, env):
        env.as_teacher()
        r = env.client.post("/api/admin/users/s01/reset-password", json={}).json()
        assert r["generated"] is True and r["shown_once"] is True and r["password"]
        assert verify_password(_hash(env, "s01"), r["password"])

    def test_a_bodyless_post_still_generates(self, env):
        env.as_teacher()
        r = env.client.post("/api/admin/users/s01/reset-password")
        assert r.status_code == 200 and r.json()["generated"] is True

    def test_unknown_user_is_404_and_Students_Cannot_call_it(self, env):
        env.as_teacher()
        assert env.client.post("/api/admin/users/nobody/reset-password",
                               json={"password": "chalkboard-2026"}).status_code == 404
        env.as_student("s01")
        assert env.client.post("/api/admin/users/s01/reset-password",
                               json={"password": "chalkboard-2026"}).status_code == 403

    def test_the_audit_row_names_actor_and_flavour_but_never_the_value(self, env):
        env.as_teacher()
        env.client.post("/api/admin/users/s01/reset-password",
                        json={"password": "chalkboard-2026"})
        rows = env.svc.storage.audit_entries(action="user.reset_password")
        assert rows, "the reset was not audited"
        row = rows[-1]
        assert row["actor"] == "teacher" and row["target"] == "s01"
        assert "chalkboard" not in str(row), "a password reached the audit trail"

    def test_the_minimum_reaches_the_page_that_asks_for_it(self, env):
        env.as_teacher()
        assert env.client.get("/api/admin/users").json()["password_min"] == 10
