"""UI smoke rows (design §7/§11): the static console is mounted at /ui,
served as-is (readable by students), and the pages are wired to the §9
JSON API + SSE. No browser in CI — the rows verify serving, wiring, and
the login contract the pages depend on."""

from __future__ import annotations

import pathlib

import pytest

from conftest import Env

UI = pathlib.Path(__file__).resolve().parents[2] / "ui"

PAGES = ["login.html", "logout.html", "index.html", "datasets.html",
         "new-run.html", "run.html", "class-board.html", "slots.html",
         "teacher.html", "accounts.html"]


@pytest.fixture
def ui_env(tmp_path):
    e = Env(tmp_path, ui_dir=UI)
    yield e
    e.close()


def test_ui_static_pages_served(ui_env):
    c = ui_env.client
    for p in PAGES:
        r = c.get(f"/ui/{p}")
        assert r.status_code == 200, p
        assert r.headers["content-type"].startswith("text/html"), p
        if p != "logout.html":
            assert "roboVLA" in r.text, p
    # shared assets + the vendored htmx (LAN-offline, §7)
    r = ui_env.client.get("/ui/app.js")
    assert r.status_code == 200 and "function api(" in r.text
    r = ui_env.client.get("/ui/style.css")
    assert r.status_code == 200 and "var(--bg)" in r.text
    r = ui_env.client.get("/ui/vendor/htmx.min.js")
    assert r.status_code == 200 and len(r.text) > 10_000
    # /ui/ -> index.html (html=True)
    r = ui_env.client.get("/ui/")
    assert r.status_code == 200 and "Dashboard" in r.text


def test_ui_pages_wired_to_the_api(ui_env):
    """Guard against wiring drift: the pages must call the real routes."""
    html = {p: ui_env.client.get(f"/ui/{p}").text for p in PAGES}
    assert "/api/runs/queue" in html["index.html"]
    # the SSE stream URL is composed by sse() in app.js; the pages call it
    assert 'sse("class"' in html["index.html"]
    assert "/stream/" in open(UI / "app.js").read()
    assert "/api/datasets/upload" in html["datasets.html"]
    assert "/api/runs" in html["new-run.html"] and "/api/tickets" in html["new-run.html"]
    assert "/api/runs/" in html["run.html"]
    assert "/api/infer/slots/claim" in html["slots.html"]
    assert "/api/auth/token" in html["slots.html"]  # the Pi token, shown once
    assert "/api/admin/nodes" in html["teacher.html"]
    assert "/api/admin/audit" in html["teacher.html"]
    assert "/api/admin/users" in html["accounts.html"]
    # nav is hx-boosted (htmx is actually used, not just vendored)
    assert html["index.html"].count('hx-boost="true"') == 1
    # the error shape the pages render (§9.6)
    assert "error.message" in html["login.html"] or "error" in html["login.html"]


def test_ui_login_contract(ui_env):
    c = ui_env.client
    # the static page loads unauthenticated; the API is what gates it
    assert c.get("/ui/index.html").status_code == 200
    assert c.get("/api/auth/me").status_code == 401
    r = c.post("/api/auth/login", json={"username": "s01", "password": "wrong"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "auth"
    r = c.post("/api/auth/login", json={"username": "s01", "password": "stu1123"})
    assert r.status_code == 200
    r = c.get("/api/auth/me")
    assert r.status_code == 200 and r.json()["user_id"] == "s01"
    assert r.json()["role"] == "student"


def test_health_exposes_fleet_config_for_the_forms(ui_env):
    """new-run.html filters node + batch options from /api/health (public)."""
    r = ui_env.client.get("/api/health")
    assert r.status_code == 200
    node = r.json()["nodes"][0]
    assert node["student_selectable"] is True
    assert node["batch_presets"] == [1, 4, 8]
