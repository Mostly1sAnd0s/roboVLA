"""agent.pull — dataset bundle pull from the console (§5.1, §9.9).

Console side is a MockTransport (no sockets): the same tar the real console
would serve (``fs.make_bundle``) with the ``X-Dataset-Sha256`` header.
Covers: success + idempotent re-pull, sha mismatch, console error pass-
through (auth / dataset_not_ready), unreachable console -> node_offline,
stale-dir replacement, bad dataset id.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import httpx
import pytest
from PIL import Image

from agent.pull import pull_bundle
from robovla.core.errors import RobovlaError
from robovla.fs import make_bundle, dataset_root

DS = "s07/task-stop-at-target/v1"
CONSOLE = "http://10.0.0.5:8501"


def _ready_dataset(data_dir) -> None:
    """A minimal ready dataset (same shape the fs tests use)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(3):
            fb = io.BytesIO()
            Image.new("RGB", (8, 8)).save(fb, "JPEG")
            zf.writestr(f"episodes/ep_{i:04d}/frames/0.jpg", fb.getvalue())
            zf.writestr(f"episodes/ep_{i:04d}/actions.json", "[]")
            zf.writestr(f"episodes/ep_{i:04d}/episode.json",
                        json.dumps({"id": f"ep_{i:04d}", "task": "stop-at-target"}))
    from robovla.fs import safe_extract_zip
    safe_extract_zip(buf.getvalue(), dataset_root(data_dir, DS))


def _console(handler_app=None, bundle: bytes | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != "Bearer tok-agent":
            return httpx.Response(401, json={"error": {"code": "auth",
                                                       "message": "invalid agent token"}})
        if request.url.path != f"/api/agent/datasets/{DS}/bundle":
            return httpx.Response(404, json={"error": {"code": "not_found",
                                                       "message": "no route"}})
        if handler_app is not None:
            return handler_app(request)
        blob = bundle
        return httpx.Response(200, content=blob,
                              headers={"X-Dataset-Sha256":
                                       hashlib.sha256(blob).hexdigest(),
                                       "Content-Type": "application/gzip"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


@pytest.fixture
def data(tmp_path):
    _ready_dataset(tmp_path)
    size, _ = make_bundle(tmp_path, DS, tmp_path / "bundle.tar.gz")
    assert size > 0
    return tmp_path


@pytest.fixture
def workdir(tmp_path):
    return tmp_path / "node"


def test_pull_success_extracts_into_workdir(data, workdir):
    out = pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=_console(bundle=(data / "bundle.tar.gz").read_bytes()))
    assert out == workdir / "datasets" / "s07__task-stop-at-target__v1"
    assert (out / "episodes" / "ep_0000" / "episode.json").is_file()
    # no temp file left behind
    assert not (workdir / "bundles").exists() or not list((workdir / "bundles").iterdir())


def test_repull_replaces_previous_extraction(data, workdir):
    bundle = (data / "bundle.tar.gz").read_bytes()
    first = pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=_console(bundle=bundle))
    (first / "marker.txt").write_text("stale content from a previous pull")
    # re-dispatch re-pulls: the console's bundle is the source of truth (§5.1)
    out = pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=_console(bundle=bundle))
    assert out == first
    assert not (out / "marker.txt").exists()
    assert (out / "episodes" / "ep_0000" / "episode.json").is_file()


def test_pull_bad_token_maps_to_auth(data, workdir):
    with pytest.raises(RobovlaError) as ei:
        pull_bundle(CONSOLE, "wrong", DS, workdir, client=_console())
    assert ei.value.code == "auth"


def test_pull_console_error_codes_pass_through(data, workdir):
    def not_ready(request):
        return httpx.Response(409, json={"error": {
            "code": "dataset_not_ready",
            "message": f"dataset '{DS}' is not ready to pull"}})
    with pytest.raises(RobovlaError) as ei:
        pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=_console(not_ready))
    assert ei.value.code == "dataset_not_ready"

    def weird_code(request):
        return httpx.Response(500, json={"error": {
            "code": "totally_new_code", "message": "boom"}})
    with pytest.raises(RobovlaError) as ei:
        pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=_console(weird_code))
    assert ei.value.code == "internal"  # unknown codes collapse, §9.6


def test_pull_sha_mismatch_is_conflict(data, workdir):
    def bad_sha(request):
        blob = (data / "bundle.tar.gz").read_bytes()
        return httpx.Response(200, content=blob,
                              headers={"X-Dataset-Sha256": "0" * 64})
    with pytest.raises(RobovlaError) as ei:
        pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=_console(bad_sha))
    assert ei.value.code == "conflict" and "sha256" in ei.value.message


def test_pull_missing_sha_header_is_conflict(data, workdir):
    def no_sha(request):
        return httpx.Response(200, content=(data / "bundle.tar.gz").read_bytes())
    with pytest.raises(RobovlaError) as ei:
        pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=_console(no_sha))
    assert ei.value.code == "conflict"


def test_pull_unreachable_console_is_node_offline(data, workdir):
    def boom(request):
        raise httpx.ConnectError("refused")
    client = httpx.Client(transport=httpx.MockTransport(boom))
    with pytest.raises(RobovlaError) as ei:
        pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=client)
    assert ei.value.code == "node_offline"


def test_pull_replaces_stale_extraction(data, workdir):
    stale = workdir / "datasets" / "s07__task-stop-at-target__v1" / "stale"
    stale.parent.mkdir(parents=True)
    (stale.parent / "stale.json").write_text("{}")  # populated but stale
    out = pull_bundle(CONSOLE, "tok-agent", DS, workdir, client=_console(bundle=(data / "bundle.tar.gz").read_bytes()))
    assert not (out / "stale.json").exists()
    assert (out / "episodes" / "ep_0000" / "episode.json").is_file()


def test_pull_bad_dataset_id_is_validation(data, workdir):
    with pytest.raises(RobovlaError) as ei:
        pull_bundle(CONSOLE, "tok-agent", "../escape/v1", workdir, client=_console())
    assert ei.value.code == "validation"
