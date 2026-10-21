"""fs — dataset/artifact file seam (design §5.1, §11 test matrix:
chunked upload incl. zip-slip payload rejected).

Zip-slip matrix rows: ``../`` traversal, absolute paths, symlink entries,
non-jpeg/json extensions. Plus: chunked-upload reassembly + whole-zip sha,
episode-tree scan into shared Episode objects, 224^2 frame prep, and the
agent-pull bundle (sha256 for §9.9).
"""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from robovla.core.errors import RobovlaError
from robovla.fs import (
    assemble_chunks,
    clear_chunks,
    make_bundle,
    resize_frames,
    safe_extract_tar,
    safe_extract_zip,
    scan_dataset,
    write_chunk,
    dataset_root,
    episodes_dir,
    sanitize_dataset_id,
)

DS = "s07/task-stop-at-target/v2"


def _episode_zip(tmp: Path, ep_count: int = 2) -> bytes:
    """A well-formed dataset zip: episode folders with frames/actions/meta."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(ep_count):
            ep = f"ep_{i:04d}"
            frames = f"episodes/{ep}/frames/"
            for f in range(3):
                zf.writestr(f"{frames}{f}.jpg", _jpg_bytes(i * 3 + f, f + 1))
            actions = [{"t": k * 0.1, "vx": 0.1, "vy": 0.0, "wz": 0.05} for k in range(3)]
            zf.writestr(f"episodes/{ep}/actions.json", json.dumps(actions))
            meta = {
                "id": ep,
                "task": "stop-at-target",
                "prompt": "stop at the target",
                "success": True,
                "pilot": "s07",
                "created": "2026-10-20T10:00:00Z",
            }
            zf.writestr(f"episodes/{ep}/episode.json", json.dumps(meta))
    return buf.getvalue()


def _jpg_bytes(r: int, g: int, b: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color=(r % 256, g % 256, b)).save(buf, "JPEG")
    return buf.getvalue()


def _zip_with_entry(name: str, payload: bytes = b"evil") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, payload)
    return buf.getvalue()


# --- zip-slip matrix -----------------------------------------------------------


def test_zip_slip_traversal_rejected(tmp_path):
    blob = _zip_with_entry("../../../etc/evil.json")
    with pytest.raises(RobovlaError, match="zip-slip"):
        safe_extract_zip(blob, tmp_path / "ds")


def test_zip_slip_absolute_path_rejected(tmp_path):
    blob = _zip_with_entry("/etc/passwd.json")
    with pytest.raises(RobovlaError, match="zip-slip"):
        safe_extract_zip(blob, tmp_path / "ds")


def test_zip_symlink_entry_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("episodes/ep_0001/link.json")
        info.external_attr = 0o120777 << 16  # unix symlink mode
        zf.writestr(info, b"../secret")
    with pytest.raises(RobovlaError, match="symlink"):
        safe_extract_zip(buf.getvalue(), tmp_path / "ds")


def test_non_jpeg_json_extension_rejected(tmp_path):
    blob = _zip_with_entry("episodes/ep_0001/payload.py")
    with pytest.raises(RobovlaError, match="only jpeg/json"):
        safe_extract_zip(blob, tmp_path / "ds")


def test_bad_zip_rejected(tmp_path):
    with pytest.raises(RobovlaError, match="not a valid zip"):
        safe_extract_zip(b"not a zip at all", tmp_path / "ds")


def test_wellformed_zip_extracts_and_scans(tmp_path):
    blob = _episode_zip(tmp_path)
    dest = tmp_path / "ds"
    safe_extract_zip(blob, dest)
    scan = scan_dataset(dest / "episodes")
    assert len(scan.episodes) == 2
    assert scan.task == "stop-at-target"
    assert scan.episodes[0].frame_count == 3
    assert len(scan.episodes[0].actions) == 3
    assert scan.total_bytes > 0


# --- chunked upload --------------------------------------------------------------


def test_chunk_reassembly_and_sha(tmp_path):
    blob = _episode_zip(tmp_path)
    import hashlib

    sha = hashlib.sha256(blob).hexdigest()
    third = len(blob) // 3
    parts = [blob[:third], blob[third : 2 * third], blob[2 * third :]]
    for i, p in enumerate(parts):
        write_chunk(tmp_path, DS, i, p)
    out = assemble_chunks(tmp_path, DS, len(parts), sha)
    assert out == blob
    clear_chunks(tmp_path, DS)
    with pytest.raises(RobovlaError, match="missing"):
        assemble_chunks(tmp_path, DS, len(parts), sha)


def test_chunk_sha_mismatch_rejected(tmp_path):
    blob = _episode_zip(tmp_path)
    write_chunk(tmp_path, DS, 0, blob)
    with pytest.raises(RobovlaError, match="sha256 mismatch"):
        assemble_chunks(tmp_path, DS, 1, "0" * 64)


def test_dataset_id_sanitization():
    assert sanitize_dataset_id(DS) == "s07__task-stop-at-target__v2"
    with pytest.raises(RobovlaError):
        sanitize_dataset_id("s07/../etc/v1")


# --- frame prep (224^2) -----------------------------------------------------------


def test_resize_frames_to_224(tmp_path):
    blob = _episode_zip(tmp_path)
    safe_extract_zip(blob, tmp_path / "ds")
    eps = tmp_path / "ds" / "episodes"
    n = resize_frames(eps)
    assert n == 6  # 2 episodes x 3 frames
    for p in eps.rglob("*.jpg"):
        with Image.open(p) as im:
            assert im.size == (224, 224)


# --- bundle (agent pull) ------------------------------------------------------------


def test_bundle_tar_and_sha(tmp_path):
    blob = _episode_zip(tmp_path)
    safe_extract_zip(blob, dataset_root(tmp_path, DS))
    (dataset_root(tmp_path, DS) / "norm_stats.json").write_text(json.dumps({"min": [0] * 7}))
    size, sha = make_bundle(tmp_path, DS, tmp_path / "bundles" / f"{DS}.tar.gz")
    assert size > 0 and len(sha) == 64
    with tarfile.open(tmp_path / "bundles" / f"{DS}.tar.gz") as tf:
        names = tf.getnames()
    assert any(n.endswith("norm_stats.json") for n in names)
    assert any(n.endswith("episode.json") for n in names)


def test_episode_scan_rejects_mixed_tasks(tmp_path):
    blob = _episode_zip(tmp_path, ep_count=1)
    buf = io.BytesIO(blob)
    with zipfile.ZipFile(buf) as zf:
        meta = json.loads(zf.read("episodes/ep_0000/episode.json"))
    meta["id"] = "ep_0001"
    meta["task"] = "other-task"
    buf2 = io.BytesIO()
    with zipfile.ZipFile(buf2, "w") as zf:
        for n in ("episodes/ep_0000/episode.json", "episodes/ep_0000/actions.json",
                  "episodes/ep_0000/frames/0.jpg"):
            with zipfile.ZipFile(io.BytesIO(blob)) as src:
                zf.writestr(n, src.read(n))
        zf.writestr("episodes/ep_0001/episode.json", json.dumps(meta))
        zf.writestr("episodes/ep_0001/actions.json", "[]")
        zf.writestr("episodes/ep_0001/frames/0.jpg", _jpg_bytes(1, 2))
    safe_extract_zip(buf2.getvalue(), tmp_path / "ds")
    with pytest.raises(RobovlaError, match="mixes tasks"):
        scan_dataset(tmp_path / "ds" / "episodes")


# --- safe tar extraction (agent-pull seam, §9.9) -------------------------------------


def _tar_bytes(entries: dict[str, bytes | None]) -> bytes:
    """Build a tar.gz; ``None`` value = directory."""
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            else:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _link_tar(name: str, linkname: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name)
        info.type = tarfile.SYMTYPE
        info.linkname = linkname
        tf.addfile(info)
    return buf.getvalue()


def test_tar_roundtrip_matches_zip_contents(tmp_path):
    src = dataset_root(tmp_path, DS)
    safe_extract_zip(_episode_zip(tmp_path), src)
    size, _ = make_bundle(tmp_path, DS, tmp_path / "b.tar.gz")
    dest = tmp_path / "node" / "datasets"
    written = safe_extract_tar((tmp_path / "b.tar.gz").read_bytes(), dest)
    assert size > 0
    # same episode lands at workdir/datasets/<sanitized>/episodes/...
    root = dest / sanitize_dataset_id(DS)
    assert (root / "episodes" / "ep_0000" / "episode.json").is_file()
    assert any(p.suffix == ".json" for p in written)


def test_tar_slip_traversal_rejected(tmp_path):
    blob = _tar_bytes({"episodes/../evil.json": b"{}"})
    with pytest.raises(RobovlaError, match="tar-slip"):
        safe_extract_tar(blob, tmp_path / "ds")


def test_tar_slip_absolute_path_rejected(tmp_path):
    blob = _tar_bytes({"/etc/evil.json": b"{}"})
    with pytest.raises(RobovlaError, match="tar-slip"):
        safe_extract_tar(blob, tmp_path / "ds")


def test_tar_symlink_entry_rejected(tmp_path):
    with pytest.raises(RobovlaError, match="symlink"):
        safe_extract_tar(_link_tar("episodes/ep_0000/link.json", "/etc/passwd"),
                         tmp_path / "ds")


def test_tar_bad_extension_rejected(tmp_path):
    blob = _tar_bytes({"episodes/ep_0000/payload.py": b"import os"})
    with pytest.raises(RobovlaError, match="only jpeg/json"):
        safe_extract_tar(blob, tmp_path / "ds")


def test_tar_bad_archive_rejected(tmp_path):
    with pytest.raises(RobovlaError, match="not a valid tar"):
        safe_extract_tar(b"this is not a tar", tmp_path / "ds")
