"""backup.py — the backup/restore seam shared by the admin route and the CLI
(design §8.4/§8.5: nightly backup = DB + datasets + adapters; restore is the
documented SPOF procedure, covered here).

The core correctness property: the console DB is SQLite in WAL mode, so a
*raw file copy* of state.db is not a consistent snapshot. ``snapshot_db``
uses sqlite3's online backup API; the test below keeps writes parked in the
-wal file and proves a restore sees them.

Restore is zip-slip-safe with the same discipline as ``fs.safe_extract_zip``
and only accepts the bundle layout: ``datasets/**``, ``artifacts/**``, and a
top-level ``state.db``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from robovla import backup
from robovla.core.errors import RobovlaError
from robovla.storage import Storage

CLOCK = lambda: "2026-10-21T09:00:00Z"


# -- helpers -------------------------------------------------------------------


def _zip(entries: dict[str, bytes], symlinks: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        for name, target in (symlinks or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o120777 << 16  # symlink mode bits
            zf.writestr(info, target)
    return buf.getvalue()


def _bundle_dir(tmp_path: Path, data_dir: Path) -> Path:
    dest = tmp_path / "backups"
    backup.make_backup_bundle(data_dir, dest, clock=CLOCK)
    zips = sorted(dest.glob("robovla-backup-*.zip"))
    assert len(zips) == 1
    return zips[0]


# -- snapshot_db: WAL consistency ------------------------------------------------


def test_snapshot_db_captures_wal_pending_rows(tmp_path):
    """The bug this seam exists to fix: with autocheckpoint disabled, committed
    rows live only in state.db-wal. A raw copy of state.db would miss them;
    the online backup must not."""
    data = tmp_path / "data"
    db = data / "state.db"
    s = Storage(db)
    s.set_kv("k1", "v1")
    s._db.execute("PRAGMA wal_autocheckpoint = 0")  # keep everything in -wal
    s.set_kv("k2", "v2")
    assert (data / "state.db-wal").exists()  # precondition: rows are in the WAL

    out = tmp_path / "snap.db"
    assert backup.snapshot_db(db, out) is True
    s.close()  # close (and checkpoint) AFTER the snapshot

    import sqlite3
    with sqlite3.connect(out) as c:
        rows = dict(c.execute("SELECT key, value FROM kv"))
    assert rows == {"k1": "v1", "k2": "v2"}


def test_snapshot_db_missing_returns_false(tmp_path):
    assert backup.snapshot_db(tmp_path / "no.db", tmp_path / "out.db") is False


# -- make_backup_bundle ------------------------------------------------------------


def test_bundle_contains_datasets_artifacts_and_db(tmp_path):
    data = tmp_path / "data"
    s = Storage(data / "state.db")
    s.set_kv("cfg", "1")
    s.close()
    (data / "datasets" / "s01" / "x" / "v1").mkdir(parents=True)
    (data / "datasets" / "s01" / "x" / "v1" / "ep.json").write_bytes(b'{"n": 1}')
    (data / "artifacts" / "j_20261021_0001").mkdir(parents=True)
    (data / "artifacts" / "j_20261021_0001" / "adapter.bin").write_bytes(b"adpt")

    recorded: list[dict] = []
    dest = tmp_path / "bk"
    info = backup.make_backup_bundle(data, dest, clock=CLOCK, record=recorded.append)

    assert info["status"] == "ok"
    assert info["path"].endswith("robovla-backup-2026-10-21T09-00-00Z.zip")
    blob = Path(info["path"]).read_bytes()
    assert info["size_bytes"] == len(blob)
    import hashlib
    assert info["sha256"] == hashlib.sha256(blob).hexdigest()
    with zipfile.ZipFile(info["path"]) as zf:
        assert set(zf.namelist()) == {
            "state.db",
            "datasets/s01/x/v1/ep.json",
            "artifacts/j_20261021_0001/adapter.bin",
        }
    assert recorded == [info]  # the record hook saw the same info


def test_bundle_without_db_or_dirs_is_still_valid(tmp_path):
    data = tmp_path / "empty"
    data.mkdir()
    dest = tmp_path / "bk"
    info = backup.make_backup_bundle(data, dest, clock=CLOCK)
    with zipfile.ZipFile(info["path"]) as zf:
        assert zf.namelist() == []


# -- restore_backup ------------------------------------------------------------------


def test_restore_roundtrip(tmp_path):
    data = tmp_path / "data"
    s = Storage(data / "state.db")
    s.set_kv("k", "v")
    s.close()
    (data / "datasets" / "s01" / "x" / "v1").mkdir(parents=True)
    (data / "datasets" / "s01" / "x" / "v1" / "ep.json").write_bytes(b'{"n": 1}')

    bundle = _bundle_dir(tmp_path, data)
    fresh = tmp_path / "restored"
    out = backup.restore_backup(bundle, fresh)

    assert out["db"] is True
    assert out["files"] == 2  # state.db + ep.json
    assert (fresh / "datasets" / "s01" / "x" / "v1" / "ep.json").read_bytes() == b'{"n": 1}'
    s2 = Storage(fresh / "state.db")
    assert s2.get_kv("k") == "v"  # the restored DB opens and serves reads
    s2.close()


def test_restore_without_db_reports_no_db(tmp_path):
    data = tmp_path / "data"
    (data / "datasets" / "s01" / "x" / "v1").mkdir(parents=True)
    (data / "datasets" / "s01" / "x" / "v1" / "ep.json").write_bytes(b"e")
    bundle = _bundle_dir(tmp_path, data)
    out = backup.restore_backup(bundle, tmp_path / "fresh")
    assert out["db"] is False
    assert out["files"] == 1


def test_restore_refuses_existing_db_without_force(tmp_path):
    bundle = _bundle_dir(tmp_path, _seeded(tmp_path))
    target = tmp_path / "t"
    target.mkdir()
    (target / "state.db").write_bytes(b"old")
    with pytest.raises(RobovlaError) as ei:
        backup.restore_backup(bundle, target)
    assert ei.value.code == "conflict"
    out = backup.restore_backup(bundle, target, force=True)
    assert out["db"] is True


def test_restore_rejects_malicious_and_foreign_entries(tmp_path):
    cases = [
        {"datasets/../evil.json": b"x"},  # '..' traversal (checked on raw name)
        {"/abs.json": b"x"},              # absolute path
        {"other/x.json": b"x"},           # foreign top level
    ]
    for entries in cases:
        blob = _zip(entries)
        with pytest.raises(RobovlaError) as ei:
            backup.restore_backup(_write(tmp_path, blob), tmp_path / "f")
        assert ei.value.code == "validation"
    symlink = _zip({"datasets/a.json": b"x"},
                   symlinks={"datasets/link": "../../etc/passwd"})
    with pytest.raises(RobovlaError) as ei:
        backup.restore_backup(_write(tmp_path, symlink), tmp_path / "f")
    assert ei.value.code == "validation"
    # and the good entry alone is fine
    ok = _write(tmp_path, _zip({"datasets/a.json": b"ok"}))
    assert backup.restore_backup(ok, tmp_path / "g")["files"] == 1


def test_restore_rejects_garbage_zip(tmp_path):
    blob = tmp_path / "junk.zip"
    blob.write_bytes(b"not a zip at all")
    with pytest.raises(RobovlaError) as ei:
        backup.restore_backup(blob, tmp_path / "f")
    assert ei.value.code == "validation"


# -- module-level helpers -----------------------------------------------------------


def _seeded(tmp_path: Path) -> Path:
    data = tmp_path / "seed"
    s = Storage(data / "state.db")
    s.set_kv("k", "v")
    s.close()
    return data


def _write(tmp_path: Path, blob: bytes) -> Path:
    p = tmp_path / f"case-{len(list(tmp_path.iterdir()))}.zip"
    p.write_bytes(blob)
    return p
