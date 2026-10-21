"""Backup & restore seam (design §8.4/§8.5).

Shared by the admin route (``POST /api/admin/backups/trigger``) and the CLI
(``robovla cli backup|restore``). The console DB is SQLite in WAL mode, so a
snapshot goes through sqlite3's **online backup API** — a raw file copy of
``state.db`` is not a consistent snapshot (committed rows may live only in
the ``-wal`` file).

Bundle layout (zip):

    state.db          DB snapshot (top level)
    datasets/**       uploaded / prepared datasets
    artifacts/**      stored artifacts (adapters, merged exports, ...)

``datasets/`` on a *node* workdir is the pulled bundle — ephemeral and
excluded from node backups (§10); this seam covers the **console** data dir.
Restore accepts exactly that layout and nothing else (zip-slip-safe, same
discipline as ``fs.safe_extract_zip``).
"""

from __future__ import annotations

import hashlib
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .core.errors import RobovlaError
from .fs import zip_entry_is_link

#: Top-level dirs zipped into a bundle (relative to the console data dir).
_BUNDLE_ROOTS = ("datasets", "artifacts")

#: Top-level DB file inside a bundle (and in the console data dir).
_DB_NAME = "state.db"


def now_rfc3339() -> str:
    """Current UTC time, RFC 3339 (the console timestamp format)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot_db(db_path: str | Path, out_path: str | Path) -> bool:
    """Consistent snapshot of a SQLite DB via the online backup API.

    WAL-safe: committed rows parked in the ``-wal`` file are captured.
    Returns ``False`` (writes nothing) when ``db_path`` does not exist.
    Raises ``sqlite3.DatabaseError`` on a corrupt/unreadable source.
    """
    db_path, out_path = Path(db_path), Path(out_path)
    if not db_path.exists():
        return False
    src = sqlite3.connect(str(db_path))
    try:
        dst = sqlite3.connect(str(out_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return True


def make_backup_bundle(
    data_dir: str | Path,
    dest: str | Path,
    *,
    clock: Callable[[], str] | None = None,
    record: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Create the backup bundle (§8.4): datasets + artifacts + DB snapshot.

    ``dest`` is the destination *directory* (created if needed); the bundle
    is written as ``robovla-backup-<stamp>.zip`` with the ``:`` in the stamp
    replaced. ``record`` (optional) is called with the result dict — the
    admin route uses it to log the ``backups`` table row. Returns
    ``{path, size_bytes, sha256, status}``.
    """
    data_dir, dest = Path(data_dir), Path(dest)
    clock = clock or now_rfc3339
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"robovla-backup-{clock().replace(':', '-')}.zip"
    snap = dest / f".{out.name}.db"  # temp snapshot, never inside data_dir
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for sub in _BUNDLE_ROOTS:
                base = data_dir / sub
                if base.is_dir():
                    for f in sorted(base.rglob("*")):
                        if f.is_file():
                            zf.write(f, f.relative_to(data_dir).as_posix())
            if snapshot_db(data_dir / _DB_NAME, snap):
                zf.write(snap, _DB_NAME)
    finally:
        snap.unlink(missing_ok=True)
    blob = out.read_bytes()
    info: dict[str, Any] = {
        "path": str(out),
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "status": "ok",
    }
    if record is not None:
        record(info)
    return info


def restore_backup(
    bundle: str | Path,
    data_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Restore a bundle into a console data dir (the SPOF procedure, §10).

    Accepts exactly the bundle layout: ``datasets/**``, ``artifacts/**``,
    top-level ``state.db``. Refuses a **conflict** when
    ``data_dir/state.db`` already exists unless ``force`` replaces it.
    Returns ``{files: <count incl. the DB>, db: bool}``.
    """
    bundle, data_dir = Path(bundle), Path(data_dir)
    try:
        zf = zipfile.ZipFile(bundle)
    except (zipfile.BadZipFile, OSError) as e:
        raise RobovlaError(
            "validation", f"backup bundle is not a valid zip: {e}") from e
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    with zf:
        names = zf.namelist()
        if _DB_NAME in names and (data_dir / _DB_NAME).exists() and not force:
            raise RobovlaError(
                "conflict",
                f"{data_dir / _DB_NAME} already exists — re-run with force to replace")
        written = 0
        for info in zf.infolist():
            name = info.filename
            if "\x00" in name:
                raise RobovlaError("validation", f"refusing entry with NUL byte: {name!r}")
            if name.startswith("/") or ".." in Path(name).parts:
                raise RobovlaError("validation", f"zip-slip: unsafe entry {name!r}")
            top = Path(name).parts[0] if name else ""
            if top not in (*_BUNDLE_ROOTS, _DB_NAME):
                raise RobovlaError(
                    "validation", f"foreign top-level entry {name!r} (not a backup bundle)")
            if zip_entry_is_link(info):
                raise RobovlaError("validation", f"refusing symlink entry {name!r}")
            target = (data_dir / name).resolve()
            if target != data_dir and data_dir not in target.parents:
                raise RobovlaError(
                    "validation", f"zip-slip: entry {name!r} escapes the data dir")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written += 1
    return {"files": written, "db": _DB_NAME in names}
