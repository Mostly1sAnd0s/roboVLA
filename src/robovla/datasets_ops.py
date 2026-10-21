"""Dataset operations (design §9.2): chunked upload accounting, finalize ->
prep -> QA gate, revalidate, bundle for the agent pull, frame previews.

Uploads are raw slices of one zip (the whole-zip sha256 rides on every
chunk; verified at finalize). The 2 GB cap and the 2-concurrent limit are
§9.8 rules; zip-slip safety lives in :mod:`robovla.fs`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import fs, prep
from .core.errors import RobovlaError
from .service import MAX_UPLOADS_PER_STUDENT, ConsoleService


def upload_chunk(svc: ConsoleService, user: dict[str, Any], dataset_id: str,
                 total_chunks: int, chunk_index: int, sha256: str,
                 data: bytes) -> dict[str, Any]:
    if svc.maintenance:
        raise RobovlaError("conflict", "console is in maintenance mode (read-only)")
    if not dataset_id.startswith(f"{user['user_id']}/"):
        raise RobovlaError("forbidden", "dataset_id must be yours (sNN/task/vN)")
    owner = dataset_id.split("/")[0]
    ds = svc.storage.get_dataset(dataset_id)
    fresh = ds is None
    if fresh:
        if not svc.limits.acquire("upload", owner, MAX_UPLOADS_PER_STUDENT):
            raise RobovlaError("rate_limited", "2 concurrent uploads per student (§9.8)")
        ds = svc.storage.create_dataset(dataset_id, owner=owner, status="uploading",
                                        at=svc.clock())
    if ds["status"] not in ("uploading", "preparing"):
        if fresh:
            svc.limits.release("upload", owner)
        raise RobovlaError("conflict", f"dataset is {ds['status']}; upload closed")
    if total_chunks < 1 or not (0 <= chunk_index < total_chunks):
        raise RobovlaError("validation", "chunk_index out of range for total_chunks")
    if not sha256 or len(sha256) != 64:
        raise RobovlaError("validation", "sha256 must be a 64-hex-digit digest")
    limits = svc.student_limits(svc.storage.get_user(owner) or {"user_id": owner})
    if svc.storage.get_dataset(dataset_id)["total_bytes"] + len(data) > limits["max_dataset_bytes"]:
        if fresh:
            svc.limits.release("upload", owner)
        raise RobovlaError("limit_exceeded",
                           f"dataset exceeds the {limits['max_dataset_bytes']} byte cap")
    try:
        fs.write_chunk(svc.data_dir, dataset_id, chunk_index, data)
    except BaseException:
        if fresh:
            svc.limits.release("upload", owner)
        raise
    return {"dataset_id": dataset_id, "chunk_index": chunk_index,
            "total_bytes": svc.storage.get_dataset(dataset_id)["total_bytes"]}


def finalize_dataset(svc: ConsoleService, user: dict[str, Any], dataset_id: str,
                     total_chunks: int, sha256: str) -> dict[str, Any]:
    """Assemble the zip (409 if incomplete), extract, prep, QA gate."""
    ds = svc.own_dataset(user, dataset_id)
    if ds["status"] not in ("uploading", "preparing"):
        raise RobovlaError("conflict", f"dataset is {ds['status']}; not in upload")
    owner = ds["owner"]
    limits = svc.student_limits(svc.storage.get_user(owner) or {"user_id": owner})
    try:
        report = prep.finalize_upload(
            svc.data_dir, dataset_id, total_chunks, sha256, svc.storage,
            limits["max_dataset_bytes"], svc.clock())
    except RobovlaError as e:
        if e.code == "validation" and "missing" in e.message:
            raise RobovlaError("conflict", "upload incomplete (chunks missing)") from e
        raise
    finally:
        svc.limits.release("upload", owner)
    svc.bus.publish("class", "dataset",
                    {"dataset_id": dataset_id, "status": ds["status"],
                     "qa": report["status"]})
    return {"dataset_id": dataset_id, "status": svc.storage.get_dataset(dataset_id)["status"],
            "qa": report}


def revalidate_dataset(svc: ConsoleService, user: dict[str, Any],
                       dataset_id: str) -> dict[str, Any]:
    """Re-run prep after corrections (§9.2; the held-out split stays stable)."""
    ds = svc.own_dataset(user, dataset_id)
    owner = ds["owner"]
    limits = svc.student_limits(svc.storage.get_user(owner) or {"user_id": owner})
    report = prep.revalidate(svc.data_dir, dataset_id, svc.storage,
                             limits["max_dataset_bytes"], svc.clock())
    if report["status"] == "ready":
        svc.storage.set_dataset(dataset_id, status="ready", rejected_reason=None,
                                at=svc.clock())
    svc.bus.publish("class", "dataset",
                    {"dataset_id": dataset_id, "status": "ready" if report["status"] == "ready"
                     else "failed"})
    return {"dataset_id": dataset_id, "qa": report}


def bundle(svc: ConsoleService, dataset_id: str) -> tuple[bytes, str]:
    """The prepared-dataset tar the agent pulls at dispatch (§9.9)."""
    return prep.bundle_bytes(svc.data_dir, dataset_id)


def frame(svc: ConsoleService, user: dict[str, Any], dataset_id: str,
          episode: str, index: int) -> Path:
    ds = svc.own_dataset(user, dataset_id)
    if ds["status"] != "ready":
        raise RobovlaError("dataset_not_ready", f"dataset is {ds['status']}")
    return prep.frame_path(svc.data_dir, dataset_id, episode, index)
