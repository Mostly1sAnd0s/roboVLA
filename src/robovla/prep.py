"""Dataset prep (design §5.2): upload finalize -> validate -> QA gate ->
norm stats -> held-out split -> 224^2 frames -> ready|failed.

The seam that composes fs (I/O) + core (rules: episode/qa/norm_stats/split).
Runs on CPU in a worker thread (Pillow resize); the service owns the clock
and the dataset status transitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import fs
from .core.errors import RobovlaError
from .core.norm_stats import compute_norm_stats
from .core.qa import qa_dataset
from .core.split import HELD_OUT_FRACTION, split
from .episode import to_action_7d, validate_episode
from .storage import Storage


def _dataset_task(dataset_id: str) -> str:
    return dataset_id.split("/")[1]


def _student(dataset_id: str) -> str:
    return dataset_id.split("/")[0]


def prepare_dataset(
    data_dir: Path,
    dataset_id: str,
    episodes_root: Path,
    max_bytes: int,
    storage: Storage,
    at: str,
) -> dict[str, Any]:
    """Run the prep pipeline over an extracted dataset; persist + status.

    Returns the QA report dict (the gate, §5.2). Dataset status becomes
    ``ready`` or ``failed``. The held-out split is stable per student+task
    (core.split — one definition, DRY).
    """
    scan = fs.scan_dataset(episodes_root)
    reports = [validate_episode(ep) for ep in scan.episodes]
    qa = qa_dataset(reports, scan.total_bytes, max_bytes)

    payload: dict[str, Any] = {"qa": qa.to_dict()}
    if qa.status == "ready":
        # norm stats over the course 7-dim action convention + held-out split
        actions = [to_action_7d(a) for ep in scan.episodes for a in ep.actions]
        stats = compute_norm_stats(actions)
        sp = split(
            [ep.id for ep in scan.episodes], _student(dataset_id), _dataset_task(dataset_id),
            HELD_OUT_FRACTION,
        )
        payload["norm_stats"] = stats.to_dict()
        payload["split"] = {
            "train": list(sp.train), "held_out": list(sp.held_out),
            "fraction": HELD_OUT_FRACTION,
        }
        fs.resize_frames(episodes_root)
        _write(episodes_root.parent, "norm_stats.json", payload["norm_stats"])
        _write(episodes_root.parent, "split.json", payload["split"])
    _write(episodes_root.parent, "qa.json", payload["qa"])
    storage.set_dataset(dataset_id, status=qa.status, at=at,
                        qa=payload["qa"], norm_stats=payload.get("norm_stats"),
                        split=payload.get("split"))
    return payload["qa"]


def finalize_upload(
    data_dir: Path,
    dataset_id: str,
    total_chunks: int,
    sha256: str,
    storage: Storage,
    max_bytes: int,
    at: str,
) -> dict[str, Any]:
    """Assemble + verify the uploaded zip, extract it, and run prep."""
    blob = fs.assemble_chunks(data_dir, dataset_id, total_chunks, sha256)
    root = fs.dataset_root(data_dir, dataset_id)
    fs.safe_extract_zip(blob, root)  # zip-slip-safe, jpeg/json only
    fs.clear_chunks(data_dir, dataset_id)
    storage.set_dataset(dataset_id, status="preparing", at=at)
    try:
        report = prepare_dataset(data_dir, dataset_id, root / "episodes", max_bytes, storage, at)
    except RobovlaError as e:
        storage.set_dataset(dataset_id, status="failed", rejected_reason=e.message, at=at)
        raise
    except Exception as e:  # corrupt/invalid content is a failed dataset, not a 500
        storage.set_dataset(
            dataset_id, status="failed", rejected_reason=f"prep error: {e.__class__.__name__}", at=at
        )
        raise RobovlaError("internal", f"dataset prep failed for {dataset_id}") from e
    if report["status"] == "ready":
        storage.set_dataset(dataset_id, status="ready", at=at)
    return report


def revalidate(data_dir: Path, dataset_id: str, storage: Storage, max_bytes: int, at: str) -> dict[str, Any]:
    """Re-run prep after corrections (same dataset id; the held-out split
    stays stable for unchanged content, §5.2 / §9.2)."""
    root = fs.dataset_root(data_dir, dataset_id)
    if not (root / "episodes").is_dir():
        raise RobovlaError("not_found", f"dataset {dataset_id!r} has no prepared episodes")
    report = prepare_dataset(data_dir, dataset_id, root / "episodes", max_bytes, storage, at)
    if report["status"] == "ready":
        storage.set_dataset(dataset_id, status="ready", rejected_reason=None, at=at)
    return report


def bundle_bytes(data_dir: Path, dataset_id: str) -> tuple[bytes, str]:
    """The prepared-dataset tar the agent pulls at dispatch (§9.9)."""
    dest = fs.bundles_dir(data_dir) / (fs.sanitize_dataset_id(dataset_id) + ".tar.gz")
    _, sha = fs.make_bundle(data_dir, dataset_id, dest)
    return dest.read_bytes(), sha


def frame_path(data_dir: Path, dataset_id: str, episode: str, index: int) -> Path:
    """Browser preview path for one frame (ownership checked by the router)."""
    p = fs.dataset_root(data_dir, dataset_id) / "episodes" / episode / "frames"
    files = sorted(p.glob("*.jpg")) if p.is_dir() else []
    if not (0 <= index < len(files)):
        raise RobovlaError("not_found", f"frame {index} not found in {episode}")
    return files[index]


def _write(base: Path, name: str, obj: Any) -> None:
    (base / name).write_text(json.dumps(obj, indent=1, sort_keys=True))
