"""Dataset/artifact file seam (design §5.1, §11: "I/O only at seams").

Layout under the console ``data_dir`` (one definition; the node agent mirrors
it under its ``workdir`` when it pulls a bundle):

    data_dir/
      datasets/<sNN>/<task>/v<N>/        # uploaded + prepared dataset
        chunks/                          # in-flight upload slices (pre-finalize)
        episodes/ep_0001/frames/*.jpg    # 224^2 after prep
        episodes/ep_0001/actions.json    # [{t, vx, vy, wz} @ 10 Hz]
        episodes/ep_0001/episode.json    # {id, task, ...}
        qa.json norm_stats.json split.json
      artifacts/<job_id>/                # adapter / merged / run config
      bundles/<dataset_id>.tar.gz        # ephemeral agent-pull bundles

Rules (DRY with the agent): zip-slip-safe extraction, jpeg/json only, path
sanitization. The *parsing* of episode contents stays faithful to the shared
spec; the *rules* over parsed episodes are core's (episode.py / qa.py).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .core.errors import RobovlaError
from .core.spec import DATASET_RE
from .episode import Action, Episode

#: Allowed file extensions inside an uploaded dataset zip (design §9.8).
ALLOWED_EXTS = frozenset({".jpg", ".jpeg", ".json"})

#: Prepared-frame size (official OpenVLA preprocessing, design §5.2).
FRAME_SIZE = (224, 224)


def _fail(message: str) -> None:
    raise RobovlaError("validation", message)


# --- layout -----------------------------------------------------------------


def sanitize_dataset_id(dataset_id: str) -> str:
    """Validate a dataset id and map it to a safe directory name."""
    if not isinstance(dataset_id, str) or not DATASET_RE.fullmatch(dataset_id):
        _fail(f"dataset_id: {dataset_id!r} must match sNN/<task>/v<N>")
    return dataset_id.replace("/", "__")


def dataset_root(data_dir: str | Path, dataset_id: str) -> Path:
    return Path(data_dir) / "datasets" / sanitize_dataset_id(dataset_id)


def chunks_dir(data_dir: str | Path, dataset_id: str) -> Path:
    return dataset_root(data_dir, dataset_id) / "chunks"


def episodes_dir(data_dir: str | Path, dataset_id: str) -> Path:
    return dataset_root(data_dir, dataset_id) / "episodes"


def artifact_dir(data_dir: str | Path, job_id: str) -> Path:
    if not job_id:
        _fail("job_id must be non-empty")
    return Path(data_dir) / "artifacts" / job_id


def bundles_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "bundles"


# --- chunked upload (raw slices of one zip; whole-file sha per chunk) ---------


def write_chunk(data_dir: str | Path, dataset_id: str, chunk_index: int, data: bytes) -> Path:
    """Persist one upload slice. ``sha256`` (whole zip) is checked at finalize."""
    if chunk_index < 0:
        _fail("chunk_index must be >= 0")
    if not isinstance(data, (bytes, bytearray)):
        _fail("chunk data must be bytes")
    d = chunks_dir(data_dir, dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{chunk_index:05d}.bin"
    p.write_bytes(bytes(data))
    return p


def assemble_chunks(
    data_dir: str | Path, dataset_id: str, total_chunks: int, expected_sha256: str
) -> bytes:
    """Concatenate slices 0..total_chunks-1 in order; verify the whole-zip sha256.

    Missing or out-of-order uploads raise ``validation`` (the API maps this to
    409 ``conflict``-style incompleteness at the route level, §9.2).
    """
    d = chunks_dir(data_dir, dataset_id)
    if total_chunks < 1:
        _fail("total_chunks must be >= 1")
    parts = []
    for i in range(total_chunks):
        p = d / f"{i:05d}.bin"
        if not p.exists():
            _fail(f"chunk {i}/{total_chunks} missing (incomplete upload)")
        parts.append(p.read_bytes())
    blob = b"".join(parts)
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected_sha256.lower():
        _fail(f"sha256 mismatch: expected {expected_sha256}, got {actual}")
    return blob


def clear_chunks(data_dir: str | Path, dataset_id: str) -> None:
    d = chunks_dir(data_dir, dataset_id)
    if d.exists():
        for p in d.iterdir():
            if p.is_file():
                p.unlink()


# --- zip-slip-safe extraction --------------------------------------------------


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    # unix file type lives in the top 4 bits of the mode field (attr >> 16);
    # S_IFLNK = 0o120000 -> (attr >> 28) == 0o12
    return (info.external_attr >> 28) == 0o12


def safe_extract_zip(blob: bytes, dest: Path) -> list[Path]:
    """Extract an uploaded dataset zip into ``dest`` (zip-slip-safe, §9.8).

    Rejects: absolute paths, ``..`` traversal, symlink/hardlink entries, and
    any file extension other than jpeg/json. Returns the written files.
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        _fail("upload is not a valid zip archive")
    written: list[Path] = []
    for info in zf.infolist():
        name = info.filename
        if _is_symlink(info):
            _fail(f"refusing symlink entry {name!r}")
        if "\x00" in name:
            _fail(f"refusing entry with NUL byte: {name!r}")
        target = (dest / name).resolve()
        if dest != target and dest not in target.parents:
            _fail(f"zip-slip: entry {name!r} escapes the dataset directory")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        ext = target.suffix.lower()
        if ext not in ALLOWED_EXTS:
            _fail(f"refusing entry {name!r}: only jpeg/json files are allowed")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            out.write(src.read())
        written.append(target)
    return written


# --- episode tree (seam: bytes -> shared Episode objects) ------------------------


def load_episode(ep_dir: Path) -> Episode:
    """Parse one episode folder into the shared :class:`Episode` (pure core
    types). Frame *reading* stays here; all rules are core's."""
    meta_path = ep_dir / "episode.json"
    acts_path = ep_dir / "actions.json"
    if not meta_path.exists() or not acts_path.exists():
        _fail(f"{ep_dir.name}: missing episode.json or actions.json")
    meta = json.loads(meta_path.read_text())
    raw = json.loads(acts_path.read_text())
    actions = tuple(
        Action(
            t=float(a["t"]), vx=float(a["vx"]), vy=float(a["vy"]), wz=float(a["wz"])
        )
        for a in raw
    )
    frames = ep_dir / "frames"
    frame_count = len(list(frames.glob("*.jpg")) + list(frames.glob("*.jpeg"))) if frames.is_dir() else 0
    return Episode(
        id=str(meta.get("id", ep_dir.name)),
        frame_count=frame_count,
        actions=actions,
        task=str(meta.get("task", "")),
        prompt=str(meta.get("prompt", "")),
        success=meta.get("success"),
        pilot=str(meta.get("pilot", "")),
        notes=str(meta.get("notes", "")),
        created=meta.get("created"),
        start_pose=meta.get("start_pose"),
    )


@dataclass(frozen=True)
class DatasetScan:
    """One prepared dataset on disk: episodes + task + size."""

    episodes: tuple[Episode, ...]
    task: str
    total_bytes: int


def scan_dataset(episodes_root: Path) -> DatasetScan:
    """Walk ``episodes/ep_*`` folders -> shared Episode objects + task + size."""
    if not episodes_root.is_dir():
        _fail("dataset has no episodes directory")
    episodes = [load_episode(d) for d in sorted(episodes_root.iterdir()) if d.is_dir()]
    if not episodes:
        _fail("dataset has no episode folders")
    tasks = {e.task for e in episodes if e.task}
    if len(tasks) > 1:
        _fail(f"dataset mixes tasks: {sorted(tasks)} (one task per dataset, §5.1)")
    total = sum(f.stat().st_size for f in episodes_root.rglob("*") if f.is_file())
    return DatasetScan(tuple(episodes), next(iter(tasks), ""), total)


def resize_frames(episodes_root: Path, size: tuple[int, int] = FRAME_SIZE) -> int:
    """Resize every frame to the official 224^2 preprocessing (§5.2).

    In-place (the prepared dataset is the training source of truth).
    Returns the number of frames processed.
    """
    from PIL import Image  # seam import: heavy dep stays out of core

    n = 0
    for ext in ("*.jpg", "*.jpeg"):
        for p in sorted(episodes_root.rglob(ext)):
            if not p.is_file():
                continue
            with Image.open(p) as im:
                im.convert("RGB").resize(size).save(p.with_suffix(".jpg"), "JPEG", quality=90)
            if p.suffix == ".jpeg":
                p.unlink()
            n += 1
    return n


# --- bundle (agent pull, §9.9) ------------------------------------------------------


def make_bundle(data_dir: str | Path, dataset_id: str, dest: Path) -> tuple[int, str]:
    """Tar.gz of a *ready* dataset (episodes + qa/stats/split) for the agent
    pull. Returns (size_bytes, sha256)."""
    root = dataset_root(data_dir, dataset_id).resolve()
    if not root.is_dir():
        _fail(f"dataset {dataset_id!r} not found")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(root, arcname=sanitize_dataset_id(dataset_id))
    blob = dest.read_bytes()
    return len(blob), hashlib.sha256(blob).hexdigest()
