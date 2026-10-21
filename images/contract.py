"""The job file contract — the pure half of the in-image job program.

This is the exact contract the agent's supervisor reads out of
``workdir/runs/<job_id>/`` (agent/jobs.py) and the oracle fixture
implements (tests/fixtures/fake_job.py). The GPU engine (engine.py) and the
entrypoint (robovla_job.py) build on these writers so the contract has ONE
definition. Pure: no torch, no network — runs anywhere.

    runs/<job_id>/
      metrics.jsonl    one JSON row per step (the SSE metrics event source)
      val.json         {"accuracy": x} per checkpoint (the SSE val event)
      probe.json       [7 floats] — the predict-action probe (§6)
      artifacts.json   [{"kind", "path" (workdir-relative), "nbytes"}]
      ckpt/            checkpoints (LoRA adapters / merged exports)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from robovla.core.errors import RobovlaError

#: Artifact kind per job type (the agent/console artifact views key on this).
ARTIFACT_KINDS = {"train": "adapter", "validate": "adapter", "merge": "merged"}

#: OpenVLA action space: 7 dims x 256 bins (model card: 256-bin x 7-token).
ACTION_DIMS = 7
ACTION_BINS = 256


def job_dirs(spec_path: str | Path) -> tuple[Path, Path]:
    """(job_dir, workdir) from the spec path: ``<workdir>/runs/<job_id>``."""
    job_dir = Path(spec_path).resolve().parent
    if job_dir.parent.name != "runs":
        raise RobovlaError(
            "internal", f"spec must live under <workdir>/runs/<job_id>: {spec_path}")
    return job_dir, job_dir.parent.parent


def artifact_kind(job_type: str) -> str | None:
    return ARTIFACT_KINDS.get(job_type)


def metrics_row(step: int, total: int, loss: float, lr: float, grad_norm: float,
                vram_mb: int, vram_util_pct: int, eta_s: int) -> dict[str, Any]:
    """One metrics.jsonl row (the exact key set the agent's poll delivers)."""
    return {"step": step, "total": total, "loss": loss, "lr": lr,
            "grad_norm": grad_norm, "vram_mb": vram_mb,
            "vram_util_pct": vram_util_pct, "eta_s": eta_s}


def write_metrics(job_dir: str | Path, row: dict[str, Any]) -> None:
    """Append one metrics row (the agent reads incrementally by byte offset)."""
    with open(job_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def write_val(job_dir: str | Path, accuracy: float) -> None:
    if not math.isfinite(accuracy):
        raise RobovlaError("validation", f"val accuracy must be finite: {accuracy!r}")
    (job_dir / "val.json").write_text(json.dumps({"accuracy": round(accuracy, 4)}))


def _finite_floats(probe: Any, what: str) -> list[float]:
    if not isinstance(probe, Sequence) or isinstance(probe, (str, bytes)) \
            or len(probe) != ACTION_DIMS:
        raise RobovlaError(
            "validation", f"{what} must be a sequence of {ACTION_DIMS} floats")
    out = []
    for v in probe:
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            raise RobovlaError(
                "validation", f"{what} must contain only finite numbers")
        out.append(float(v))
    return out


def write_probe(job_dir: str | Path, probe: Sequence[float]) -> None:
    """The 7-dim predict-action probe (smoke job / selftest, §6)."""
    (job_dir / "probe.json").write_text(json.dumps(_finite_floats(probe, "probe")))


def write_artifacts(job_dir: str | Path, workdir: str | Path,
                    entries: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    """The artifacts manifest: [{kind, path (workdir-relative), nbytes}].

    Every path must live under ``workdir`` (the manifest is shipped to the
    console and re-resolved there — an absolute or escaping path is a bug).
    """
    workdir = Path(workdir).resolve()
    manifest = []
    for kind, path in entries:
        p = Path(path).resolve()
        if workdir != p and workdir not in p.parents:
            raise RobovlaError(
                "validation", f"artifact {path} escapes the workdir")
        if p.is_file():
            nbytes = p.stat().st_size
        elif p.is_dir():  # e.g. the ~15 GB merged export (§6)
            nbytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        else:
            raise RobovlaError(
                "validation", f"artifact {path} is not a file or directory")
        manifest.append({"kind": kind,
                         "path": p.relative_to(workdir).as_posix(),
                         "nbytes": nbytes})
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    (job_dir / "artifacts.json").write_text(json.dumps(manifest))
    return manifest
