"""FakeRunner — the agent protocol on CPU (design §11 test matrix).

Scripted per job so the *entire* service path (submit -> queue admission ->
SSE state/metrics/val/log events -> artifacts -> smoke predict probe) runs in
tests without a GPU. Each ``poll`` advances the script by one step: one
metrics row + one log line; the final poll delivers the terminal state,
artifacts, ``val_accuracy`` and (for smoke) the 7-dim ``predict_probe``.

Deterministic: no timers, no wall clock — the service's tick loop drives it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.errors import RobovlaError
from .interface import Runner

#: What a FakeRunner job finishes with.
DONE, FAILED = "done", "failed"


@dataclass
class _Script:
    """One scripted job: ``steps`` polls of metrics, then a terminal poll."""

    steps: int = 10
    final_state: str = DONE
    error: str | None = None
    val_accuracy: float | None = None
    predict_probe: list[float] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    cursor: int = 0
    submitted: bool = False

    @property
    def total(self) -> int:
        return self.steps


def _default_metrics(step: int, total: int) -> dict[str, Any]:
    # scripted learning curve: loss decays, accuracy climbs (display-only)
    frac = step / max(total, 1)
    return {
        "step": step,
        "total": total,
        "loss": round(2.0 * (1.0 - frac) + 0.05, 4),
        "lr": 5e-4,
        "grad_norm": round(1.5 - frac, 4),
        "vram_mb": 15000,
        "vram_util_pct": 78,
        "eta_s": int(300 * (1.0 - frac)),
    }


class FakeRunner:
    """A scripted node agent (implements :class:`Runner`)."""

    def __init__(self, free_gb: float = 32.0, total_gb: float = 32.0, online: bool = True) -> None:
        self.free_gb = free_gb
        self.total_gb = total_gb
        self.online = online
        self.procs: list[dict[str, Any]] = []
        self.scripts: dict[str, _Script] = {}
        self.submitted_specs: dict[str, dict[str, Any]] = {}
        self.killed: set[str] = set()
        self.handle_to_run: dict[str, str] = {}
        self.run_handles: dict[str, list[str]] = {}
        self.predict_calls: list[dict[str, Any]] = []
        self._next_handle = 0
        self.fail_submit = False  # next submit() raises node_offline

    # -- script control (tests) ------------------------------------------------

    def script(self, job_id: str, steps: int = 10, final_state: str = DONE, **kw: Any) -> _Script:
        s = _Script(steps=steps, final_state=final_state, **kw)
        self.scripts[job_id] = s
        return s

    def set_vram(self, free_gb: float, procs: list[dict[str, Any]] | None = None) -> None:
        self.free_gb = free_gb
        self.procs = procs or []

    # -- node probes -------------------------------------------------------------

    def _check_online(self) -> None:
        if not self.online:
            raise RobovlaError("node_offline", "agent unreachable")

    def health(self) -> dict[str, Any]:
        self._check_online()
        return {
            "ok": True,
            "image": "robovla-runner-cuda:2026.10",
            "version": "0.1.0",
            "gpu": "RTX 5090 32GB",
            "disk_free_gb": 900.0,
            "model_dir": "models/openvla-7b",
        }

    def vram(self) -> dict[str, Any]:
        self._check_online()
        return {"free_gb": self.free_gb, "total_gb": self.total_gb, "procs": list(self.procs)}

    def purge_procs(self, pids: list[int]) -> None:
        self._check_online()
        known = {p["pid"] for p in self.procs if p.get("agent_spawned")}
        bad = [p for p in pids if p not in known]
        if bad:
            raise RobovlaError("validation", f"not agent-scoped pids: {bad}")
        freed = sum(p.get("vram_gb", 0) for p in self.procs if p["pid"] in pids)
        self.purged_pids = list(pids)
        self.procs = [p for p in self.procs if p["pid"] not in pids]
        self.free_gb = min(self.total_gb, self.free_gb + freed)

    def agent_log(self, lines: int = 500) -> str:
        self._check_online()
        self.agent_log_lines = lines
        return "\n".join(f"[agent] line {i}" for i in range(max(1, min(lines, 50)))) + "\n"

    # -- jobs ------------------------------------------------------------------------

    def submit(self, spec: dict[str, Any]) -> None:
        self._check_online()
        if self.fail_submit:
            self.fail_submit = False
            raise RobovlaError("node_offline", "dispatch failed (scripted)")
        job_id = spec["job_id"]
        if job_id in self.submitted_specs:
            raise RobovlaError("conflict", f"duplicate submit for {job_id}")
        self.submitted_specs[job_id] = spec
        self.scripts.setdefault(job_id, _Script())
        self.scripts[job_id].submitted = True

    def poll(self, job_id: str) -> dict[str, Any]:
        self._check_online()
        s = self.scripts.get(job_id)
        if s is None or not s.submitted:
            raise RobovlaError("not_found", f"unknown job on agent: {job_id}")
        if s.cursor < s.total:
            s.cursor += 1
            return {
                "state": "running",
                "step": s.cursor,
                "total": s.total,
                "loss": None,
                "lr": None,
                "grad_norm": None,
                "vram_mb": None,
                "vram_util_pct": None,
                "eta_s": None,
                "metrics": [_default_metrics(s.cursor, s.total)],
                "log": [f"[step {s.cursor}/{s.total}] loss={_default_metrics(s.cursor, s.total)['loss']}"],
                "artifacts": [],
                "val_accuracy": None,
                "predict_probe": None,
                "error": None,
            }
        # terminal poll (delivered once; further polls repeat it)
        state = "killed" if job_id in self.killed else s.final_state
        return {
            "state": state,
            "step": s.total,
            "total": s.total,
            "loss": None,
            "lr": None,
            "grad_norm": None,
            "vram_mb": None,
            "vram_util_pct": None,
            "eta_s": 0,
            "metrics": [],
            "log": [f"[done] job {job_id} finished ({state})"],
            "artifacts": s.artifacts,
            "val_accuracy": s.val_accuracy,
            "predict_probe": s.predict_probe if state == DONE else None,
            "error": s.error if state == FAILED else None,
        }

    def kill(self, job_id: str) -> None:
        self._check_online()
        if job_id not in self.submitted_specs:
            raise RobovlaError("not_found", f"unknown job on agent: {job_id}")
        self.killed.add(job_id)
        self.scripts[job_id].cursor = self.scripts[job_id].total  # next poll is terminal

    def tail(self, job_id: str, offset: int = 0) -> str:
        self._check_online()
        if job_id not in self.submitted_specs:
            raise RobovlaError("not_found", f"unknown job on agent: {job_id}")
        s = self.scripts[job_id]
        lines = [f"[step {i}/{s.total}] ..." for i in range(1, max(s.cursor, 1) + 1)]
        return "\n".join(lines[offset:]) + "\n"

    def fetch_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        self._check_online()
        s = self.scripts.get(job_id)
        if s is None:
            raise RobovlaError("not_found", f"unknown job on agent: {job_id}")
        return list(s.artifacts)

    # -- inference slots ------------------------------------------------------------------

    def serve(self, run_id: str, adapter: str) -> str:
        self._check_online()
        self._next_handle += 1
        handle = f"slot-{self._next_handle}"
        self.handle_to_run[handle] = run_id
        self.run_handles.setdefault(run_id, []).append(handle)
        return handle

    def release(self, handle: str) -> None:
        self._check_online()
        if handle not in self.handle_to_run:
            raise RobovlaError("not_found", f"unknown slot handle: {handle}")
        run_id = self.handle_to_run.pop(handle)
        self.run_handles[run_id].remove(handle)

    def predict(self, handle: str, image_b64: str, prompt: str) -> dict[str, Any]:
        self._check_online()
        if handle not in self.handle_to_run:
            raise RobovlaError("not_found", f"no adapter loaded for handle {handle}")
        self.predict_calls.append({"handle": handle, "prompt": prompt, "image_b64": image_b64})
        return {"action_norm": [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], "latency_ms": 180.0}


__all__ = ["FakeRunner", "Runner", "DONE", "FAILED"]
