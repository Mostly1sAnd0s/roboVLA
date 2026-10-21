"""The Runner protocol (design §4, §11): the only hardware boundary.

The console talks to each node's agent (in the podman runner container) over
LAN HTTP with the per-node agent token. ``HttpRunner`` is the single real
implementation; ``FakeRunner`` speaks the same protocol with scripted
metric/event streams so the entire service + SSE path runs on CPU (test
matrix: "FakeRunner speaks the agent protocol with scripted streams").

JobSpec (Appendix B) is the input contract — the agent derives a *fixed argv*
from it (no shell, ever). All methods are sync; the service wraps calls in
``asyncio.to_thread`` (SQLite + agent HTTP are the only real I/O).

Failure contract: unreachable agent -> ``RobovlaError("node_offline")``;
protocol violations -> ``RobovlaError("validation")``; agent-side domain
errors -> the agent's §9.6 code, re-raised unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Runner(Protocol):
    """One node's agent, as seen by the console."""

    # -- node-level probes ---------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Join probe / admin view (§8.2): {ok, image, version, gpu,
        disk_free_gb, model_dir}."""
        ...

    def vram(self) -> dict[str, Any]:
        """Measured VRAM view (§3): {free_gb, total_gb,
        procs: [{pid, name, vram_gb, agent_spawned}]}. ``agent_spawned``
        marks processes the agent launched (orphan-purge scope, §8.2)."""
        ...

    def purge_procs(self, pids: list[int]) -> None:
        """Kill agent-scoped orphan processes by pid (teacher-confirmed,
        §8.2). Host-level processes are never touched."""
        ...

    def agent_log(self, lines: int = 500) -> str:
        """The agent's own service log, last N lines (§8.2/§8.5)."""
        ...

    # -- jobs -----------------------------------------------------------------

    def submit(self, spec: dict[str, Any]) -> None:
        """Launch a job process (fixed argv from the JobSpec). The agent owns
        the process and reaps it (zombie mitigation, §10)."""
        ...

    def poll(self, job_id: str) -> dict[str, Any]:
        """Incremental job view (§9.4 event sources):
        {state: queued|running|finishing|done|failed,
         step, total, loss, lr, grad_norm, vram_mb, vram_util_pct, eta_s,
         metrics: [row, ...]  (new since last poll),
         log: [line, ...]       (new tail lines),
         artifacts: [{kind, path, nbytes}],
         val_accuracy: float|None,
         predict_probe: [float x7]|None   (smoke job, §6),
         error: str|None}"""
        ...

    def kill(self, job_id: str) -> None:
        """SIGTERM the job process (10 s grace -> SIGKILL, §8.3)."""
        ...

    def tail(self, job_id: str, offset: int = 0) -> str:
        """Log tail from ``offset`` (GET /api/runs/{id}/logs?offset=, §9.3)."""
        ...

    def fetch_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        """List completed artifacts: [{kind, path, nbytes}]."""
        ...

    # -- inference slots (§9.5) -------------------------------------------------

    def serve(self, run_id: str, adapter: str) -> str:
        """Load the run's adapter into an 8-bit inference server; returns a
        slot handle owned by the agent."""
        ...

    def release(self, handle: str) -> None:
        """Unload the adapter for a slot handle (VRAM freed, probed)."""
        ...

    def predict(self, handle: str, image_b64: str, prompt: str) -> dict[str, Any]:
        """One robot step: {action_norm: [float x7], latency_ms}. The console
        un-normalizes with the dataset stats (§9.5)."""
        ...
