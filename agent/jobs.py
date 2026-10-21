"""JobRunner — the agent's process supervisor (design §4, §10).

Owns every job process the node runs: submit (validate spec with the *same*
core validator the console uses, §11 DRY -> write spec.json -> derive fixed
argv -> spawn), incremental polling (new metrics.jsonl rows, new log lines,
checkpoint val, smoke probe, artifact manifest), kill (SIGTERM -> grace ->
SIGKILL, §8.3), and the spawn registry that scopes the agent-scoped VRAM
purge (§8.2). The agent reaps its own children (zombie mitigation, §10).

Agent-side lifecycle: ``queued`` (bundle being pulled / starting) ->
``running`` -> ``done | failed | killed``. The console marks a job running
the moment dispatch succeeds; a ``queued`` poll view means "the process is
still starting" and the console simply waits (service tolerates it).

Incremental = since-last-poll per job (byte offsets), terminal views repeat
(SPOF rebuild: a rebuilt console re-polls and converges, §10). All file I/O
is on the node workdir: ``runs/<job_id>/`` per the standard layout.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from robovla.core.budgets import INFER_JOB_TYPES
from robovla.core.errors import RobovlaError
from robovla.core.spec import validate_job_spec
from robovla import proc

from .argv import job_argv

#: puller: (canonical spec dict, workdir) -> None; raises RobovlaError on
#: failure (the job then fails fast and the console reschedules, §10).
Puller = Callable[[dict, Path], None]

_UNDELIVERED = object()


def _read_tail_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Complete lines written since ``offset`` (a partial trailing line is
    held back for the next poll). Offsets reset if the file shrank."""
    if not path.exists():
        return [], offset
    data = path.read_bytes()
    if len(data) < offset:
        offset = 0
    chunk = data[offset:]
    last_nl = chunk.rfind(b"\n")
    if last_nl < 0:
        return [], offset
    complete = chunk[:last_nl + 1]
    return complete.decode(errors="replace").splitlines(), offset + last_nl + 1


@dataclass
class _Job:
    job_id: str
    spec: dict
    dir: Path
    argv: list[str]
    proc: subprocess.Popen | None = None
    state: str = "queued"
    error: str | None = None
    kill_requested: bool = False
    metrics_off: int = 0
    log_off: int = 0
    last_val: Any = _UNDELIVERED
    last_probe: Any = _UNDELIVERED
    seen_artifacts: set[str] = field(default_factory=set)
    terminal_view: dict[str, Any] | None = None


class JobRunner:
    """Supervise job processes under ``workdir/runs/<job_id>``."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        argv_builder: Callable[..., list[str]] = job_argv,
        spawn: Callable[..., subprocess.Popen] = proc.spawn_fixed_argv,
        kill: Callable[..., str] = proc.kill_process,
        grace_s: float = 10.0,
        puller: Puller | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.workdir = Path(workdir)
        self._argv_builder = argv_builder
        self._spawn = spawn
        self._kill = kill
        self.grace_s = grace_s
        self._puller = puller
        self._log = log or (lambda line: None)
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()

    # -- registry (vram probe / purge scope, §8.2) ---------------------------------

    def spawned_pids(self) -> frozenset[int]:
        """Live pids this agent spawned (agent-scoped purge + smi marking)."""
        with self._lock:
            return frozenset(
                j.proc.pid for j in self._jobs.values()
                if j.proc is not None and j.proc.poll() is None)

    def known_job(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    # -- submit ---------------------------------------------------------------------

    def submit(self, spec: dict) -> None:
        """Validate + register + start (bundle pull -> spawn) in a worker."""
        obj = validate_job_spec(spec)  # one schema, one validator (§11 DRY)
        canonical = obj.to_dict()
        if canonical["type"] in INFER_JOB_TYPES:
            # serving IS the job (§9.5): the console never dispatches an infer
            # spec (it calls serve); reject here so the agent's sole entry
            # point cannot be tricked into spawning a "serving" process.
            raise RobovlaError("validation",
                               "infer jobs are served, not dispatched (§9.5)")
        job_id = canonical["job_id"]
        with self._lock:
            if job_id in self._jobs:
                raise RobovlaError("conflict", f"duplicate submit for {job_id}")
        jdir = self.workdir / "runs" / job_id
        argv = self._argv_builder(canonical, jdir)  # infer -> validation (§9.5)
        proc.check_argv(argv)
        job = _Job(job_id=job_id, spec=canonical, dir=jdir, argv=argv)
        with self._lock:
            if job_id in self._jobs:  # re-check (race-safe)
                raise RobovlaError("conflict", f"duplicate submit for {job_id}")
            self._jobs[job_id] = job
        jdir.mkdir(parents=True, exist_ok=True)
        (jdir / "spec.json").write_text(json.dumps(canonical, indent=2))
        (jdir / "argv.json").write_text(json.dumps(argv))
        self._log(f"[jobs] {job_id}: submitted type={canonical['type']} "
                  f"student={canonical['student']}")
        threading.Thread(target=self._start, args=(job,), daemon=True,
                         name=f"robovla-agent-{job_id}").start()

    def _start(self, job: _Job) -> None:
        """Worker: pull the dataset bundle, then spawn the job process.

        Kill-safe: a kill that lands while the bundle is pulling (or mid-
        spawn) wins — nothing is left running unwatched.
        """
        if self._puller is not None:
            try:
                self._puller(job.spec, self.workdir)
                self._log(f"[jobs] {job.job_id}: bundle ready")
            except RobovlaError as e:
                self._finish(job, "failed", f"bundle pull: {e.message}")
                return
        with self._lock:
            if job.kill_requested:
                self._log(f"[jobs] {job.job_id}: killed before start")
                return
        try:
            p = self._spawn(job.argv, cwd=self.workdir, log_path=job.dir / "log")
        except RobovlaError as e:
            self._finish(job, "failed", f"spawn: {e.message}")
            return
        with self._lock:
            if job.kill_requested:  # kill raced the spawn: kill it, stay killed
                pid = p.pid
                job.proc = None
            else:
                job.proc = p
                pid = None
            job.state = "killed" if pid is not None else "running"
        if pid is not None:
            self._kill(pid, self.grace_s)
            self._log(f"[jobs] {job.job_id}: killed before start")
        else:
            self._log(f"[jobs] {job.job_id}: running pid={p.pid}")

    def _finish(self, job: _Job, state: str, error: str | None) -> None:
        """Terminal-state setter; no-op if already terminal (kill race)."""
        with self._lock:
            if job.state in ("done", "failed", "killed"):
                return
            job.state = state
            job.error = error
        self._log(f"[jobs] {job.job_id}: {state}"
                  + (f" ({error})" if error else ""))

    # -- polling ---------------------------------------------------------------------

    def poll(self, job_id: str) -> dict[str, Any]:
        """Incremental job view (the ``Runner.poll`` contract, §9.4 sources)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RobovlaError("not_found", f"unknown job on agent: {job_id}")
            self._reap(job)
            # metrics (JSONL) + log tail, incremental
            metrics: list[dict[str, Any]] = []
            new_metric_lines, job.metrics_off = _read_tail_lines(
                job.dir / "metrics.jsonl", job.metrics_off)
            for line in new_metric_lines:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        metrics.append(row)
                except ValueError:
                    pass  # partial/corrupt row: skip (the job owns its output)
            log_lines, job.log_off = _read_tail_lines(job.dir / "log", job.log_off)
            state = job.state
            error = job.error
            # incremental checkpoint / probe values
            val = self._changed(job, "val.json", "accuracy", job.last_val)
            job.last_val = val if val is not None else job.last_val
            probe = self._changed(job, "probe.json", None, job.last_probe)
            job.last_probe = probe if probe is not None else job.last_probe
            arts = self._new_artifacts(job)
            if state in ("done", "failed", "killed") and job.terminal_view is None:
                job.terminal_view = self._terminal_view(job)

        view = {
            "state": state,
            "step": metrics[-1].get("step") if metrics else None,
            "total": metrics[-1].get("total") if metrics else None,
            "loss": metrics[-1].get("loss") if metrics else None,
            "lr": metrics[-1].get("lr") if metrics else None,
            "grad_norm": metrics[-1].get("grad_norm") if metrics else None,
            "vram_mb": metrics[-1].get("vram_mb") if metrics else None,
            "vram_util_pct": metrics[-1].get("vram_util_pct") if metrics else None,
            "eta_s": metrics[-1].get("eta_s") if metrics else None,
            "metrics": metrics,
            "log": log_lines,
            "artifacts": arts,
            "val_accuracy": val,
            "predict_probe": probe,
            "error": error if state in ("failed",) else None,
        }
        if job.terminal_view is not None and state in ("done", "failed", "killed"):
            # terminal view repeats (SPOF rebuild + console resync, §10)
            view.update({
                "state": state,
                "artifacts": job.terminal_view["artifacts"],
                "val_accuracy": job.terminal_view["val_accuracy"],
                "predict_probe": job.terminal_view["predict_probe"],
                "error": job.terminal_view["error"],
            })
        return view

    def _reap(self, job: _Job) -> None:
        """Non-blocking exit check -> terminal state transition (caller holds
        the lock). A kill always wins the state, whatever the exit code."""
        if job.state != "running" or job.proc is None:
            return
        rc = job.proc.poll()
        if rc is None:
            return
        if job.kill_requested:
            self._finish(job, "killed", None)
        elif rc == 0:
            self._finish(job, "done", None)
        else:
            self._finish(job, "failed", self._failure_reason(job, rc))

    def _failure_reason(self, job: _Job, rc: int) -> str:
        lines, _ = _read_tail_lines(job.dir / "log", 0)
        last = next((ln for ln in reversed(lines) if ln.strip()), None)
        if last:
            return last.strip()[:300]
        return f"job exited with code {rc}"

    def _changed(self, job: _Job, name: str, key: str | None,
                 last: Any) -> Any:
        """Deliver a job-written JSON value only when it changed (val/probe)."""
        p = job.dir / name
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
        except (ValueError, OSError):
            return None
        value = data.get(key) if key is not None else data
        if value is None:
            return None
        if value == last:
            return None
        return value

    def _manifest(self, job: _Job) -> list[dict[str, Any]]:
        """The job's artifact manifest with nbytes (workdir-relative paths)."""
        p = job.dir / "artifacts.json"
        if not p.exists():
            return []
        try:
            entries = json.loads(p.read_text())
        except (ValueError, OSError):
            return []
        out = []
        for e in entries:
            if not isinstance(e, dict) or "kind" not in e or "path" not in e:
                continue
            full = (self.workdir / str(e["path"])).resolve()
            try:
                nbytes = full.stat().st_size
            except OSError:
                nbytes = 0
            out.append({"kind": str(e["kind"]), "path": str(e["path"]),
                        "nbytes": nbytes})
        return out

    def _new_artifacts(self, job: _Job) -> list[dict[str, Any]]:
        fresh = []
        for a in self._manifest(job):
            k = f"{a['kind']}:{a['path']}"
            if k not in job.seen_artifacts:
                job.seen_artifacts.add(k)
                fresh.append(a)
        return fresh

    def _terminal_view(self, job: _Job) -> dict[str, Any]:
        """The full terminal payload (repeated on every subsequent poll —
        SPOF rebuild converges, §10)."""
        return {
            "artifacts": self._manifest(job),
            "val_accuracy": self._file_value(job, "val.json", "accuracy"),
            "predict_probe": self._file_value(job, "probe.json", None),
            "error": job.error,
        }

    @staticmethod
    def _file_value(job: _Job, name: str, key: str | None) -> Any:
        """Final value of a job-written JSON file (no change tracking)."""
        p = job.dir / name
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
        except (ValueError, OSError):
            return None
        value = data.get(key) if key is not None else data
        return value if value is not None else None

    # -- kill / tail / artifacts -------------------------------------------------------

    def jobs_for_pids(self, pids) -> list[str]:
        """Job ids whose live process pid is in ``pids`` (purge -> kill map)."""
        with self._lock:
            wanted = set(pids)
            return [j.job_id for j in self._jobs.values()
                    if j.proc is not None and j.proc.pid in wanted]

    def kill(self, job_id: str) -> None:
        """SIGTERM -> grace -> SIGKILL (§8.3). Idempotent (kill races are
        normal; a terminal job stays terminal)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RobovlaError("not_found", f"unknown job on agent: {job_id}")
            if job.state in ("done", "failed", "killed"):
                return
            job.kill_requested = True
            pid = job.proc.pid if job.proc is not None else None
        if pid is None:
            # still queued (pulling) — nothing is running to reap; the worker
            # checks kill_requested and stays terminal (kill-safe _start)
            self._finish(job, "killed", None)
            return
        self._log(f"[jobs] {job_id}: kill requested (pid={pid})")
        self._kill(pid, self.grace_s)  # blocking up to grace_s (§8.3)
        with self._lock:
            self._reap(job)  # process is gone by now; state -> killed

    def tail(self, job_id: str, offset: int = 0) -> str:
        """Log tail from line ``offset`` (the ``Runner.tail`` contract)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RobovlaError("not_found", f"unknown job on agent: {job_id}")
        lines, _ = _read_tail_lines(job.dir / "log", 0)
        return "\n".join(lines[offset:]) + ("\n" if lines[offset:] else "")

    def fetch_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RobovlaError("not_found", f"unknown job on agent: {job_id}")
            return self._manifest(job)
