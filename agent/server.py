"""The runner agent's HTTP API — the node's face to the console (design §4).

One FastAPI app on the node's LAN port (default 8601, console-only traffic,
§9.8). Every route is authenticated with the per-node pre-shared agent
token (Bearer; constant-time compare) by ONE dependency — the agent-side
mirror of the console's single RBAC dep (§9.8). Errors render the shared
§9.6 shape ``{"error": {"code", "message"}}`` with the shared status map —
the console's ``HttpRunner`` expects exactly this.

Routes (mirrored by ``runner/http.py``):
  GET  /health            join probe facts (§8.2)
  GET  /vram              measured VRAM + holding procs (§3)
  POST /vram/purge        agent-scoped orphan kill (§8.2)
  GET  /log               the agent's own service log, last N lines
  POST /jobs              submit a validated JobSpec (fixed argv, no shell)
  GET  /jobs/{id}         incremental poll view (§9.4 sources)
  POST /jobs/{id}/kill    SIGTERM -> grace -> SIGKILL (§8.3)
  GET  /jobs/{id}/logs    log tail ?offset=
  GET  /jobs/{id}/artifacts
  POST /serve             load an adapter into a slot (§9.5)
  DELETE /serve/{handle}  unload
  POST /predict           one robot step (proxied by the console)

``build_app`` takes all collaborators as parameters (tests inject fakes at
every seam); ``make_agent_app`` wires the production ones from env config.
"""

from __future__ import annotations

import hmac
import shutil
import threading
from collections import deque
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from robovla.core.errors import STATUS_BY_CODE, RobovlaError

from .argv import job_argv
from .config import AgentConfig
from .jobs import JobRunner
from .pull import pull_bundle
from .serve import SlotServing, make_default_model_factory
from .vram import VramProbe


class LogRing:
    """The agent's service log: last N lines in memory + optional file
    (§8.2/§8.5 — ``GET /api/admin/nodes/{id}/logs`` tails this)."""

    def __init__(self, path: str = "", maxlen: int = 500) -> None:
        self._lines: deque[str] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._file = None
        if path:
            self._file = open(path, "a", buffering=1)

    def log(self, line: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") \
            + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
        entry = f"{stamp} {line}"
        with self._lock:
            self._lines.append(entry)
            if self._file is not None:
                self._file.write(entry + "\n")

    def tail(self, lines: int = 500) -> str:
        with self._lock:
            recent = list(self._lines)[-max(1, min(lines, 500)):]
        return "\n".join(recent) + ("\n" if recent else "")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise RobovlaError("auth", "missing Bearer agent token")
    return auth[7:].strip()


def make_auth(cfg: AgentConfig):
    """The ONE agent-side auth dependency (§9.8): per-node token, checked
    constant-time. Never per-route."""
    def authorize(request: Request, token: str = Depends(_bearer_token)) -> None:
        if not hmac.compare_digest(token, cfg.token):
            raise RobovlaError("auth", "invalid agent token")
    return authorize


def build_app(
    cfg: AgentConfig,
    *,
    jobs: JobRunner,
    probe: VramProbe,
    serving: SlotServing,
    log: LogRing,
) -> FastAPI:
    app = FastAPI(title="robovla-agent", version=cfg.version)
    app.state.cfg = cfg
    app.state.jobs = jobs
    app.state.probe = probe
    app.state.serving = serving
    app.state.log = log
    authorize = make_auth(cfg)

    @app.exception_handler(RobovlaError)
    async def _error(request: Request, exc: RobovlaError) -> JSONResponse:
        return JSONResponse(status_code=STATUS_BY_CODE[exc.code],
                            content=exc.to_payload())

    # -- node probes -----------------------------------------------------------

    @app.get("/health", dependencies=[Depends(authorize)])
    def health() -> dict[str, Any]:
        """Join-probe facts (§8.2): the console decides pass/fail (image
        match, disk >= 50 GB, model dir present) — the agent reports
        measured values, never opinions."""
        disk_free = 0.0
        try:
            disk_free = shutil.disk_usage(cfg.workdir).free / (1024 ** 3)
        except OSError:
            pass
        gpu: str | None = None
        ok = True
        try:
            v = probe.view()
            gpu = f"{v['name']} {v['total_gb']:g}GB" if v.get("name") else None
        except RobovlaError:
            gpu, ok = None, False
        model_dir = ""
        if Path(cfg.model_dir).is_dir():
            try:
                model_dir = Path(cfg.model_dir).relative_to(cfg.workdir).as_posix()
            except ValueError:
                model_dir = cfg.model_dir
        return {"ok": ok, "image": cfg.image_tag, "version": cfg.version,
                "gpu": gpu, "disk_free_gb": round(disk_free, 1),
                "model_dir": model_dir}

    @app.get("/vram", dependencies=[Depends(authorize)])
    def vram() -> dict[str, Any]:
        return probe.view(jobs.spawned_pids())

    @app.post("/vram/purge", dependencies=[Depends(authorize)])
    async def purge(request: Request) -> dict[str, Any]:
        body = await request.json()
        pids = body.get("pids", [])
        if not isinstance(pids, list) or not all(isinstance(p, int) for p in pids):
            raise RobovlaError("validation", "pids must be a list of integers")
        killed = probe.purge(pids, jobs.spawned_pids(), grace_s=cfg.kill_grace_s)
        # a teacher-confirmed kill of a LIVE job process is a kill, not a
        # crash: mark those jobs killed so the console sees the true cause
        for job_id in jobs.jobs_for_pids(pids):
            jobs.kill(job_id)
        log.log(f"[vram] purged {killed}/{len(pids)} agent-scoped pids "
                f"{pids} (teacher-confirmed)")
        return {"ok": True, "killed": killed}

    @app.get("/log", dependencies=[Depends(authorize)])
    def agent_log(lines: int = 500) -> dict[str, Any]:
        return {"log": log.tail(lines)}

    # -- jobs -------------------------------------------------------------------

    @app.post("/jobs", dependencies=[Depends(authorize)])
    async def submit(request: Request) -> dict[str, Any]:
        body = await request.json()
        spec = body.get("spec")
        if not isinstance(spec, dict):
            raise RobovlaError("validation", "spec (JobSpec object) is required")
        jobs.submit(spec)  # validates with the shared core schema (DRY)
        return {"ok": True}

    @app.get("/jobs/{job_id}", dependencies=[Depends(authorize)])
    def poll(job_id: str) -> dict[str, Any]:
        return jobs.poll(job_id)

    @app.post("/jobs/{job_id}/kill", dependencies=[Depends(authorize)])
    def kill(job_id: str) -> dict[str, Any]:
        jobs.kill(job_id)
        return {"ok": True}

    @app.get("/jobs/{job_id}/logs", dependencies=[Depends(authorize)])
    def logs(job_id: str, offset: int = 0) -> dict[str, Any]:
        return {"log": jobs.tail(job_id, offset)}

    @app.get("/jobs/{job_id}/artifacts", dependencies=[Depends(authorize)])
    def artifacts(job_id: str) -> dict[str, Any]:
        return {"artifacts": jobs.fetch_artifacts(job_id)}

    # -- inference slots (§9.5) ----------------------------------------------------

    @app.post("/serve", dependencies=[Depends(authorize)])
    async def serve(request: Request) -> dict[str, Any]:
        body = await request.json()
        handle = serving.serve(str(body.get("run_id", "")),
                               str(body.get("adapter", "")))
        return {"handle": handle}

    @app.delete("/serve/{handle}", dependencies=[Depends(authorize)])
    def release(handle: str) -> dict[str, Any]:
        serving.release(handle)
        return {"ok": True}

    @app.post("/predict", dependencies=[Depends(authorize)])
    async def predict(request: Request) -> dict[str, Any]:
        body = await request.json()
        return serving.predict(str(body.get("handle", "")),
                               str(body.get("image", "")),
                               str(body.get("prompt", "")))

    return app


def make_puller(cfg: AgentConfig, log: LogRing, *, client=None):
    """The dispatch-time bundle pull (§5.1/§9.9) — None when no console URL
    is configured (self-contained dev lab: jobs run on local data)."""
    if not cfg.console_url:
        return None

    def puller(spec: dict, workdir: Path) -> None:
        ds = spec["dataset"]
        log.log(f"[pull] {spec['job_id']}: dataset {ds} from {cfg.console_url}")
        dest = pull_bundle(cfg.console_url, cfg.token, ds, workdir, client=client)
        log.log(f"[pull] {spec['job_id']}: dataset ready at {dest.name}")

    return puller


def make_agent_app(cfg: AgentConfig) -> FastAPI:
    """Production wiring (the entrypoint the image runs)."""
    log = LogRing(cfg.log_path)
    jobs = JobRunner(cfg.workdir, grace_s=cfg.kill_grace_s,
                     # the image default (/opt/robovla/bin) when the env is
                     # unset; ROBOVLA_JOB_BIN overrides (direct-venv nodes)
                     argv_builder=partial(job_argv, bin_path=cfg.job_bin),
                     puller=make_puller(cfg, log), log=log.log)
    probe = VramProbe(cfg.gpu)
    serving = SlotServing(make_default_model_factory(cfg.workdir, log=log.log))
    log.log(f"[agent] robovla-agent {cfg.version} up (gpu={cfg.gpu}, "
            f"image={cfg.image_tag}, workdir={cfg.workdir})")
    return build_app(cfg, jobs=jobs, probe=probe, serving=serving, log=log)


def main() -> None:
    from .config import from_env
    cfg = from_env()
    import uvicorn
    uvicorn.run(make_agent_app(cfg), host=cfg.bind, port=cfg.port,
                log_level="warning")


__all__ = ["LogRing", "build_app", "make_agent_app", "make_auth",
           "make_puller", "main"]
