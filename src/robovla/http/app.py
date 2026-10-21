"""FastAPI app factory (design §9: router wiring, error shape, rate limits).

``create_app`` wires the service into the app; every route hangs off the
service (one seam, §11). Errors surface as the stable §9.6 shape
``{"error": {"code", "message"}}`` with the status from ``STATUS_BY_CODE`` —
one handler, every route.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..core.errors import STATUS_BY_CODE, RobovlaError
from ..service import ConsoleService
from . import admin, admin_users, auth, datasets, health, infer, runs, stream, tickets

log = logging.getLogger("robovla")


def _error_response(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=STATUS_BY_CODE.get(code, 500),
                         content={"error": {"code": code, "message": message}})


def create_app(svc: ConsoleService, *, ui_dir: str | Path | None = None,
               fleet_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="roboVLA console", version="0.1.0")
    app.state.svc = svc
    app.state.fleet_path = Path(fleet_path) if fleet_path else None
    svc.fleet_path = app.state.fleet_path

    @app.on_event("startup")
    async def _startup() -> None:
        await svc.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await svc.stop()

    @app.exception_handler(RobovlaError)
    async def _robovla_error(request: Request, e: RobovlaError) -> JSONResponse:
        return _error_response(e.code, e.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, e: RequestValidationError) -> JSONResponse:
        return _error_response("validation", f"malformed request: {e.errors()[:1]}")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, e: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error_response("internal", "internal error (see service log)")

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(datasets.router)
    app.include_router(runs.router)
    app.include_router(tickets.router)
    app.include_router(stream.router)
    app.include_router(infer.router)
    app.include_router(infer.robot_router)
    app.include_router(infer.agent_router)
    app.include_router(admin.router)
    app.include_router(admin_users.router)

    if ui_dir is not None and Path(ui_dir).is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    return app
