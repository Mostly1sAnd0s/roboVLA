"""server — the console production bootstrap (fleet.yaml → ConsoleService
→ uvicorn). ``robovla serve`` is the console host's only production entry
(the robovla-console.service unit runs it; design Appendix C).

Every I/O is a seam: the fleet file read (:func:`load_fleet`), the per-node
agent tokens from the host environment (fleet ``token_ref`` names —
plaintext never in fleet.yaml, §9.8), and uvicorn itself (injected in
tests — no sockets in the CPU suite).

The console host has no GPU: the runners built here are HttpRunner only
(console → agent HTTP); the runner image and the node layout are the node
host's job (§10).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .core.config import FleetConfig, parse_fleet
from .core.errors import RobovlaError
from .service import ConsoleService
from .storage import Storage

#: The student UI dir (repo ui/, mounted at /ui by the app factory).
REPO_UI = Path(__file__).resolve().parents[2] / "ui"

#: Default fleet path (relative to the working directory): the operator
#: copies the repo's fleet.yaml (Appendix A) to the console host and wires
#: the placeholder IPs there (wiring time, §14 Q1).
DEFAULT_FLEET = "fleet.yaml"
FLEET_ENV = "ROBOVLA_FLEET"


def load_fleet(path: str | Path) -> FleetConfig:
    """fleet.yaml file → FleetConfig. The file read is the seam; the parse
    is the shared pure core validator (one schema, DRY)."""
    p = Path(path)
    if not p.is_file():
        raise RobovlaError("not_found", f"fleet file not found: {p}")
    return parse_fleet(p.read_text())


def build_service(
    fleet: FleetConfig,
    data_dir: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    runners: dict | None = None,
) -> ConsoleService:
    """A ConsoleService from a validated fleet.

    Default runners: one HttpRunner per node at the fleet's
    ``agent.lan_ip:port`` with the token read from ``env`` under the node's
    ``token_ref`` name (default: ``os.environ``). ``runners`` overrides the
    whole map (tests inject FakeRunner; then no token env is needed).
    """
    from .runner import HttpRunner

    e = os.environ if env is None else env
    if runners is None:
        runners = {}
        for node in fleet.nodes:
            token = (e.get(node.agent.token_ref) or "").strip()
            if not token:
                raise RobovlaError(
                    "validation",
                    f"env {node.agent.token_ref!r} (agent token for node "
                    f"{node.id!r}) is not set on this host — set it in the "
                    f"console host environment (shown once at node add, §9.8)")
            runners[node.id] = HttpRunner(
                f"http://{node.agent.lan_ip}:{node.agent.port}", token)
    data = Path(data_dir) if data_dir else Path(fleet.console.data_dir)
    storage = Storage(data / "state.db")
    return ConsoleService(data, fleet, storage, runners)


def serve(args: Any, *, env: Mapping[str, str] | None = None,
          uvicorn: Any | None = None) -> int:
    """The ``robovla serve`` body: fleet → service → app → uvicorn.

    ``args``: ``fleet`` / ``data_dir`` / ``bind`` / ``port`` (CLI overrides;
    None falls back to the fleet values). ``env``: token environment
    (default: ``os.environ``). ``uvicorn``: the server seam (default: the
    real uvicorn, imported lazily so the CPU suite never starts a socket).
    """
    fleet_path = (getattr(args, "fleet", None)
                  or os.environ.get(FLEET_ENV, "").strip() or DEFAULT_FLEET)
    fleet = load_fleet(fleet_path)
    svc = build_service(fleet, getattr(args, "data_dir", None) or None, env=env)
    from .http.app import create_app

    app = create_app(svc, ui_dir=REPO_UI if REPO_UI.is_dir() else None,
                     fleet_path=fleet_path)
    uvi = uvicorn or _uvicorn()
    uvi.run(app, host=args.bind or fleet.console.bind,
            port=int(args.port or fleet.console.port))
    return 0


def _uvicorn():
    import uvicorn  # lazy: the CPU suite injects a fake and never binds
    return uvicorn


__all__ = ["DEFAULT_FLEET", "FLEET_ENV", "REPO_UI", "build_service",
           "load_fleet", "serve"]
