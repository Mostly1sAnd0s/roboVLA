"""``robovla cli`` — operational entrypoint (design line 488/503).

Commands::

    robovla cli backup         --data-dir X --dest Y
    robovla cli restore        --bundle X --data-dir Y [--force]
    robovla cli gpu_selftest   --node ID --dataset DS
                               [--agent-url U | --config fleet.yaml]
                               [--token T | env ROBOVLA_AGENT_TOKEN]
                               [--job-id J] [--timeout S]
    robovla cli model_download --url U --dest DIR [--name N]
    robovla cli node bootstrap --workdir DIR [--model-dir DIR]

House rules apply: shared seams only (backup, the Runner protocol, an
injectable fetcher — no new I/O paths), §9.6 error codes, no shell (argv
only), exit 0 = success / 1 = domain error / 2 = usage error.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .backup import make_backup_bundle, now_rfc3339, restore_backup
from .core.budgets import estimate_job_gb
from .core.config import parse_fleet
from .core.errors import RobovlaError
from .core.spec import build_job_spec
from .runner.http import HttpRunner
from .storage import Storage

#: The system student: provisioning jobs (selftests at node join, §10) are not
#: class work. s01-s12 are the twelve student seats.
SYSTEM_STUDENT = "s00"

#: Admission budget for a selftest when no fleet config is given: the base
#: train budget of the default 32 GB node (Appendix A pai1).
DEFAULT_SELFTEST_BUDGET_GB = 18.0

TOKEN_ENV = "ROBOVLA_AGENT_TOKEN"


# -- backup / restore -------------------------------------------------------------


def cmd_backup(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    db = data_dir / "state.db"
    store = Storage(db) if db.exists() else None
    try:
        info = make_backup_bundle(
            data_dir, args.dest,
            record=(lambda i: store.add_backup(
                now_rfc3339(), i["path"], i["size_bytes"],
                i["sha256"], i["status"])) if store else None)
    except (RobovlaError, OSError, sqlite3.Error) as e:
        return _err(f"backup failed: {e}")
    finally:
        if store is not None:
            store.close()
    print(f"backup: {info['path']} ({info['size_bytes']} B, "
          f"sha256 {info['sha256'][:16]})")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    try:
        out = restore_backup(args.bundle, args.data_dir, force=args.force)
    except RobovlaError as e:
        return _err(f"{e.code}: {e.message}")
    print(f"restored {out['files']} file(s), db={'yes' if out['db'] else 'no'} "
          f"-> {args.data_dir}")
    return 0


# -- gpu_selftest (provisioning at node join, §10) -----------------------------------


def _ensure_scheme(url: str) -> str:
    if url.startswith(("http://", "https://")):
        return url
    return f"http://{url}"


def _agent_ref(url: str) -> str:
    """The JobSpec ``node.agent`` form (``ip:port``) from a base URL."""
    return url.split("://", 1)[-1].split("/", 1)[0]


def _resolve_agent(args: argparse.Namespace) -> tuple[str, str, float, int]:
    """(base_url, agent ip:port, budget_gb, batch) for the selftest."""
    if args.agent_url:
        url = _ensure_scheme(args.agent_url)
        return url, _agent_ref(url), DEFAULT_SELFTEST_BUDGET_GB, 4
    if not args.config:
        raise RobovlaError(
            "validation", "no agent endpoint: pass --agent-url or --config fleet.yaml")
    try:
        fleet = parse_fleet(Path(args.config).read_text())
    except RobovlaError as e:
        raise RobovlaError("validation", f"fleet config: {e.message}") from e
    node = next((n for n in fleet.nodes if n.id == args.node), None)
    if node is None:
        raise RobovlaError("not_found", f"node {args.node!r} not in fleet config")
    mb = max(node.presets.batch)
    url = f"http://{node.agent.lan_ip}:{node.agent.port}"
    budget = estimate_job_gb(node.gpu.budgets.train_gb, node.gpu.budgets.infer_gb,
                             node.gpu.quant, "gpu_selftest", mb, mb)
    return url, _agent_ref(url), budget, mb


def run_selftest(
    *,
    runner,
    node_id: str,
    dataset: str,
    agent: str,
    job_id: str | None = None,
    budget_gb: float = DEFAULT_SELFTEST_BUDGET_GB,
    batch: int = 4,
    timeout_s: float = 1800.0,
    poll_s: float = 1.0,
    out: Callable[[str], None] | None = None,
    err: Callable[[str], None] | None = None,
) -> int:
    """Submit a ``gpu_selftest`` JobSpec to a node's agent; block until terminal.

    ``runner`` is the :class:`~robovla.runner.interface.Runner` seam (the CLI
    wires ``HttpRunner``; tests wire ``FakeRunner``). Returns 0 on ``done``,
    1 on ``failed``/offline/timeout. Protocol violations (e.g. a duplicate
    ``job_id``) propagate as ``RobovlaError`` — the teacher must see them.
    """
    out = out or (lambda m: print(m))
    err = err or (lambda m: print(m, file=sys.stderr))
    if job_id is None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        job_id = f"j_{day}_0001"
    spec = build_job_spec(
        job_id=job_id, student=SYSTEM_STUDENT, job_type="gpu_selftest",
        dataset=dataset, node_id=node_id, node_agent=agent,
        budget_gb=budget_gb, batch=batch)
    out(f"gpu_selftest {job_id} -> {node_id} ({agent})")
    try:
        runner.submit(spec.to_dict())
    except RobovlaError as e:
        if e.code == "node_offline":
            err(f"{job_id}: {e.code} — {e.message}")
            return 1
        raise
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            view = runner.poll(job_id)
        except RobovlaError as e:
            if e.code == "node_offline":
                err(f"{job_id}: {e.code} — {e.message}")
                return 1
            raise
        state = view["state"]
        if state == "done":
            extra = (f" val={view['val_accuracy']}"
                     if view.get("val_accuracy") is not None else "")
            out(f"{job_id}: done{extra}")
            return 0
        if state == "failed":
            err(f"{job_id}: failed — {view.get('error') or 'no error detail'}")
            return 1
        if time.monotonic() >= deadline:
            err(f"{job_id}: still {state} after {int(timeout_s)} s — check the "
                f"agent log (GET /jobs/{job_id}/logs on {agent})")
            return 1
        time.sleep(poll_s)


def cmd_gpu_selftest(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get(TOKEN_ENV, "")
    if not token:
        return _err(f"missing agent token: pass --token or set {TOKEN_ENV}")
    try:
        url, agent_ref, budget, batch = _resolve_agent(args)
    except RobovlaError as e:
        return _err(f"{e.code}: {e.message}")
    runner = HttpRunner(url, token)
    try:
        return run_selftest(runner=runner, node_id=args.node, dataset=args.dataset,
                            agent=agent_ref, job_id=args.job_id,
                            budget_gb=budget, batch=batch, timeout_s=args.timeout)
    except RobovlaError as e:
        return _err(f"{e.code}: {e.message}")
    finally:
        runner.close()


# -- model_download ----------------------------------------------------------------


def _http_fetch(url: str, path: Path) -> int:
    import httpx

    n = 0
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
            if r.status_code >= 400:
                raise RobovlaError(
                    "node_offline", f"model download HTTP {r.status_code} for {url}")
            with open(path, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
    except httpx.HTTPError as e:
        path.unlink(missing_ok=True)
        raise RobovlaError(
            "node_offline", f"model download failed: {e.__class__.__name__}") from e
    except RobovlaError:
        path.unlink(missing_ok=True)
        raise
    return n


def download_model(
    url: str,
    dest: str | Path,
    *,
    name: str | None = None,
    fetch: Callable[[str, Path], int] | None = None,
) -> dict[str, Any]:
    """Download base-model weights into ``dest`` (the node's model volume).

    ``fetch(url, path) -> bytes_written`` is the I/O seam (default: streaming
    httpx GET; tests inject a fake). Refuses to overwrite an existing file
    (``conflict``) — the teacher removes it deliberately. Returns
    ``{path, nbytes}``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    name = name or (url.rstrip("/").rsplit("/", 1)[-1] or "model.bin")
    path = dest / name
    if path.exists():
        raise RobovlaError("conflict", f"{path} already exists — remove it first")
    n = (fetch or _http_fetch)(url, path)
    return {"path": str(path), "nbytes": n}


def cmd_model_download(args: argparse.Namespace) -> int:
    try:
        out = download_model(args.url, args.dest, name=args.name)
    except RobovlaError as e:
        return _err(f"{e.code}: {e.message}")
    print(f"downloaded {out['nbytes']} B -> {out['path']}")
    return 0


# -- node bootstrap ------------------------------------------------------------------


def bootstrap_node(workdir: str | Path, model_dir: str | Path | None = None) -> dict[str, str]:
    """Create (idempotently) the node-host data layout (design §10).

    ``workdir/{datasets,runs,artifacts}`` plus the model dir (default
    ``workdir/models`` — the volume the runner container mounts). Returns
    ``{str(path): "dir"}`` for every dir created or confirmed.
    """
    workdir = Path(workdir)
    if workdir.exists() and not workdir.is_dir():
        raise RobovlaError("validation",
                           f"workdir {workdir} exists and is not a directory")
    model_dir = Path(model_dir) if model_dir else workdir / "models"
    made = {}
    for d in (workdir / "datasets", workdir / "runs", workdir / "artifacts", model_dir):
        d.mkdir(parents=True, exist_ok=True)
        made[str(d)] = "dir"
    return made


def cmd_node_bootstrap(args: argparse.Namespace) -> int:
    try:
        made = bootstrap_node(args.workdir, args.model_dir)
    except RobovlaError as e:
        return _err(f"{e.code}: {e.message}")
    for p in made:
        print(f"ok: {p}")
    return 0


# -- entrypoint ---------------------------------------------------------------------


def _err(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robovla",
        description="roboVLA operational CLI: backup/restore, node provisioning "
                    "(gpu_selftest, model_download, bootstrap).")
    groups = p.add_subparsers(dest="group", required=True)
    cli = groups.add_parser("cli", help="operational commands").add_subparsers(
        dest="command", required=True)

    b = cli.add_parser("backup",
                       help="backup bundle (datasets + artifacts + DB snapshot)")
    b.add_argument("--data-dir", required=True)
    b.add_argument("--dest", required=True, help="destination directory")
    b.set_defaults(fn=cmd_backup)

    r = cli.add_parser("restore", help="restore a backup bundle into a data dir")
    r.add_argument("--bundle", required=True)
    r.add_argument("--data-dir", required=True)
    r.add_argument("--force", action="store_true",
                   help="replace an existing state.db in the target")
    r.set_defaults(fn=cmd_restore)

    g = cli.add_parser("gpu_selftest",
                       help="run the stack selftest on a node (via the agent)")
    g.add_argument("--node", required=True)
    g.add_argument("--dataset", required=True,
                   help="dataset id (held-out images for the predict probe)")
    g.add_argument("--agent-url", help="agent base URL (http://ip:port)")
    g.add_argument("--config", help="fleet.yaml (node's agent address + budgets)")
    g.add_argument("--token", help=f"agent token (default: env {TOKEN_ENV})")
    g.add_argument("--job-id", help="default j_<YYYYMMDD>_0001")
    g.add_argument("--timeout", type=float, default=1800.0)
    g.set_defaults(fn=cmd_gpu_selftest)

    m = cli.add_parser("model_download",
                       help="download base-model weights into a directory")
    m.add_argument("--url", required=True)
    m.add_argument("--dest", required=True)
    m.add_argument("--name", help="file name (default: from the URL)")
    m.set_defaults(fn=cmd_model_download)

    node = cli.add_parser("node", help="node-host provisioning").add_subparsers(
        dest="node_cmd", required=True)
    nb = node.add_parser("bootstrap",
                         help="create the node data layout (idempotent)")
    nb.add_argument("--workdir", required=True)
    nb.add_argument("--model-dir", help="default <workdir>/models")
    nb.set_defaults(fn=cmd_node_bootstrap)
    return p


def main(argv: list[str] | None = None) -> int:
    """Console-script entrypoint (``uv run robovla cli ...``).

    Returns the process exit code; usage errors exit 2 (argparse).
    """
    argv = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except RobovlaError as e:
        return _err(f"{e.code}: {e.message}")


__all__ = ["main", "run_selftest", "download_model", "bootstrap_node",
           "SYSTEM_STUDENT", "DEFAULT_SELFTEST_BUDGET_GB", "TOKEN_ENV"]
