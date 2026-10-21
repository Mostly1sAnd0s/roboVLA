"""Dataset bundle pull from the console (design §5.1, §9.9).

The agent's **only** call toward the console: at dispatch, before spawning a
job process, it pulls the prepared dataset tar over LAN HTTP, authenticated
with the *same* per-node pre-shared agent token (the console verifies it
against the argon2 hash in its DB). The bundle is ~1 GB at GbE —
seconds-to-minutes — so it streams to a temp file while computing sha256,
then verifies against the console's ``X-Dataset-Sha256`` header and
extracts into the node workdir with the shared tar-slip-safe extractor
(``robovla.fs.safe_extract_tar`` — the same rules as the upload side, §9.8).

``datasets/`` under the workdir is the pulled bundle: ephemeral, re-pullable
at dispatch, excluded from backup (§10). Pulled at **every** dispatch — the
console's bundle is the source of truth (re-prep can change content under
the same dataset id), so an existing dir is replaced, never trusted.

Errors: console error codes pass through unchanged (§9.6 stable set);
connection failure -> ``node_offline`` (the job fails fast and the console
reschedules on the next fitting runner, §10).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import httpx

from robovla.core.errors import CODES, RobovlaError
from robovla.fs import safe_extract_tar, sanitize_dataset_id

#: ~1 GB over GbE is seconds-to-minutes (§5.1); generous on purpose.
BUNDLE_TIMEOUT_S = 1800.0
_CHUNK = 1024 * 1024


def _console_error(resp: httpx.Response) -> RobovlaError:
    """Map a console error body to the shared §9.6 RobovlaError."""
    code, message = "internal", f"HTTP {resp.status_code}"
    try:
        body = resp.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            code = err.get("code", code)
            message = str(err.get("message", message))
    except (ValueError, AttributeError):
        pass
    if code not in CODES:
        code = "internal"
    return RobovlaError(code, f"console bundle: {message}")


def pull_bundle(
    console_url: str,
    token: str,
    dataset_id: str,
    workdir: str | Path,
    *,
    client: httpx.Client | None = None,
    timeout_s: float = BUNDLE_TIMEOUT_S,
) -> Path:
    """Pull + verify + extract the dataset bundle; returns the dataset dir.

    Pulled **at every dispatch** (§5.1): the console's bundle is the source
    of truth (re-prep can change content under the same dataset id), so an
    existing dir is replaced, not trusted. The pull is idempotent — the
    same bundle always yields the same tree.
    """
    sanitized = sanitize_dataset_id(dataset_id)  # core pattern (DRY)
    workdir = Path(workdir)
    dest_root = workdir / "datasets" / sanitized

    url = f"{console_url.rstrip('/')}/api/agent/datasets/{dataset_id}/bundle"
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout_s)
    tmp_path = workdir / "bundles" / f".{sanitized}.tmp"
    try:
        if owns_client:
            client.timeout = httpx.Timeout(timeout_s)
        sha = hashlib.sha256()
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with client.stream(
                "GET", url, headers={"Authorization": f"Bearer {token}"}
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()  # drain the error body
                    raise _console_error(resp)
                expected = resp.headers.get("X-Dataset-Sha256", "").lower()
                if not expected:
                    raise RobovlaError("conflict",
                                       "console bundle: missing X-Dataset-Sha256")
                with open(tmp_path, "wb") as out:
                    for chunk in resp.iter_bytes(_CHUNK):
                        sha.update(chunk)
                        out.write(chunk)
        except httpx.HTTPError as e:
            raise RobovlaError("node_offline",
                               f"console unreachable for bundle pull: "
                               f"{e.__class__.__name__}") from e
        if sha.hexdigest() != expected:
            raise RobovlaError("conflict",
                               "bundle sha256 mismatch (corrupt transfer)")
        # replace any stale partial extraction, then extract (tar-slip-safe)
        if dest_root.exists():
            shutil.rmtree(dest_root)
        safe_extract_tar(tmp_path.read_bytes(), workdir / "datasets")
    finally:
        tmp_path.unlink(missing_ok=True)
        if owns_client:
            client.close()
    return dest_root
