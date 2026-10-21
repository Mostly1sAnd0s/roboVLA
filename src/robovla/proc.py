"""Fixed-argv exec — the "no shell, ever" seam (design §9.8, Appendix C).

Every external process the system launches goes through this module:

* argv is a **list of opaque strings** executed with ``shell=False``. User
  text (dataset ids, prompts, notes) is never interpolated into a shell
  string; at worst it becomes one argv element of a program the image owns.
* ``check_argv`` enforces well-formedness only. A string with spaces, quotes,
  or ``$`` is a *legal* argv element — escaping is the shell's job, and there
  is no shell.
* ``run_fixed_argv`` — short probe commands (``nvidia-smi``/``rocm-smi``):
  synchronous, returns the CompletedProcess; the caller decides what a
  nonzero exit means.
* ``spawn_fixed_argv`` — long job processes: async Popen with stdout/stderr
  redirected to a log file (the agent's log tail, §9.3).
* ``kill_process`` — SIGTERM, up to ``grace_s``, then SIGKILL (§8.3: 10 s
  grace). ``reap`` — best-effort ``waitpid`` for agent-scoped orphans
  (§8.2): it never raises for a pid we do not own.

One error type (:class:`RobovlaError`); malformed argv -> ``validation``,
spawn/exec/timeout failures -> ``internal``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .core.errors import RobovlaError


def check_argv(argv: Any) -> list[str]:
    """Validate a fixed argv (raises ``validation``). Returns it unchanged."""
    if not isinstance(argv, list) or not argv:
        raise RobovlaError("validation", "argv must be a non-empty list of strings")
    for i, arg in enumerate(argv):
        if not isinstance(arg, str):
            raise RobovlaError("validation", f"argv[{i}] must be a string: {arg!r}")
        if not arg:
            raise RobovlaError("validation", f"argv[{i}] must be non-empty")
        if "\x00" in arg:
            raise RobovlaError("validation", f"argv[{i}] contains a NUL byte")
    return list(argv)


def run_fixed_argv(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
    timeout_s: float | None = None,
    input_bytes: bytes | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one fixed-argv command synchronously (probe commands).

    Returns the CompletedProcess (nonzero exits are data, not errors); raises
    ``internal`` only for exec/timeout failures.
    """
    argv = check_argv(argv)
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            timeout=timeout_s,
            input=input_bytes.decode("utf-8") if input_bytes is not None else None,
            env=env,
            capture_output=True,
            text=True,
            shell=False,  # the invariant — never overridden, ever
        )
    except subprocess.TimeoutExpired as e:
        raise RobovlaError("internal",
                           f"command timed out after {timeout_s:g} s: {argv[0]}") from e
    except OSError as e:
        raise RobovlaError("internal", f"exec failed: {argv[0]}: {e.strerror or e}") from e


def spawn_fixed_argv(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
    log_path: str | Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Spawn a long job process (fixed argv, ``shell=False``).

    stdout + stderr are redirected to ``log_path`` (the agent's per-job log
    tail, §9.3). ``env`` is *extra* variables merged over the inherited
    environment. Raises ``internal`` if the program cannot start.
    """
    argv = check_argv(argv)
    full_env = {**os.environ, **(env or {})}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "ab") as log:
            return subprocess.Popen(
                argv,
                cwd=str(cwd) if cwd is not None else None,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=full_env,
                shell=False,  # the invariant — never overridden, ever
            )
    except OSError as e:
        raise RobovlaError("internal", f"spawn failed: {argv[0]}: {e.strerror or e}") from e


def alive(pid: int) -> bool:
    """True if ``pid`` exists and is not a reaped zombie.

    ``os.kill(pid, 0)`` succeeds for zombies (they linger until reaped), so
    the liveness check reaps first when we are the parent (``waitpid``
    ``WNOHANG``), then falls back to the signal probe for pids we do not own.
    """
    if reap(pid):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours — conservative (single-user container)
    return True


def kill_process(pid: int, grace_s: float = 10.0, *,
                 sleep: Any = time.sleep) -> str:
    """SIGTERM -> wait up to ``grace_s`` -> SIGKILL (§8.3).

    Returns ``"term"`` (died to the SIGTERM), ``"kill"`` (needed the
    SIGKILL), or ``"gone"`` (pid already dead). Never raises for a missing
    pid — kill races are normal (§8.3: kill is idempotent).
    """
    if not alive(pid):
        return "gone"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "gone"
    deadline = time.monotonic() + grace_s
    while alive(pid):
        if time.monotonic() >= deadline:
            break
        sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if not alive(pid):
        return "term"
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "term"
    # SIGKILL is uncatchable: wait (bounded) until the pid is actually gone so
    # "kill" means dead-and-reaped, not "signal sent".
    deadline = time.monotonic() + 5.0
    while alive(pid) and time.monotonic() < deadline:
        sleep(0.01)
    return "kill"


def reap(pid: int) -> bool:
    """Best-effort ``waitpid(WNOHANG)`` for an agent-scoped orphan (§8.2).

    Returns True if this process reaped the pid. Never raises for a pid we do
    not own (``ChildProcessError``) — orphans may have been re-parented to
    pid 1 or already reaped by someone else.
    """
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    except OSError:
        return False
    return waited == pid
