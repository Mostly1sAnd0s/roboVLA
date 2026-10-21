"""proc — fixed-argv exec (design §9.8 "No shell, ever", Appendix C).

The guarantee is structural: argv is a list of opaque strings executed with
``shell=False`` — the agent never interpolates user text into a shell string.
``check_argv`` only enforces well-formedness (a string with spaces or quotes
is a *perfectly legal* argv element; that is the point of argv).

Exec is exercised with real subprocesses (``sys.executable -c`` — still a
list argv, no shell involved).
"""

from __future__ import annotations


import sys
import textwrap
import time

import pytest

from robovla.core.errors import RobovlaError
from robovla import proc


# -- check_argv -------------------------------------------------------------------


def test_check_argv_accepts_legal_args():
    argv = ["/usr/bin/python", "-c", "print('a b')", "--flag", "val with space"]
    assert proc.check_argv(argv) == argv


def test_check_argv_rejects_malformed():
    for bad in (
        None,                       # not a list
        "python -c x",              # a shell string (the thing we must never do)
        [],                         # empty
        ["python", ""],             # empty element
        ["python", "a\x00b"],       # NUL byte
        ["python", 42],             # non-string element
        [b"bytes"],                 # bytes element
    ):
        with pytest.raises(RobovlaError) as ei:
            proc.check_argv(bad)
        assert ei.value.code == "validation"


# -- run_fixed_argv ---------------------------------------------------------------


def test_run_fixed_argv_captures_output(tmp_path):
    out = proc.run_fixed_argv(
        [sys.executable, "-c", "import sys; print('hello'); print('err', file=sys.stderr)"],
        cwd=tmp_path,
    )
    assert out.returncode == 0
    assert out.stdout.strip() == "hello"
    assert out.stderr.strip() == "err"


def test_run_fixed_argv_nonzero_exit_is_returned_not_raised(tmp_path):
    out = proc.run_fixed_argv([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert out.returncode == 3  # caller decides the semantics


def test_run_fixed_argv_missing_program_is_internal(tmp_path):
    with pytest.raises(RobovlaError) as ei:
        proc.run_fixed_argv(["/nonexistent/binary-xyz", "--x"])
    assert ei.value.code == "internal"


def test_run_fixed_argv_timeout_is_internal():
    with pytest.raises(RobovlaError) as ei:
        proc.run_fixed_argv([sys.executable, "-c", "import time; time.sleep(5)"],
                            timeout_s=0.3)
    assert ei.value.code == "internal" and "timed out" in ei.value.message


def test_run_fixed_argv_cwd_and_input(tmp_path):
    (tmp_path / "cwd.txt").write_text("x")
    out = proc.run_fixed_argv(
        [sys.executable, "-c", "import os; print(os.getcwd()); print(input())"],
        cwd=tmp_path, input_bytes=b"ping\n")
    assert out.stdout.splitlines()[0] == str(tmp_path)
    assert "ping" in out.stdout


# -- spawn_fixed_argv ---------------------------------------------------------------


def test_spawn_fixed_argv_redirects_stdio_to_log(tmp_path):
    log = tmp_path / "job.log"
    p = proc.spawn_fixed_argv(
        [sys.executable, "-c", "print('line1'); print('line2', flush=True)"],
        cwd=tmp_path, log_path=log)
    assert p.wait(timeout=10) == 0
    assert log.read_text() == "line1\nline2\n"


def test_spawn_fixed_argv_env_passthrough(tmp_path):
    log = tmp_path / "job.log"
    p = proc.spawn_fixed_argv(
        [sys.executable, "-c", "import os; print(os.environ['ROBOVLA_TEST_MARKER'])"],
        cwd=tmp_path, log_path=log,
        env={"ROBOVLA_TEST_MARKER": "mark-1"})
    assert p.wait(timeout=10) == 0
    assert log.read_text().strip() == "mark-1"


def test_spawn_fixed_argv_missing_program_is_internal(tmp_path):
    with pytest.raises(RobovlaError) as ei:
        proc.spawn_fixed_argv(["/nonexistent/binary-xyz"], cwd=tmp_path,
                              log_path=tmp_path / "l")
    assert ei.value.code == "internal"


# -- kill_process (SIGTERM -> grace -> SIGKILL, §8.3) --------------------------------


def test_kill_process_term_when_it_exits_quickly():
    # default SIGTERM disposition kills the process -> "term". The raw reap
    # inside kill_process steals the exit status from Popen (CPython returns
    # 0 there), so liveness is asserted through proc, not Popen.wait().
    p = _spawn_sleeping(30)
    outcome = proc.kill_process(p.pid, grace_s=1.0)
    assert outcome == "term"
    assert not proc.alive(p.pid)


def test_kill_process_kill_after_grace_when_term_ignored():
    p = _spawn_sleeping(30, trap_term=True)
    t0 = time.monotonic()
    outcome = proc.kill_process(p.pid, grace_s=0.3)
    elapsed = time.monotonic() - t0
    assert outcome == "kill"
    assert elapsed < 2.0  # did not wait out the full sleep
    assert not proc.alive(p.pid)


def test_kill_process_gone_when_pid_missing():
    assert proc.kill_process(999_999_999, grace_s=0.1) == "gone"


def test_reap_only_our_children():
    # reaping an arbitrary pid must be a best-effort no-op, never an exception
    assert proc.reap(999_999_999) is False
    p = _spawn_sleeping(5)
    proc.kill_process(p.pid, grace_s=0.5)
    p.wait(timeout=5)  # direct-child reap via Popen
    assert proc.reap(p.pid) is False


# -- helpers ------------------------------------------------------------------------


def _spawn_sleeping(seconds: int, trap_term: bool = False):
    """A real child process that sleeps; optionally ignores SIGTERM.

    Waits for the child's ``ready`` marker so signal tests run *after* the
    Python-level handler is installed (a SIGTERM during interpreter startup
    would use the default C disposition and kill it anyway).
    """
    code = textwrap.dedent(f"""
        import signal, sys, time
        {f'signal.signal(signal.SIGTERM, lambda *a: None)' if trap_term else ''}
        print('ready', flush=True)
        time.sleep({seconds})
        sys.exit(0)
    """).strip()
    import tempfile, os
    fd, path = tempfile.mkstemp(prefix="robovla-proc-test-", suffix=".log")
    os.close(fd)
    p = proc.spawn_fixed_argv([sys.executable, "-c", code], cwd=None,
                              log_path=path)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if "ready" in open(path).read():
            return p
        time.sleep(0.02)
    raise AssertionError(f"child never became ready: {p.wait()}")
