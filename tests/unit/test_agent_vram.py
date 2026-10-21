"""agent.vram — nvidia-smi / rocm-smi probe (design §3, §8.2, §8.5).

Parsers are pure (canned smi output); VramProbe runs them over an injectable
proc seam, so the whole probe is testable on a GPU-less CI box. The contract
is the FakeRunner oracle: {free_gb, total_gb, procs:[{pid,name,vram_gb,
agent_spawned}]}, purge rejects non-agent pids, probe failure -> node_offline.
"""

from __future__ import annotations

import subprocess

import pytest

from agent.vram import (
    ROCM_ARGV,
    VramProbe,
    parse_cuda_gpu,
    parse_cuda_procs,
    parse_rocm,
)
from robovla.core.errors import RobovlaError


def cp(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["smi"], returncode=rc,
                                       stdout=stdout, stderr=stderr)


# -- cuda parsers ---------------------------------------------------------------


def test_parse_cuda_gpu():
    v = parse_cuda_gpu(["NVIDIA GeForce RTX 5090, 32608, 1024"])
    assert v["name"] == "NVIDIA GeForce RTX 5090"
    assert v["total_gb"] == pytest.approx(31.84)
    assert v["free_gb"] == pytest.approx(30.84)


def test_parse_cuda_gpu_bad_rows():
    for bad in ([], ["no commas"], ["a,b", "x,y,z,w"], ["n,abc,def"]):
        with pytest.raises(RobovlaError) as ei:
            parse_cuda_gpu(bad)
        assert ei.value.code == "node_offline"


def test_parse_cuda_procs_skips_noise():
    procs = parse_cuda_procs(["", "  ", "1234, python, 10240",
                              "5678, /usr/bin/node, 512", "warning: not a row"])
    assert procs == [
        {"pid": 1234, "name": "python", "vram_gb": pytest.approx(10.0)},
        {"pid": 5678, "name": "/usr/bin/node", "vram_gb": pytest.approx(0.5)},
    ]


def test_parse_cuda_procs_empty_is_empty():
    assert parse_cuda_procs([]) == []


# -- rocm parser -------------------------------------------------------------------


def test_parse_rocm():
    data = {
        "card0": {
            "card_name": "AMD Radeon 8060S",
            "vram": {"total": 137438953472, "used": 2147483648},
            "processes": {
                "4321": {"name": "python", "mem_used": 1073741824},
                "notapid": {"name": "x", "mem_used": 1},
            },
        }
    }
    v = parse_rocm(data)
    assert v["name"] == "AMD Radeon 8060S"
    assert v["total_gb"] == pytest.approx(128.0)
    assert v["free_gb"] == pytest.approx(126.0)
    assert v["procs"] == [{"pid": 4321, "name": "python",
                           "vram_gb": pytest.approx(1.0)}]


def test_parse_rocm_bad_shapes():
    for bad in (None, {}, {"gpu0": {}},
                {"card0": {"vram": {}, "processes": {}}},
                {"card0": {}}):
        with pytest.raises(RobovlaError) as ei:
            parse_rocm(bad)
        assert ei.value.code == "node_offline"


# -- VramProbe (proc seam) -----------------------------------------------------------


def _cuda_probe(procs: dict | None = None):
    """A fake smi runner: canned CSV for both queries."""
    procs = procs or {}
    def run(argv: list[str], **kw):
        if "query-gpu" in " ".join(argv):
            return cp(0, "NVIDIA GeForce RTX 5090, 32608, 23552\n")
        return cp(0, "\n".join(f"{p}, {n}, {m}\n" for p, (n, m) in procs.items()))
    return run


def test_probe_view_marks_agent_spawned():
    probe = VramProbe("cuda", runner=_cuda_probe(
        {"777": ("python", 9216), "999": ("ninfer", 30208)}))
    v = probe.view(known_pids=frozenset({777}))
    # gpu line: total 32608 MiB, used 23552 MiB -> free 9056 MiB
    assert v["free_gb"] == pytest.approx(8.84)
    assert v["total_gb"] == pytest.approx(31.84)
    by_pid = {p["pid"]: p for p in v["procs"]}
    assert by_pid[777]["agent_spawned"] is True
    assert by_pid[999]["agent_spawned"] is False


def test_probe_rocm_view():
    def run(argv: list[str], **kw):
        assert argv == ROCM_ARGV
        return cp(0, '{"card0": {"vram": {"total": 137438953472, '
                     '"used": 0}, "processes": {}}}')
    v = VramProbe("rocm", runner=run).view()
    assert v["total_gb"] == pytest.approx(128.0) and v["procs"] == []


def test_probe_failure_is_node_offline():
    def missing(argv, **kw):
        return cp(127, stderr="nvidia-smi: command not found")
    with pytest.raises(RobovlaError) as ei:
        VramProbe("cuda", runner=missing).view()
    assert ei.value.code == "node_offline"

    def timeout(argv, **kw):
        raise RobovlaError("internal", "command timed out after 10 s: nvidia-smi")
    with pytest.raises(RobovlaError) as ei:
        VramProbe("cuda", runner=timeout).view()
    assert ei.value.code == "node_offline" and "timed out" in ei.value.message


def test_probe_bad_gpu_family():
    with pytest.raises(RobovlaError) as ei:
        VramProbe("metal")
    assert ei.value.code == "validation"


# -- purge (agent-scoped only, §8.2) ---------------------------------------------------


def test_purge_rejects_non_agent_pids():
    probe = VramProbe("cuda", runner=_cuda_probe())
    with pytest.raises(RobovlaError) as ei:
        probe.purge([4242], known_pids=frozenset({111}))
    assert ei.value.code == "validation"
    assert "not agent-scoped pids: [4242]" in ei.value.message


def test_purge_kills_only_scoped_pids():
    import sys
    import tempfile
    import time

    from robovla import proc

    # two real children: one in the registry, one not
    code = "import time\ntime.sleep(30)"
    p1 = proc.spawn_fixed_argv([sys.executable, "-c", code],
                               cwd=None, log_path=tempfile.mkstemp()[1])
    p2 = proc.spawn_fixed_argv([sys.executable, "-c", code],
                               cwd=None, log_path=tempfile.mkstemp()[1])
    time.sleep(0.3)
    probe = VramProbe("cuda", runner=_cuda_probe())
    killed = probe.purge([p1.pid], known_pids=frozenset({p1.pid}))
    assert killed == 1
    assert not proc.alive(p1.pid)
    assert proc.alive(p2.pid)  # host-level twin untouched
    proc.kill_process(p2.pid)
