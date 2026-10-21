"""VRAM probe — measured free VRAM + holding processes (design §3, §8.2, §8.5).

Tenancy is manual (§3): the console never loads/stops host models — it shows
what the agent measures here. Two smi families, both as *fixed argv* through
the shared ``robovla.proc`` seam (no shell, ever; injectable in tests):

* CUDA (pai1 / campus nodes): ``nvidia-smi --query-*`` CSV (values in MiB
  with ``nounits``) — the verified shape on RTX 5090 (driver 615.71).
* ROCm (home dev lab, gfx1151): ``rocm-smi -j`` JSON (values in bytes) —
  shape documented below and verified at ``home_selftest`` (§11).

The view mirrors ``Runner.vram()`` (the FakeRunner oracle):
``{free_gb, total_gb, name, procs: [{pid, name, vram_gb, agent_spawned}]}``.
``agent_spawned`` marks pids in the agent's own spawn registry — the scope
for the teacher-confirmed orphan purge (§8.2). Host-level processes (e.g. the
27B assistant) are *shown* but never touched.

Probe failure (smi missing/hung/nonzero with no output) -> ``node_offline``:
a node whose GPU can't be measured is a node that can't take work (§10
fail-fast).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from robovla.core.errors import RobovlaError
from robovla import proc

# -- fixed argv (the only smi invocations this agent makes) ------------------------

CUDA_GPU_ARGV = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.used",
    "--format=csv,noheader,nounits",
]
CUDA_PROCS_ARGV = [
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
]
ROCM_ARGV = ["rocm-smi", "--showpids", "--showmeminfo", "vram", "-j"]

PROBE_TIMEOUT_S = 10.0

#: Expected ROCm JSON shape (one card; verified at home_selftest, §11):
#:   {"card0": {"vram": {"total": <bytes>, "used": <bytes>},
#:              "processes": {"<pid>": {"name": str, "mem_used": <bytes>}}}}
_ROCM_MEM_KEYS = ("vram", "mem_info", "memory")
_ROCM_PROC_KEYS = ("processes", "process_list", "pids")


def _gb_from_mib(mib: float) -> float:
    return round(mib / 1024.0, 2)


def _gb_from_bytes(b: float) -> float:
    return round(b / (1024.0 ** 3), 2)


# -- pure parsers (unit-tested on canned outputs) -----------------------------------


def parse_cuda_gpu(lines: list[str]) -> dict[str, Any]:
    """First line ``name, total_mib, used_mib`` -> {name, total_gb, free_gb}."""
    if not lines:
        raise RobovlaError("node_offline", "gpu probe failed: no nvidia-smi output")
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != 3:
        raise RobovlaError("node_offline",
                           f"gpu probe failed: unexpected nvidia-smi row {lines[0]!r}")
    name = parts[0]
    try:
        total_mib, used_mib = float(parts[1]), float(parts[2])
    except ValueError:
        raise RobovlaError("node_offline",
                           f"gpu probe failed: non-numeric row {lines[0]!r}") from None
    return {"name": name, "total_gb": _gb_from_mib(total_mib),
            "free_gb": _gb_from_mib(total_mib - used_mib)}


def parse_cuda_procs(lines: list[str]) -> list[dict[str, Any]]:
    """Lines of ``pid, process_name, used_memory_mib`` -> procs list."""
    procs: list[dict[str, Any]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue  # smi prints blank lines / warnings; skip non-rows
        try:
            pid = int(parts[0])
            used_mib = float(parts[2])
        except ValueError:
            continue
        procs.append({"pid": pid, "name": parts[1], "vram_gb": _gb_from_mib(used_mib)})
    return procs


def _rocm_card(data: Any) -> dict[str, Any]:
    """Pick the first card block (single-GPU nodes, §1)."""
    if not isinstance(data, dict) or not data:
        raise RobovlaError("node_offline", "gpu probe failed: empty rocm-smi output")
    for key, card in data.items():
        if key.startswith("card") and isinstance(card, dict):
            return card
    raise RobovlaError("node_offline", "gpu probe failed: no card block in rocm-smi output")


def parse_rocm(data: Any) -> dict[str, Any]:
    """rocm-smi JSON -> {name, total_gb, free_gb, procs} (values in bytes)."""
    card = _rocm_card(data)
    mem = None
    for k in _ROCM_MEM_KEYS:
        if isinstance(card.get(k), dict):
            mem = card[k]
            break
    if mem is None or "total" not in mem:
        raise RobovlaError("node_offline", "gpu probe failed: no vram block in rocm-smi output")
    total_b = float(mem["total"])
    used_b = float(mem.get("used", 0))
    procs: list[dict[str, Any]] = []
    block = None
    for k in _ROCM_PROC_KEYS:
        if isinstance(card.get(k), dict):
            block = card[k]
            break
    for pid_s, info in (block or {}).items():
        try:
            pid = int(pid_s)
        except (TypeError, ValueError):
            continue
        if not isinstance(info, dict):
            continue
        procs.append({
            "pid": pid,
            "name": str(info.get("name", "?")),
            "vram_gb": _gb_from_bytes(float(info.get("mem_used", 0))),
        })
    name = str(card.get("card_name", "AMD GPU"))
    return {"name": name, "total_gb": _gb_from_bytes(total_b),
            "free_gb": _gb_from_bytes(total_b - used_b), "procs": procs}


# -- the probe (proc seam injectable for CPU tests) ----------------------------------

Runner = Callable[..., "proc.subprocess.CompletedProcess[str]"]


class VramProbe:
    """One node's GPU, as measured (implements the data behind Runner.vram)."""

    def __init__(self, gpu: str = "cuda", runner: Runner | None = None) -> None:
        if gpu not in ("cuda", "rocm"):
            raise RobovlaError("validation", f"unknown gpu family: {gpu!r}")
        self.gpu = gpu
        self._run = runner or self._default_run

    @staticmethod
    def _default_run(argv: list[str], **kw: Any):
        return proc.run_fixed_argv(argv, timeout_s=PROBE_TIMEOUT_S, **kw)

    def _lines(self, argv: list[str]) -> list[str]:
        try:
            out = self._run(argv)
        except RobovlaError as e:
            raise RobovlaError("node_offline", f"gpu probe failed: {e.message}") from e
        if out.returncode != 0:
            detail = (out.stderr or out.stdout or "").strip().splitlines()
            raise RobovlaError("node_offline",
                               f"gpu probe failed: {argv[0]} exit {out.returncode}"
                               + (f": {detail[0]}" if detail else ""))
        return [ln for ln in out.stdout.splitlines() if ln.strip()]

    def view(self, known_pids: frozenset[int] | set[int] | None = None) -> dict[str, Any]:
        """Measured VRAM view (the ``/vram`` payload; ``agent_spawned`` set
        from the caller's spawn registry)."""
        known = frozenset(known_pids or ())
        if self.gpu == "cuda":
            head = parse_cuda_gpu(self._lines(CUDA_GPU_ARGV))
            procs = parse_cuda_procs(self._lines(CUDA_PROCS_ARGV))
        else:
            raw = self._lines(ROCM_ARGV)
            data = json.loads("\n".join(raw)) if raw else None
            head = parse_rocm(data)
            procs = head.pop("procs")
        for p in procs:
            p["agent_spawned"] = p["pid"] in known
        return {**head, "procs": procs}

    def purge(self, pids: list[int], known_pids: frozenset[int] | set[int],
              grace_s: float = 10.0) -> int:
        """Kill agent-scoped orphan processes (§8.2). Returns count killed.

        Host-level pids (not in the spawn registry) are rejected with
        ``validation`` — the agent never touches processes it didn't spawn
        (matches the FakeRunner oracle's message).
        """
        known = frozenset(known_pids)
        bad = [p for p in pids if p not in known]
        if bad:
            raise RobovlaError("validation", f"not agent-scoped pids: {bad}")
        killed = 0
        for pid in pids:
            if proc.kill_process(pid, grace_s) != "gone":
                killed += 1
            proc.reap(pid)
        return killed
