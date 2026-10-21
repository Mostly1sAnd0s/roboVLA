"""fleet.yaml schema + loader + diff/reload rules (design §10, Appendix A).

One config: ``fleet.yaml`` — one schema, one validator, hot reload
revalidates. Core stays pure: :func:`parse_fleet` takes YAML *text*; reading
the file is the seam's job (app/cli). The fleet is the only place hardware
lives (design §10: "hardware lives in config, not code") — no node name or IP
appears anywhere else.

:func:`diff_fleet` produces flat before/after change records for the admin
audit log (§8: every config change is audited); the reload *behavior*
(active jobs unaffected, queued jobs revalidated against new budgets) is the
scheduler's use of the new config, not this module's.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import yaml

from .budgets import FIXED_GB, QUANT_PROFILES
from .errors import RobovlaError

#: Supported fleet.yaml version.
FLEET_VERSION = 1

#: Node id format — also referenced by JobSpec.node.id (one pattern, DRY).
NODE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

_ADMIN_RATE_RE = re.compile(r"^(\d+)/min$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_BATCH_OPTIONS = (1, 4, 8)  # §3: allowed batch (1 / 4 / 8 with matching grad-accum)


def _fail(message: str) -> None:
    raise RobovlaError("validation", message)


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{path}: must be a non-empty string")
    return value


def _int(value: Any, path: str, *, lo: int, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path}: must be an integer")
    if value < lo or (hi is not None and value > hi):
        _fail(f"{path}: must be in [{lo}, {hi}]")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{path}: must be a boolean")
    return value


def _num(value: Any, path: str, *, lo: float, hi: float | None = None, exclusive_lo: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{path}: must be a finite number")
    v = float(value)
    if (v <= lo if exclusive_lo else v < lo) or (hi is not None and v > hi):
        _fail(f"{path}: out of bounds")
    return v


def _dict(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        _fail(f"{path}: must be an object")
    return value


def _missing(d: dict, keys: tuple[str, ...], path: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        _fail(f"{path}: missing key(s) {', '.join(missing)}")


def _ipv4(value: Any, path: str) -> str:
    s = _str(value, path)
    parts = s.split(".")
    if len(parts) != 4 or any(not p.isdigit() or int(p) > 255 or (len(p) > 1 and p[0] == "0") for p in parts):
        _fail(f"{path}: {s!r} is not a valid IPv4 address")
    return s


# --- dataclasses -------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    lan_ip: str
    port: int
    token_ref: str  # env var name on the console host; plaintext never in YAML (§9.8)


@dataclass(frozen=True)
class GpuBudgets:
    train_gb: float
    infer_gb: float


@dataclass(frozen=True)
class GpuConfig:
    vram_gb: float
    quant: str
    train_slots: int
    infer_slots: int
    budgets: GpuBudgets


@dataclass(frozen=True)
class Presets:
    batch: tuple[int, ...]  # allowed student batch options (max = base-budget batch)


@dataclass(frozen=True)
class NodeConfig:
    id: str
    label: str
    image: str
    workdir: str
    agent: AgentConfig
    gpu: GpuConfig
    presets: Presets
    priority: int
    student_selectable: bool
    enabled: bool


@dataclass(frozen=True)
class StudentLimits:
    max_running: int
    max_queued: int
    max_dataset_bytes: int
    max_run_min: int
    run_now: bool


@dataclass(frozen=True)
class Limits:
    student: StudentLimits
    admin_rate: str
    admin_rate_per_min: int


@dataclass(frozen=True)
class BackupConfig:
    enabled: bool
    at: str  # "HH:MM" (local time on the console host)
    dest: str


@dataclass(frozen=True)
class Retention:
    run_logs_per_student: int
    merged_export_days: int
    backup: BackupConfig


@dataclass(frozen=True)
class ConsoleConfig:
    bind: str
    port: int
    tls: bool
    data_dir: str
    session_ttl_h: int
    limits: Limits
    retention: Retention


@dataclass(frozen=True)
class FleetConfig:
    version: int
    console: ConsoleConfig
    nodes: tuple[NodeConfig, ...]

    def nodes_by_id(self) -> dict[str, NodeConfig]:
        return {n.id: n for n in self.nodes}

    def enabled_nodes(self) -> list[NodeConfig]:
        return [n for n in self.nodes if n.enabled]

    def nodes_by_priority(self) -> list[NodeConfig]:
        """First-fit order: ``priority`` higher first; ties keep file order
        (stable sort — design §3)."""
        return sorted(self.nodes, key=lambda n: -n.priority)


@dataclass(frozen=True)
class Change:
    """One audited config change (before/after, §8: append-only audit log)."""

    scope: str  # e.g. "console.limits.student" | "nodes.<node-id>.gpu" | "nodes"
    field: str  # e.g. "max_queued" | "train_gb" | "added" | "removed"
    before: Any
    after: Any

    def to_dict(self) -> dict:
        return {"scope": self.scope, "field": self.field, "before": self.before, "after": self.after}


# --- validators ----------------------------------------------------------------


def _node(d: Any, path: str) -> NodeConfig:
    if not isinstance(d, dict):
        _fail(f"{path}: must be an object")
    _missing(d, ("id", "label", "image", "workdir", "agent", "gpu", "presets"), path)
    for key in ("label", "image", "workdir"):
        if not isinstance(d.get(key), str) or not d[key]:
            _fail(f"{path}.{key}: must be a non-empty string")

    agent = _dict(d["agent"], f"{path}.agent")
    _missing(agent, ("lan_ip", "port", "token_ref"), f"{path}.agent")

    gpu = _dict(d["gpu"], f"{path}.gpu")
    _missing(gpu, ("vram_gb", "quant", "train_slots", "infer_slots", "budgets"), f"{path}.gpu")
    budgets = _dict(gpu["budgets"], f"{path}.gpu.budgets")
    _missing(budgets, ("train_gb", "infer_gb"), f"{path}.gpu.budgets")

    presets = _dict(d.get("presets", {}), f"{path}.presets")
    batches = presets.get("batch")
    if (
        not isinstance(batches, list)
        or not batches
        or len(set(batches)) != len(batches)
        or any(b not in _BATCH_OPTIONS for b in batches)
    ):
        _fail(f"{path}.presets.batch: must be a non-empty subset of {_BATCH_OPTIONS}")

    node_id = d["id"]
    if not NODE_ID_RE.fullmatch(node_id):
        _fail(f"{path}.id: {node_id!r} must match {NODE_ID_RE.pattern}")

    quant = d["gpu"].get("quant")
    if quant not in QUANT_PROFILES:
        _fail(f"{path}.gpu.quant: {quant!r} must be one of {sorted(QUANT_PROFILES)}")

    train_gb = _num(budgets["train_gb"], f"{path}.gpu.budgets.train_gb", lo=0.0, exclusive_lo=True)
    infer_gb = _num(budgets["infer_gb"], f"{path}.gpu.budgets.infer_gb", lo=0.0, exclusive_lo=True)
    floor = FIXED_GB[quant]
    if train_gb < floor:
        _fail(
            f"{path}.gpu.budgets.train_gb: {train_gb:g} GB is below the {quant} model floor ({floor:g} GB)"
        )

    if "priority" not in d:
        d = {**d, "priority": 0}
    return NodeConfig(
        id=node_id,
        label=d["label"],
        image=d["image"],
        workdir=d["workdir"],
        agent=AgentConfig(
            lan_ip=_ipv4(agent["lan_ip"], f"{path}.agent.lan_ip"),
            port=_int(agent["port"], f"{path}.agent.port", lo=1, hi=65535),
            token_ref=_str(agent["token_ref"], f"{path}.agent.token_ref"),
        ),
        gpu=GpuConfig(
            vram_gb=_num(d["gpu"]["vram_gb"], f"{path}.gpu.vram_gb", lo=0.0, exclusive_lo=True),
            quant=quant,
            train_slots=_int(d["gpu"]["train_slots"], f"{path}.gpu.train_slots", lo=1, hi=2),
            infer_slots=_int(d["gpu"]["infer_slots"], f"{path}.gpu.infer_slots", lo=1, hi=2),
            budgets=GpuBudgets(train_gb=train_gb, infer_gb=infer_gb),
        ),
        presets=Presets(batch=tuple(batches)),
        priority=_int(d["priority"], f"{path}.priority", lo=-100, hi=100),
        student_selectable=_bool(d.get("student_selectable", False), f"{path}.student_selectable"),
        enabled=_bool(d.get("enabled", False), f"{path}.enabled"),
    )


def _console(d: Any) -> ConsoleConfig:
    c = _dict(d, "console")
    _missing(
        c,
        ("bind", "port", "tls", "data_dir", "session_ttl_h", "limits", "retention"),
        "console",
    )
    limits = _dict(c["limits"], "console.limits")
    _missing(limits, ("student", "admin_rate"), "console.limits")
    s = _dict(limits["student"], "console.limits.student")
    _missing(
        s,
        ("max_running", "max_queued", "max_dataset_bytes", "max_run_min", "run_now"),
        "console.limits.student",
    )
    retention = _dict(c["retention"], "console.retention")
    _missing(retention, ("run_logs_per_student", "merged_export_days", "backup"), "console.retention")
    backup = _dict(retention["backup"], "console.retention.backup")
    _missing(backup, ("enabled", "at", "dest"), "console.retention.backup")

    rate = _str(limits["admin_rate"], "console.limits.admin_rate")
    m = _ADMIN_RATE_RE.fullmatch(rate)
    if not m or int(m.group(1)) < 1:
        _fail(f"console.limits.admin_rate: {rate!r} must look like '60/min'")
    at = _str(backup["at"], "console.retention.backup.at")
    if not _TIME_RE.fullmatch(at):
        _fail(f"console.retention.backup.at: {at!r} must be 'HH:MM' (24 h)")

    return ConsoleConfig(
        bind=_str(c["bind"], "console.bind"),
        port=_int(c["port"], "console.port", lo=1, hi=65535),
        tls=_bool(c["tls"], "console.tls"),
        data_dir=_str(c["data_dir"], "console.data_dir"),
        session_ttl_h=_int(c["session_ttl_h"], "console.session_ttl_h", lo=1, hi=168),
        limits=Limits(
            student=StudentLimits(
                max_running=_int(s["max_running"], "console.limits.student.max_running", lo=0),
                max_queued=_int(s["max_queued"], "console.limits.student.max_queued", lo=0),
                max_dataset_bytes=_int(
                    s["max_dataset_bytes"], "console.limits.student.max_dataset_bytes", lo=1
                ),
                max_run_min=_int(s["max_run_min"], "console.limits.student.max_run_min", lo=1),
                run_now=_bool(s["run_now"], "console.limits.student.run_now"),
            ),
            admin_rate=rate,
            admin_rate_per_min=int(m.group(1)),
        ),
        retention=Retention(
            run_logs_per_student=_int(
                retention["run_logs_per_student"], "console.retention.run_logs_per_student", lo=1
            ),
            merged_export_days=_int(
                retention["merged_export_days"], "console.retention.merged_export_days", lo=1
            ),
            backup=BackupConfig(
                enabled=_bool(backup["enabled"], "console.retention.backup.enabled"),
                at=at,
                dest=_str(backup["dest"], "console.retention.backup.dest"),
            ),
        ),
    )


def validate_fleet(d: Any) -> FleetConfig:
    """Validate a raw fleet mapping -> :class:`FleetConfig` (pure)."""
    if not isinstance(d, dict):
        _fail("fleet config must be a mapping")
    _missing(d, ("version", "console", "nodes"), "fleet")
    version = d["version"]
    if version != FLEET_VERSION:
        _fail(f"version: must be {FLEET_VERSION} (got {version!r})")

    nodes_raw = d["nodes"]
    if not isinstance(nodes_raw, list) or not nodes_raw:
        _fail("nodes: must be a non-empty list")

    nodes = tuple(_node(n, f"nodes[{i}]") for i, n in enumerate(nodes_raw))
    seen: set[str] = set()
    for n in nodes:
        if n.id in seen:
            _fail(f"nodes: duplicate node id {n.id!r}")
        seen.add(n.id)

    return FleetConfig(version=version, console=_console(d["console"]), nodes=nodes)


def parse_fleet(text: str) -> FleetConfig:
    """YAML text -> FleetConfig. File I/O stays at the seam (DRY)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        _fail(f"fleet YAML is malformed: {e.__class__.__name__}")
    return validate_fleet(data)


# --- diff (audit feed for hot reload / admin edits) -----------------------------

_CONSOLE_LEAF = "console"
_NODE_SCOPES = ("label", "image", "workdir", "priority", "student_selectable", "enabled")


def _diff_nodes(old: FleetConfig, new: FleetConfig) -> list[Change]:
    changes: list[Change] = []
    old_by_id, new_by_id = old.nodes_by_id(), new.nodes_by_id()
    for nid in new_by_id:
        if nid not in old_by_id:
            changes.append(Change("nodes", "added", None, nid))
    for nid in old_by_id:
        if nid not in new_by_id:
            changes.append(Change("nodes", "removed", nid, None))
    for nid in old_by_id.keys() & new_by_id.keys():
        o, n = old_by_id[nid], new_by_id[nid]
        scope = f"nodes.{nid}"
        for field in _NODE_SCOPES:
            if getattr(o, field) != getattr(n, field):
                changes.append(Change(scope, field, getattr(o, field), getattr(n, field)))
        if o.gpu.quant != n.gpu.quant:
            changes.append(Change(f"{scope}.gpu", "quant", o.gpu.quant, n.gpu.quant))
        if o.gpu.vram_gb != n.gpu.vram_gb:
            changes.append(Change(f"{scope}.gpu", "vram_gb", o.gpu.vram_gb, n.gpu.vram_gb))
        if o.gpu.train_slots != n.gpu.train_slots:
            changes.append(Change(f"{scope}.gpu", "train_slots", o.gpu.train_slots, n.gpu.train_slots))
        if o.gpu.infer_slots != n.gpu.infer_slots:
            changes.append(Change(f"{scope}.gpu", "infer_slots", o.gpu.infer_slots, n.gpu.infer_slots))
        if o.gpu.budgets.train_gb != n.gpu.budgets.train_gb:
            changes.append(Change(f"{scope}.gpu", "train_gb", o.gpu.budgets.train_gb, n.gpu.budgets.train_gb))
        if o.gpu.budgets.infer_gb != n.gpu.budgets.infer_gb:
            changes.append(Change(f"{scope}.gpu", "infer_gb", o.gpu.budgets.infer_gb, n.gpu.budgets.infer_gb))
        if o.presets.batch != n.presets.batch:
            changes.append(Change(f"{scope}.presets", "batch", o.presets.batch, n.presets.batch))
        if o.agent != n.agent:
            changes.append(Change(f"{scope}.agent", "endpoint", o.agent, n.agent))
    return changes


def diff_fleet(old: FleetConfig, new: FleetConfig) -> list[Change]:
    """Flat before/after change list between two validated fleets.

    Consumed by the admin endpoints (audit log before→after JSON, §8) and by
    the hot-reload path (§10: validate → apply; queued jobs revalidated).
    """
    changes: list[Change] = []
    oc, nc = old.console, new.console
    scope = _CONSOLE_LEAF
    if oc.bind != nc.bind:
        changes.append(Change(scope, "bind", oc.bind, nc.bind))
    if oc.port != nc.port:
        changes.append(Change(scope, "port", oc.port, nc.port))
    if oc.tls != nc.tls:
        changes.append(Change(scope, "tls", oc.tls, nc.tls))
    if oc.data_dir != nc.data_dir:
        changes.append(Change(scope, "data_dir", oc.data_dir, nc.data_dir))
    if oc.session_ttl_h != nc.session_ttl_h:
        changes.append(Change(scope, "session_ttl_h", oc.session_ttl_h, nc.session_ttl_h))
    oss, nss = oc.limits.student, nc.limits.student
    for field in ("max_running", "max_queued", "max_dataset_bytes", "max_run_min", "run_now"):
        if getattr(oss, field) != getattr(nss, field):
            changes.append(Change(f"{scope}.limits.student", field, getattr(oss, field), getattr(nss, field)))
    if oc.limits.admin_rate != nc.limits.admin_rate:
        changes.append(Change(f"{scope}.limits", "admin_rate", oc.limits.admin_rate, nc.limits.admin_rate))
    orr, nrr = oc.retention, nc.retention
    if orr.run_logs_per_student != nrr.run_logs_per_student:
        changes.append(Change(f"{scope}.retention", "run_logs_per_student", orr.run_logs_per_student, nrr.run_logs_per_student))
    if orr.merged_export_days != nrr.merged_export_days:
        changes.append(Change(f"{scope}.retention", "merged_export_days", orr.merged_export_days, nrr.merged_export_days))
    for field in ("enabled", "at", "dest"):
        if getattr(orr.backup, field) != getattr(nrr.backup, field):
            changes.append(Change(f"{scope}.retention.backup", field, getattr(orr.backup, field), getattr(nrr.backup, field)))
    changes.extend(_diff_nodes(old, new))
    return changes
