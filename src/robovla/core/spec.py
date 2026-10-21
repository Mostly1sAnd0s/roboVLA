"""The JobSpec contract — one schema, one validator, one preset catalog (design §11).

JobSpec (Appendix B) is the only artifact shared by UI, scheduler, runners,
docs, and the crosscheck job. Everything that shapes a job reads the preset
catalog defined here — the QLoRA default recipe (8-bit, rank 32, alpha 32,
lr 5e-4, aug on, ``max_seq_len`` fixed at 287) is defined exactly once, and
``build_job_spec`` routes its output through ``validate_job_spec`` so there is
a single path from form values to a valid spec (DRY).
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

from .budgets import INFER_JOB_TYPES, QUANT_PROFILES, TRAIN_JOB_TYPES
from .config import NODE_ID_RE
from .errors import RobovlaError

# --- the catalog -----------------------------------------------------------

#: Job types (Appendix B) — derived from the slot-class partition in
#: :mod:`robovla.core.budgets` (one source; the partition cannot drift).
JOB_TYPES = TRAIN_JOB_TYPES | INFER_JOB_TYPES

#: Official OpenVLA context: 256 visual tokens + prompt/action text.
#: Fixed — lives in the catalog, never in the student form (design §6).
MAX_SEQ_LEN = 287

#: Allowed student batch options for bnb8 nodes (design §3: 1 / 4 / 8).
ALLOWED_BATCHES = (1, 4, 8)

#: bf16 (home dev lab) batches; grad-accum is free there (no bnb recipe).
BF16_BATCHES = (1, 2, 4, 8)

#: The QLoRA recipe's effective batch — grad-accum auto-matches to it:
#: node max batch 1 => grad_accum 8; 4 => 2; 8 => 1 (setup doc §2).
RECIPE_EFFECTIVE_BATCH = 8

DEFAULT_MODEL = "openvla/openvla-7b"
DEFAULT_PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"
DEFAULT_LR = 5e-4
DEFAULT_EPOCHS = 3
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_SEED = 42

#: Default run cap per job type (minutes); the per-student cap is enforced
#: at submission by the service (design §8.1), this is the spec-level bound.
DEFAULT_MAX_MIN = {
    "smoke": 15,
    "train": 180,
    "validate": 60,
    "merge": 60,
    "crosscheck": 180,
    "infer": 720,
    "gpu_selftest": 30,
    "home_selftest": 30,
}

#: Keys the student "advanced" section may override; everything else (incl.
#: ``max_seq_len``) is catalog-fixed and rejected.
OVERRIDABLE = frozenset({"rank", "alpha", "lr", "epochs", "image_aug", "seed"})

# --- patterns ---------------------------------------------------------------

JOB_ID_RE = re.compile(r"^j_\d{8}_\d{4}$")
STUDENT_RE = re.compile(r"^s\d{2}$")
DATASET_RE = re.compile(r"^s\d{2}/[a-z0-9][a-z0-9-]*/v\d+$")
AGENT_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3}):(\d{1,5})$")

_TOP_LEVEL = (
    "job_id", "student", "type", "model", "dataset", "node", "quant", "lora",
    "optim", "image_aug", "seed", "prompt_template", "limits", "resume_from",
    "idempotency_key",
)


# --- dataclasses -------------------------------------------------------------


@dataclass(frozen=True)
class Lora:
    rank: int
    alpha: int


@dataclass(frozen=True)
class Optim:
    lr: float
    epochs: int
    batch: int
    grad_accum: int
    max_seq_len: int


@dataclass(frozen=True)
class NodeRef:
    id: str
    agent: str


@dataclass(frozen=True)
class Limits:
    max_min: int
    budget_gb: float


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    student: str
    type: str
    model: str
    dataset: str
    node: NodeRef
    quant: str
    lora: Lora
    optim: Optim
    image_aug: bool
    seed: int
    prompt_template: str
    limits: Limits
    resume_from: str | None
    idempotency_key: str | None

    def to_dict(self) -> dict[str, Any]:
        """Exact Appendix B JSON shape (the agent derives a fixed argv from it)."""
        return {
            "job_id": self.job_id,
            "student": self.student,
            "type": self.type,
            "model": self.model,
            "dataset": self.dataset,
            "node": {"id": self.node.id, "agent": self.node.agent},
            "quant": self.quant,
            "lora": {"rank": self.lora.rank, "alpha": self.lora.alpha},
            "optim": {
                "lr": self.optim.lr,
                "epochs": self.optim.epochs,
                "batch": self.optim.batch,
                "grad_accum": self.optim.grad_accum,
                "max_seq_len": self.optim.max_seq_len,
            },
            "image_aug": self.image_aug,
            "seed": self.seed,
            "prompt_template": self.prompt_template,
            "limits": {"max_min": self.limits.max_min, "budget_gb": self.limits.budget_gb},
            "resume_from": self.resume_from,
            "idempotency_key": self.idempotency_key,
        }


# --- validation helpers ------------------------------------------------------


def _fail(message: str) -> None:
    raise RobovlaError("validation", message)


def _str(value: Any, path: str, *, pattern: re.Pattern | None = None, max_len: int | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{path}: must be a non-empty string")
    if max_len is not None and len(value) > max_len:
        _fail(f"{path}: at most {max_len} characters")
    if pattern is not None and not pattern.fullmatch(value):
        _fail(f"{path}: {value!r} does not match the required format")
    return value


def _int(value: Any, path: str, *, lo: int, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path}: must be an integer")
    if value < lo or (hi is not None and value > hi):
        _fail(f"{path}: must be in [{lo}, {hi}]")
    return value


def _num(value: Any, path: str, *, lo: float, hi: float | None = None, exclusive_lo: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{path}: must be a finite number")
    v = float(value)
    if (v <= lo if exclusive_lo else v < lo) or (hi is not None and v > hi):
        _fail(f"{path}: out of bounds")
    return v


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{path}: must be a boolean")
    return value


def _dict(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        _fail(f"{path}: must be an object")
    return value


def _agent_endpoint(value: Any, path: str) -> str:
    m = AGENT_RE.fullmatch(_str(value, path)) if isinstance(value, str) else None
    if not m:
        _fail(f"{path}: must be 'ipv4:port' (see fleet.yaml agent.lan_ip + port)")
    octets = (int(m.group(i)) for i in range(1, 5))
    port = int(m.group(5))
    if any(o > 255 for o in octets) or not (1 <= port <= 65535):
        _fail(f"{path}: {value!r} has an invalid octet or port")
    return value


def _uuid(value: Any, path: str) -> str | None:
    if value is None:
        return None
    try:
        uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        _fail(f"{path}: must be a UUID string or null")
    return str(value)


def _nested(value: Any, path: str, keys: tuple[str, ...]) -> dict:
    d = _dict(value, path)
    missing = [k for k in keys if k not in d]
    if missing:
        _fail(f"{path}: missing key(s) {', '.join(missing)}")
    return d


# --- the validator -----------------------------------------------------------


def validate_job_spec(d: Any) -> JobSpec:
    """Validate a raw JobSpec dict (Appendix B) -> :class:`JobSpec`.

    Raises :class:`RobovlaError` with code ``validation`` and a field path in
    the message. Strict: unknown keys are rejected (typos fail, they do not
    silently vanish).
    """
    if not isinstance(d, dict):
        _fail("JobSpec must be a JSON object")
    missing = [k for k in _TOP_LEVEL if k not in d]
    if missing:
        _fail(f"JobSpec: missing key(s) {', '.join(missing)}")
    extra = [k for k in d if k not in _TOP_LEVEL]
    if extra:
        _fail(f"JobSpec: unknown key(s) {', '.join(sorted(extra))}")

    node = _nested(d["node"], "node", ("id", "agent"))
    lora = _nested(d["lora"], "lora", ("rank", "alpha"))
    optim = _nested(d["optim"], "optim", ("lr", "epochs", "batch", "grad_accum", "max_seq_len"))
    limits = _nested(d["limits"], "limits", ("max_min", "budget_gb"))

    if d["type"] not in JOB_TYPES:
        _fail(f"type: {d['type']!r} is not a known job type")
    if d["quant"] not in QUANT_PROFILES:
        _fail(f"quant: {d['quant']!r} is not a known quant profile")

    batch = _int(optim["batch"], "optim.batch", lo=1, hi=8)
    accum = _int(optim["grad_accum"], "optim.grad_accum", lo=1, hi=64)
    if d["quant"] == "bnb8":
        if batch not in ALLOWED_BATCHES:
            _fail(f"optim.batch: bnb8 batches are {ALLOWED_BATCHES}")
        expected = RECIPE_EFFECTIVE_BATCH // batch
        if accum != expected:
            _fail(f"optim.grad_accum: bnb8 batch {batch} requires grad_accum {expected}")
    elif batch not in BF16_BATCHES:
        _fail(f"optim.batch: bf16 batches are {BF16_BATCHES}")

    seq_len = _int(optim["max_seq_len"], "optim.max_seq_len", lo=1, hi=1024)
    if seq_len != MAX_SEQ_LEN:
        _fail(f"optim.max_seq_len: fixed at {MAX_SEQ_LEN} (official OpenVLA context)")

    resume = d["resume_from"]
    if resume is not None:
        resume = _str(resume, "resume_from", max_len=128)

    return JobSpec(
        job_id=_str(d["job_id"], "job_id", pattern=JOB_ID_RE),
        student=_str(d["student"], "student", pattern=STUDENT_RE),
        type=d["type"],
        model=_str(d["model"], "model", max_len=200),
        dataset=_str(d["dataset"], "dataset", pattern=DATASET_RE),
        node=NodeRef(
            id=_str(node["id"], "node.id", pattern=NODE_ID_RE),
            agent=_agent_endpoint(node["agent"], "node.agent"),
        ),
        quant=d["quant"],
        lora=Lora(
            rank=_int(lora["rank"], "lora.rank", lo=4, hi=64),
            alpha=_int(lora["alpha"], "lora.alpha", lo=4, hi=128),
        ),
        optim=Optim(
            lr=_num(optim["lr"], "optim.lr", lo=0.0, hi=0.01, exclusive_lo=True),
            epochs=_int(optim["epochs"], "optim.epochs", lo=1, hi=10),
            batch=batch,
            grad_accum=accum,
            max_seq_len=seq_len,
        ),
        image_aug=_bool(d["image_aug"], "image_aug"),
        seed=_int(d["seed"], "seed", lo=0, hi=2**31 - 1),
        prompt_template=_check_prompt(_str(d["prompt_template"], "prompt_template", max_len=1000)),
        limits=Limits(
            max_min=_int(limits["max_min"], "limits.max_min", lo=1, hi=1440),
            budget_gb=_num(limits["budget_gb"], "limits.budget_gb", lo=0.0, hi=256.0, exclusive_lo=True),
        ),
        resume_from=resume,
        idempotency_key=_uuid(d["idempotency_key"], "idempotency_key"),
    )


# --- the preset catalog --------------------------------------------------------


def _check_prompt(template: str) -> str:
    if "{task}" not in template:
        _fail("prompt_template: must contain the {task} slot (model-card format)")
    return template


def build_job_spec(
    *,
    job_id: str,
    student: str,
    job_type: str,
    dataset: str,
    node_id: str,
    node_agent: str,
    budget_gb: float,
    batch: int = 4,
    quant: str = "bnb8",
    seed: int = DEFAULT_SEED,
    model: str = DEFAULT_MODEL,
    overrides: dict[str, Any] | None = None,
    max_min: int | None = None,
    resume_from: str | None = None,
    idempotency_key: str | None = None,
) -> JobSpec:
    """Build a JobSpec from the preset catalog + (optional) advanced overrides.

    The "advanced" form reads this catalog — it cannot set catalog-fixed
    fields (``max_seq_len``, batch/grad-accum coupling, quant). The result is
    always run through :func:`validate_job_spec` (one validator, DRY).
    """
    if job_type not in JOB_TYPES:
        _fail(f"type: {job_type!r} is not a known job type")
    if quant not in QUANT_PROFILES:
        _fail(f"quant: {quant!r} is not a known quant profile")

    bad = [k for k in (overrides or {}) if k not in OVERRIDABLE]
    if bad:
        _fail(f"overrides: {', '.join(sorted(bad))} are catalog-fixed (max_seq_len, batch, grad_accum, quant)")

    if quant == "bnb8":
        if batch not in ALLOWED_BATCHES:
            _fail(f"batch: bnb8 batches are {ALLOWED_BATCHES}")
        grad_accum = RECIPE_EFFECTIVE_BATCH // batch
    else:
        if batch not in BF16_BATCHES:
            _fail(f"batch: bf16 batches are {BF16_BATCHES}")
        grad_accum = 1

    d = {
        "job_id": job_id,
        "student": student,
        "type": job_type,
        "model": model,
        "dataset": dataset,
        "node": {"id": node_id, "agent": node_agent},
        "quant": quant,
        "lora": {"rank": DEFAULT_LORA_RANK, "alpha": DEFAULT_LORA_ALPHA},
        "optim": {
            "lr": DEFAULT_LR,
            "epochs": DEFAULT_EPOCHS,
            "batch": batch,
            "grad_accum": grad_accum,
            "max_seq_len": MAX_SEQ_LEN,
        },
        "image_aug": True,
        "seed": seed,
        # The {task} slot stays in the spec (Appendix B); the agent substitutes
        # the dataset's task at runtime — one prompt format, model-card style.
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "limits": {
            "max_min": DEFAULT_MAX_MIN[job_type] if max_min is None else max_min,
            "budget_gb": budget_gb,
        },
        "resume_from": resume_from,
        "idempotency_key": idempotency_key,
    }
    if overrides:
        d["lora"]["rank"] = overrides.get("rank", d["lora"]["rank"])
        d["lora"]["alpha"] = overrides.get("alpha", d["lora"]["alpha"])
        d["optim"]["lr"] = overrides.get("lr", d["optim"]["lr"])
        d["optim"]["epochs"] = overrides.get("epochs", d["optim"]["epochs"])
        d["image_aug"] = overrides.get("image_aug", d["image_aug"])
        d["seed"] = overrides.get("seed", d["seed"])
    return validate_job_spec(d)
