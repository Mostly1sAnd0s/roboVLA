"""VRAM budget calculator (design §3 — the budget guard).

``train_gb`` / ``infer_gb`` in fleet.yaml are the budgets at the node's
**max allowed batch**. Smaller batches scale the estimate down: training VRAM
splits into a batch-independent fixed part (quantized weights + LoRA
adapters/optimizer) and a batch-proportional variable part (activations):

    estimate = fixed[quant] + (base - fixed[quant]) * batch / max_batch

so a batch-1 job on the 32 GB node (base 18 GB at batch 8) needs ~11 GB, and
two concurrent train slots fit only under reduced-batch presets — while two
full-batch slots (36 GB) do not. Inference serves the model once, so it always
costs the node's ``infer_gb``.
"""

from __future__ import annotations

from .errors import RobovlaError

#: Batch-independent VRAM floor per quant profile (GB).
#:   bnb8: 8-bit 7.5B weights (~8 GB) + LoRA adapters/optimizer (~2 GB)
#:   bf16: 15 GB bf16 weights (design §1: "bf16 training doesn't fit")
FIXED_GB = {"bnb8": 10.0, "bf16": 15.0}

#: Quant profiles (fleet.yaml ``gpu.quant`` / JobSpec ``quant``) — single
#: definition, derived from the VRAM model above.
QUANT_PROFILES = frozenset(FIXED_GB)

#: Job types that consume a **train** slot and the scaled train budget.
TRAIN_JOB_TYPES = frozenset(
    {"smoke", "train", "validate", "merge", "crosscheck", "gpu_selftest", "home_selftest"}
)

#: Job types that consume an **infer** slot and the full ``infer_gb`` budget.
INFER_JOB_TYPES = frozenset({"infer"})


def _check_num(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise RobovlaError("validation", f"{name} must be a finite number")
    return float(value)


def estimate_train_gb(base_gb: float, quant: str, batch: int, max_batch: int) -> float:
    """Scaled train budget for ``batch`` given the node's base at ``max_batch``.

    ``batch >= max_batch`` costs the full ``base_gb``; smaller batches scale
    the variable part down, never below the fixed floor or above the base.
    """
    base = _check_num(base_gb, "base_gb")
    if base <= 0:
        raise RobovlaError("validation", "base_gb must be > 0")
    if quant not in FIXED_GB:
        raise RobovlaError("validation", f"unknown quant profile: {quant!r}")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise RobovlaError("validation", "batch must be an int >= 1")
    if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
        raise RobovlaError("validation", "max_batch must be an int >= 1")
    fixed = FIXED_GB[quant]
    if base < fixed:
        raise RobovlaError(
            "validation",
            f"train budget {base_gb:g} GB is below the {quant} model floor ({fixed:g} GB)",
        )
    if batch >= max_batch:
        return base
    scaled = fixed + (base - fixed) * (batch / max_batch)
    return min(max(scaled, fixed), base)


def estimate_job_gb(
    base_train_gb: float,
    base_infer_gb: float,
    quant: str,
    job_type: str,
    batch: int | None,
    max_batch: int,
) -> float:
    """Admission budget for one job on one node (GB).

    * ``infer`` jobs always cost ``base_infer_gb`` (the adapter is served once);
    * train-class jobs scale with ``batch`` against the node's ``max_batch``
      (the batch at which the base budget was measured — from the node's
      preset list in fleet.yaml);
    * ``batch=None`` (merge, selftest) costs the full train base — they load
      the whole model regardless of batch.
    """
    if job_type in INFER_JOB_TYPES:
        infer = _check_num(base_infer_gb, "base_infer_gb")
        if infer <= 0:
            raise RobovlaError("validation", "base_infer_gb must be > 0")
        return infer
    if job_type not in TRAIN_JOB_TYPES:
        raise RobovlaError("validation", f"unknown job type for budget: {job_type!r}")
    if batch is None:
        base = _check_num(base_train_gb, "base_train_gb")
        if base <= 0:
            raise RobovlaError("validation", "base_train_gb must be > 0")
        return base
    return estimate_train_gb(base_train_gb, quant, batch, max_batch)


def fits(free_gb: float, required_gb: float) -> bool:
    """The budget guard: free VRAM >= required (inclusive)."""
    free = _check_num(free_gb, "free_gb")
    required = _check_num(required_gb, "required_gb")
    return free >= required
