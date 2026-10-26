"""wandb sink — the local stand-in for the wandb surface the pinned official
finetune.py uses (design §6 crosscheck, the documented loss-sink follow-up).

The pinned openvla ``vla-scripts/finetune.py`` (SHA c8f03f48, read 2026-10-23)
touches wandb exactly twice, main process only:

    wandb.init(entity=..., project=..., name=...)
    wandb.log({"train_loss": ..., "action_accuracy": ..., "l1_loss": ...},
              step=gradient_step_idx)        # every 10 gradient steps

LAN policy is credential-free wandb; the real package under
``WANDB_MODE=disabled`` would make both calls no-ops and the official's
per-step loss would be lost. The crosscheck wrapper
(``images/crosscheck_finetune.py``) therefore places this module FIRST on the
child's ``PYTHONPATH`` under the name ``wandb`` (image path
``/opt/robovla/sink/wandb.py``), shadowing the pip-installed package: the
official's log stream lands in the job's STANDARD ``metrics.jsonl``, which
the agent's incremental poll, the SSE ``metrics`` event and the run-detail
sparkline already consume — one metrics channel, no second curve format.
The official process is otherwise untouched (the sink is a logging stand-in,
disclosed in the crosscheck results, like the build-time #90 patch).

Row mapping (the house key set — ``images/contract.py`` ``metrics_row``
vocabulary, so the crosscheck curve renders like a console-engine job's):
    loss    <- train_loss (the official's grad-accum-window smoothed value)
    step    <- the official ``step=`` kwarg (gradient step index)
    total   <- ``ROBOVLA_METRICS_TOTAL`` (the wrapper's derived max_steps)
    lr      <- ``ROBOVLA_METRICS_LR`` (the official's constant AdamW lr)
    + every other finite-numeric payload key passes through verbatim
      (action_accuracy, l1_loss — kept for the student curve comparison)

Non-finite / non-numeric payload values are dropped. The ``init`` event goes
to stdout (the wrapper forwards the child's stdout into the job log) — never
into ``metrics.jsonl``: the agent treats every JSON dict line there as a
metrics row, so an init header would corrupt the stream. Env is read per
call (the child's env is set at spawn, after the module import).

Fail-loud: unknown attributes raise (module PEP 562 ``__getattr__``) — the
pinned surface is init/log, so anything else means a re-pin and the sink +
its tests must be updated together. Stdlib only: this file is COPYed into
the image under an aliased name and must stay import-independent.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping

#: Marker string (the real wandb package is shadowed, not used).
__version__ = "robovla-wandb-sink"

#: The env contract (the wrapper sets all three; the sink reads per call).
ENV_SINK = "ROBOVLA_METRICS_SINK"    # the job's metrics.jsonl path
ENV_TOTAL = "ROBOVLA_METRICS_TOTAL"  # max_steps (the official's step total)
ENV_LR = "ROBOVLA_METRICS_LR"        # the official's constant learning rate

__all__ = ["init", "log", "ENV_SINK", "ENV_TOTAL", "ENV_LR"]


def _sink_path() -> str | None:
    """The configured sink file (None = disabled = a no-op log())."""
    return os.environ.get(ENV_SINK) or None


def _finite(value: Any) -> bool:
    """Finite int/float (bool excluded — it is an int subclass)."""
    return (isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value))


def _row(payload: Mapping[str, Any], step: int) -> dict[str, Any]:
    """The house metrics row from one official log payload."""
    row: dict[str, Any] = {"step": step}
    total = os.environ.get(ENV_TOTAL)
    if total is not None and total.isdigit():
        row["total"] = int(total)
    loss = payload.get("train_loss")
    if _finite(loss):
        row["loss"] = float(loss)
    lr = os.environ.get(ENV_LR)
    if lr is not None:
        row["lr"] = float(lr)
    for key, value in payload.items():
        if key != "train_loss" and _finite(value):
            row[key] = value
    return row


def init(*args: Any, **kwargs: Any) -> "_Run":
    """The pinned script's ``wandb.init`` — announces to the job log only.

    Writes nothing to the sink file (see the module docstring: a header
    line would be read as a metrics row by the agent's poll). The return
    value is ignored by the pinned script; it is a marker object whose
    attributes fail loud.
    """
    print("[wandb-sink] init: "
          + ", ".join(f"{k}={v}" for k, v in kwargs.items())
          + f" (sink: {_sink_path() or 'disabled'})", flush=True)
    return _Run()


def log(payload: Mapping[str, Any], step: int | None = None) -> None:
    """Capture one official ``wandb.log`` call as a metrics row.

    ``step`` is required (the pinned script always logs with
    ``step=gradient_step_idx``; a missing/non-int step means the contract
    changed — fail loud rather than emit a stepless row the UI cannot plot).
    Unset sink env: a silent no-op (the old WANDB_MODE=disabled behavior).
    """
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError(
            f"robovla wandb sink: step must be an int (pinned script "
            f"contract), got {step!r}")
    path = _sink_path()
    if path is None:
        return
    with open(path, "a") as f:
        f.write(json.dumps(_row(payload, step)) + "\n")
        f.flush()  # the agent polls incrementally; each row must be durable


class _Run:
    """The ``init`` return marker (the pinned script ignores it)."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"robovla wandb sink: run.{name} is not part of the pinned "
            "finetune.py wandb surface — update images/wandb_sink.py if the "
            "pinned script's usage changed")


def __getattr__(name: str) -> Any:
    """PEP 562 module-level fail-loud: the surface is init/log (only)."""
    raise AttributeError(
        f"robovla wandb sink: attribute {name!r} is not part of the pinned "
        "finetune.py wandb surface (init/log) — update images/wandb_sink.py "
        "if the pinned script's usage changed")
