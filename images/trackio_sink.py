"""trackio sink — the local stand-in for the Trackio surface lerobot-train
uses (SmolVLA family, the wandb-sink pattern of the crosscheck applied to
lerobot v0.7.0).

lerobot v0.7.0 ``lerobot-train`` logs its training curve through
``TrackioLogger`` (lerobot/common/trackio_utils.py), whose RAW module
surface is exactly:

    trackio.init(project=..., name=..., group=..., space_id=...,
                 private=..., config=..., resume=..., capture_console=...)
    trackio.log({"train/<key>": value, ...}, step=<loop step>)  # log_freq
    trackio.finish()
    trackio.Artifact(name, type=...) / trackio.log_artifact(a)
        (model artifacts — the wrapper passes --trackio.disable_artifact=
        true, so TrackioLogger returns BEFORE touching these; the sink
        keeps them as loud-never-crash no-ops in case the flag drifts)

The LAN policy is credential-free (the real trackio writes a local SQLite
run nobody on the class can see). The SmolVLA wrapper
(``images/smolvla_finetune.py``) therefore places this module FIRST on the
child's ``PYTHONPATH`` under the name ``trackio`` (image path
``/opt/robovla/sink/trackio.py``): every logged dict lands in the job's
STANDARD ``metrics.jsonl``, which the agent's incremental poll, the SSE
``metrics`` event and the run-detail sparkline already consume — one
metrics channel, no UI change (the crosscheck wandb-sink precedent).

Row mapping (the house ``contract.metrics_row`` vocabulary; lerobot logs
``MetricsTracker.to_dict()`` keys under the ``train/`` prefix):

    step       <- the pinned ``step=`` kwarg (lerobot loop step)
    total      <- ROBOVLA_METRICS_TOTAL (the wrapper's derived --steps)
    loss       <- train/loss
    lr         <- train/lr
    grad_norm  <- train/grad_norm
    vram_mb    <- train/gpu_mem_gb x 1024
    eta_s      <- (total - step) x train/step_s  (window-average step time)
    + every other finite-numeric train/* key passes through with the
      prefix stripped (steps, samples, epochs, data_s, prep_s, ...)
    + eval/* keys pass through likewise (held-out eval is off in v1; if a
      future wrapper enables --eval_steps the curve rides the same row).

Non-finite / non-numeric values are dropped. ``init`` and ``finish`` print
ONE line to stdout (the wrapper forwards the child's stdout into the job
log) — never into metrics.jsonl (the agent reads every JSON dict line
there as a metrics row). Env is read per call (the child's env is set at
spawn, after the module import). The env var NAMES match the crosscheck
wandb sink (one vocabulary; a CPU test pins the strings).

Fail-loud: unknown module attributes raise (PEP 562 ``__getattr__``) —
the pinned surface is init/log/finish (+ the artifact no-ops), so anything
else means a lerobot re-pin and this sink and its tests update together.
Stdlib only: the image COPYs this file under an aliased name; it must
stay import-independent.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping

#: Marker string (the real trackio package is shadowed, not used).
__version__ = "robovla-trackio-sink"

#: The env contract (the wrapper sets both; the sink reads per call).
ENV_SINK = "ROBOVLA_METRICS_SINK"    # the job's metrics.jsonl path
ENV_TOTAL = "ROBOVLA_METRICS_TOTAL"  # the derived --steps total

#: train/* key -> house metrics key (fixed renames; other numerics pass
#: through with the prefix stripped).
_RENAMES = {"grad_norm": "grad_norm", "gpu_mem_gb": "vram_mb"}

__all__ = ["init", "log", "finish", "Artifact", "log_artifact",
           "ENV_SINK", "ENV_TOTAL"]

_warned = set()


def _sink_path() -> str | None:
    """The configured sink file (None = disabled = a no-op log())."""
    return os.environ.get(ENV_SINK) or None


def _finite(value: Any) -> bool:
    """Finite int/float (bool excluded — it is an int subclass)."""
    return (isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value))


def _row(payload: Mapping[str, Any], step: int) -> dict[str, Any]:
    """One lerobot ``log`` payload -> a house metrics row.

    Lerobot logs every metric with a ``<mode>/`` prefix (TrackioLogger
    namespaces train vs eval); the sink strips it. Conflicts (a train/
    and an eval/ key of the same name in one call) keep the train value.
    """
    row: dict[str, Any] = {"step": step}
    total = os.environ.get(ENV_TOTAL)
    if total is not None and total.isdigit():
        row["total"] = int(total)
    mapped = 0.0
    for key, value in payload.items():
        if "/" in key:
            mode, name = key.split("/", 1)
        else:
            mode, name = "train", key
        if mode not in ("train", "eval") or not _finite(value):
            continue
        if name == "gpu_mem_gb":
            out, value = "vram_mb", float(value) * 1024.0
        else:
            out = _RENAMES.get(name, name)
        if mode == "eval" and out in row:
            continue
        row[out] = round(float(value), 6) if isinstance(value, float) else value
        if name == "step_s":
            mapped = float(value)
    if total is not None and total.isdigit() and mapped > 0:
        remaining = int(total) - step
        if remaining >= 0:
            row["eta_s"] = int(remaining * mapped)
    return row


def init(*args: Any, **kwargs: Any) -> "_Run":
    """The pinned ``trackio.init`` — announces to the job log only.

    Writes nothing to the sink file (see the module docstring). The
    return value must carry a ``.url`` attribute (TrackioLogger reads
    ``run.url`` to decide local-vs-Space reporting); local mode is None.
    """
    print("[trackio-sink] init: project="
          f"{kwargs.get('project')!r} name={kwargs.get('name')!r} "
          f"(sink: {_sink_path() or 'disabled'})", flush=True)
    return _Run()


def log(payload: Mapping[str, Any], step: int | None = None) -> None:
    """Capture one lerobot ``log`` call as a metrics row.

    ``step`` is required (TrackioLogger's ``log_dict`` always forwards
    its step kwarg to ``trackio.log``; a missing/non-int step means the
    pinned surface changed — fail loud rather than emit a stepless row
    the UI cannot plot). Unset sink env: a silent no-op.
    """
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError(
            f"robovla trackio sink: step must be an int (pinned lerobot "
            f"contract), got {step!r}")
    path = _sink_path()
    if path is None:
        return
    with open(path, "a") as f:
        f.write(json.dumps(_row(payload, step)) + "\n")
        f.flush()  # the agent polls incrementally; each row must be durable


def finish() -> None:
    """The pinned ``trackio.finish`` — nothing pending (rows flush per call)."""
    print("[trackio-sink] finish", flush=True)


class Artifact:
    """Model-artifact marker (the wrapper disables artifacts; a call here
    means the flag drifted — announce it, never kill a training run over
    a metrics-side upload)."""

    def __init__(self, name: str, type: str | None = None, **kwargs: Any):
        self.name, self.type = name, type

    def add_file(self, path: str, **kwargs: Any) -> None:
        _warn_once("Artifact.add_file")


def log_artifact(artifact: Any, **kwargs: Any) -> None:
    _warn_once("log_artifact")


def _warn_once(what: str) -> None:
    if what not in _warned:
        _warned.add(what)
        print(f"[trackio-sink] {what} called despite "
              "--trackio.disable_artifact=true — upload skipped "
              "(lerobot surface drift? update images/trackio_sink.py)",
              flush=True)


class _Run:
    """The ``init`` return marker: ``.url`` only (local mode = None)."""

    url = None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"robovla trackio sink: run.{name} is not part of the pinned "
            "lerobot TrackioLogger surface — update images/trackio_sink.py "
            "if the pinned lerobot's usage changed")


def __getattr__(name: str) -> Any:
    """PEP 562 module-level fail-loud: the surface is
    init/log/finish/Artifact/log_artifact (only)."""
    raise AttributeError(
        f"robovla trackio sink: attribute {name!r} is not part of the "
        "pinned lerobot TrackioLogger surface — update "
        "images/trackio_sink.py if the pinned lerobot's usage changed")
