"""Inference slots (§9.5) — the serving registry inside the agent.

A claimed slot **is** an infer job: the console dispatches it with
``serve(run_id, adapter)`` and the agent loads that run's adapter into the
serving model; ``release`` unloads it (VRAM freed, then probed by the
console). Robots never talk to the agent — the console proxies
``/api/robot/predict`` here after its own token + lease check (§9.5), so
the handle is opaque to the outside and the agent token is the only auth.

The model behind a slot is a seam (:class:`Model`): Phase 0 tests inject a
deterministic model; production injects :class:`TorchVlaModel` — the single
place the GPU stack pin lands (Phase 1). Predict is serialized per slot
(GPU is a shared resource; p95 < 400 ms target §9.5).
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Protocol
from PIL import Image

from robovla.core.errors import RobovlaError
from robovla.core.spec import DEFAULT_MODEL


class Model(Protocol):
    """One loaded adapter: one predict, one unload."""

    def predict(self, image_b64: str, prompt: str) -> list[float]:
        """One robot step -> 7-dim *normalized* action (the console
        un-normalizes with the dataset stats, §9.5)."""
        ...

    def unload(self) -> None:
        ...


class TorchVlaModel:
    """The real 8-bit serving model — the single place the GPU stack pin
    lands (design §6, Phase 1).

    Imports the stack lazily so the Phase-0 CPU suite never touches it.
    The load/predict call shapes follow the openvla repo's public API and
    are verified at ``gpu_selftest`` (test matrix §11) — this class is the
    one seam the pin touches; nothing else in the agent imports torch.
    """

    def __init__(self, model_id: str, adapter_path: str | Path) -> None:
        if not model_id:
            raise RobovlaError("validation", "model id is required")
        self._model_id = model_id
        self._adapter = Path(adapter_path)
        if not self._adapter.is_file():
            raise RobovlaError("not_found",
                               f"adapter not found on node: {self._adapter}")
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoProcessor
                from openvla import load_openvla  # pinned openvla repo (§6)
            except ImportError as e:
                raise RobovlaError(
                    "internal",
                    f"serving stack unavailable in this image: {getattr(e, 'name', e)}") from e
            quantization_config = (
                load_openvla.quantization_config
                if hasattr(load_openvla, "quantization_config") else None)
            base = load_openvla(self._model_id, quantization_config)
            self._model = PeftModel.from_pretrained(base, str(self._adapter))
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model.eval()
            self._torch = torch

    def predict(self, image_b64: str, prompt: str) -> list[float]:
        self._ensure_loaded()
        if not prompt:
            raise RobovlaError("validation", "prompt is required")
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as e:
            raise RobovlaError("validation", f"image is not a decodable JPEG: {e.__class__.__name__}") from e
        torch = self._torch
        with torch.inference_mode():
            inputs = self._processor(prompt, images=image).to(
                self._model.device, dtype=self._model.dtype)
            action = self._model.predict_action(**inputs)
        return [float(x) for x in action.reshape(-1)[:7]]

    def unload(self) -> None:
        with self._load_lock:
            self._model = None
            self._processor = None
            if self._torch is not None:
                try:
                    self._torch.cuda.empty_cache()
                except Exception:  # pragma: no cover - no-GPU path
                    pass


class SlotServing:
    """Slot handles over loaded models (the ``/serve`` + ``/predict`` core)."""

    def __init__(self, model_factory, *, log=None) -> None:
        if model_factory is None:
            raise RobovlaError("validation", "model_factory is required")
        self._factory = model_factory
        self._log = log or (lambda line: None)
        self._slots: dict[str, dict[str, Any]] = {}
        self._next = 0
        self._lock = threading.Lock()

    # -- slots -----------------------------------------------------------------

    def serve(self, run_id: str, adapter: str) -> str:
        """Load the run's adapter; returns the slot handle. Load failures
        raise (the console keeps the job queued and retries, §10)."""
        if not run_id:
            raise RobovlaError("validation", "run_id is required")
        if not adapter:
            raise RobovlaError("validation", "adapter path is required")
        model = self._factory(run_id, adapter)  # may raise (e.g. stack)
        with self._lock:
            self._next += 1
            handle = f"slot-{self._next}"
            self._slots[handle] = {
                "run_id": run_id, "adapter": adapter, "model": model,
                "lock": threading.Lock(), "created_at": time.time(),
            }
        self._log(f"[serve] {handle}: run={run_id} adapter={adapter}")
        return handle

    def release(self, handle: str) -> None:
        with self._lock:
            slot = self._slots.pop(handle, None)
        if slot is None:
            raise RobovlaError("not_found", f"unknown slot handle: {handle}")
        slot["model"].unload()
        self._log(f"[serve] {handle}: released (run={slot['run_id']})")

    def active(self) -> list[dict[str, Any]]:
        """Live slots (admin/health view): handle, run, adapter, age."""
        with self._lock:
            now = time.time()
            return [
                {"handle": h, "run_id": s["run_id"], "adapter": s["adapter"],
                 "age_s": int(now - s["created_at"])}
                for h, s in self._slots.items()
            ]

    # -- predict ------------------------------------------------------------------

    def predict(self, handle: str, image_b64: str, prompt: str) -> dict[str, Any]:
        """One robot step: {action_norm[7], latency_ms} (the console
        un-normalizes with the dataset stats, §9.5)."""
        with self._lock:
            slot = self._slots.get(handle)
        if slot is None:
            raise RobovlaError("not_found", f"no adapter loaded for handle {handle}")
        t0 = time.perf_counter()
        with slot["lock"]:  # serialize per slot (shared GPU)
            action = slot["model"].predict(image_b64, prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if (not isinstance(action, (list, tuple)) or len(action) != 7
                or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in action)):
            raise RobovlaError("internal",
                               "serving model returned an invalid 7-dim action")
        return {"action_norm": [float(x) for x in action],
                "latency_ms": round(latency_ms, 1)}


def make_default_model_factory(workdir: str | Path, *,
                               log=None):
    """Production factory: TorchVlaModel, model id from the run's spec if
    present on this node (else the catalog default), adapter resolved under
    the workdir."""
    def factory(run_id: str, adapter: str):
        model_id = DEFAULT_MODEL
        spec_path = Path(workdir) / "runs" / run_id / "spec.json"
        if spec_path.exists():
            try:
                model_id = json.loads(spec_path.read_text()).get("model") or model_id
            except (ValueError, OSError):
                pass
        return TorchVlaModel(model_id, Path(workdir) / adapter)
    return factory
