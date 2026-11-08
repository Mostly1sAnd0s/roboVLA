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

import numpy as np
from PIL import Image

from robovla.core.errors import RobovlaError
from robovla.core.spec import DEFAULT_MODEL, MODEL_FAMILY_SMOLVLA, model_family
from robovla.smolvla import (SERVING_META_NAME, SMOLVLA_CAMERA_KEY,
                             SMOLVLA_STATE_DIM, SMOLVLA_STATE_KEY,
                             parse_serving_meta, to_house_action)


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


class SmolVlaModel:
    """The SmolVLA serving model — the family's half of the ``Model`` seam.

    Loads a trained policy directory (the wrapper's artifact: lerobot's
    ``pretrained_model/`` — config + weights + saved processor states +
    the stamped ``serving.json``, ``robovla.smolvla``). The GPU stack is
    imported lazily (the §11 boundary; the same rule as TorchVlaModel).

    Chunking (SmolVLA's own answer to the OpenVLA latency rebase): the
    policy's ``select_action`` caches a 50-step action chunk internally,
    so a 10 Hz robot consumes fresh actions and the GPU recomputes every
    ~5 s — one predict per robot request, the §9.5 API unchanged. The
    ``prompt`` is the RAW task string (the chat processor owns the
    format; the openvla In:/Out: wrap does not apply here).

    ``loader`` is the stack seam (CPU tests inject a deterministic fake;
    production gets ``_load_smolvla_stack``).
    """

    def __init__(self, model_id: str, adapter_path: str | Path,
                 *, loader=None) -> None:
        if not model_id:
            raise RobovlaError("validation", "model id is required")
        self._model_id = model_id
        self._adapter = Path(adapter_path)
        if not self._adapter.is_dir():
            raise RobovlaError("not_found",
                               f"policy dir not found on node: {self._adapter}")
        meta = self._adapter / SERVING_META_NAME
        if not meta.is_file():
            raise RobovlaError(
                "not_found",
                f"policy dir {self._adapter} carries no {SERVING_META_NAME} — "
                "not a servable smolvla artifact (train with the smolvla "
                "wrapper; the stamp lands on a clean exit)")
        try:
            self._stats = parse_serving_meta(json.loads(meta.read_text()))
        except (ValueError, OSError) as e:
            raise RobovlaError(
                "validation", f"unreadable {SERVING_META_NAME}: {e}") from e
        self._loader = loader or _load_smolvla_stack
        self._bundle = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self._bundle is None:
            with self._load_lock:
                if self._bundle is None:
                    try:
                        self._bundle = self._loader(self._adapter)
                    except RobovlaError:
                        raise
                    except ImportError as e:
                        raise RobovlaError(
                            "internal",
                            f"serving stack unavailable in this image: "
                            f"{getattr(e, 'name', e)}") from e
        return self._bundle

    def predict(self, image_b64: str, prompt: str) -> list[float]:
        policy, preprocess, postprocess, torch = self._ensure_loaded()
        if not prompt:
            raise RobovlaError("validation", "prompt is required")
        try:
            img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as e:
            raise RobovlaError(
                "validation",
                f"image is not a decodable JPEG: {e.__class__.__name__}") from e
        frame = {
            # lerobot convention: CHW float [0, 1] (what the dataset
            # reader hands the trainer's preprocessor at train time).
            SMOLVLA_CAMERA_KEY: torch.from_numpy(
                np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0),
            # the same constant-zero placeholder state the converter
            # trained with (prepare_state requires the key)
            SMOLVLA_STATE_KEY: np.zeros(SMOLVLA_STATE_DIM, dtype=np.float32),
            "task": prompt,
        }
        with torch.inference_mode():
            batch = preprocess(frame)
            # the single-frame preprocess batches images + language but
            # passes the state through UNBATCHED (probe, 2026-11-08:
            # (3,) reaches embed_prefix and dies on shape[1]); the model
            # contract is (B, state_dim) — batch it here, shape-guarded
            st = batch.get(SMOLVLA_STATE_KEY)
            if getattr(st, "ndim", 2) == 1:
                batch[SMOLVLA_STATE_KEY] = st.reshape((1, -1))
            action = postprocess(policy.select_action(batch))
        raw = [float(x) for x in torch.asarray(action).reshape(-1)[:3]]
        return list(to_house_action(raw, self._stats))

    def unload(self) -> None:
        with self._load_lock:
            self._bundle = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - no-GPU path
                pass


def _load_smolvla_stack(adapter_dir: Path):
    """The real stack load (lazy; the only place the smolvla image's lerobot
    pin is touched at serve time). Returns
    ``(policy, preprocess, postprocess, torch)``."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(str(adapter_dir))
    preprocess, postprocess = make_pre_post_processors(
        policy.config, str(adapter_dir),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    policy.eval()
    return policy, preprocess, postprocess, torch


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
    """Production factory, routed by model FAMILY (``core.spec.model_family``
    over the run's spec.json model id — the same derivation the agent argv
    uses, §11 DRY): TorchVlaModel for the openvla engine path,
    SmolVlaModel for a lerobot-trained policy dir (the spec.json is
    optional — without it the openvla catalog default stands, the
    pre-family contract unchanged)."""
    def factory(run_id: str, adapter: str):
        model_id = DEFAULT_MODEL
        spec_path = Path(workdir) / "runs" / run_id / "spec.json"
        if spec_path.exists():
            try:
                model_id = json.loads(spec_path.read_text()).get("model") or model_id
            except (ValueError, OSError):
                pass
        path = Path(workdir) / adapter
        if model_family(model_id) == MODEL_FAMILY_SMOLVLA:
            return SmolVlaModel(model_id, path)
        return TorchVlaModel(model_id, path)
    return factory
