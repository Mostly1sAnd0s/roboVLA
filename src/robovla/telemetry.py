"""Robot view: the RAM-only frame/action ring behind §9.10.

The robot piggybacks telemetry (last frame + last action) on its drive
loop; students watch THEIR robot live. Nothing here is persisted —
tonight's frames are teaching aids, not records (dataset truth already
lives in the recorder's uploads). One ring per student, newest last.
"""

from __future__ import annotations

import base64
import threading
import time
from collections import deque
from typing import Any

MAX_FRAME_BYTES = 2 * 1024 * 1024  # a VGA JPEG is ~8 KB; 2 MB is absurd-on-purpose


class RobotViewStore:
    def __init__(self, maxlen: int = 30) -> None:
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._rings: dict[str, deque[dict[str, Any]]] = {}
        self._seq = 0

    def put(self, student: str, frame_jpeg: bytes, action: list[float],
            latency_ms: float, prompt: str, slot_job: str) -> int:
        if len(frame_jpeg) > MAX_FRAME_BYTES:
            raise ValueError(f"telemetry frame too large ({len(frame_jpeg)} B)")
        with self._lock:
            self._seq += 1
            seq = self._seq
            ring = self._rings.setdefault(student, deque(maxlen=self._maxlen))
            ring.append({"seq": seq, "ts": time.time(), "jpeg": frame_jpeg,
                         "action": list(action), "latency_ms": latency_ms,
                         "prompt": prompt, "slot_job": slot_job})
            return seq

    def view(self, student: str) -> dict[str, Any] | None:
        """Newest sample + the recent action history (sparkline data)."""
        with self._lock:
            ring = self._rings.get(student)
            if not ring:
                return None
            newest = dict(ring[-1])
            hist = [{"seq": s["seq"], "action": s["action"],
                     "latency_ms": s["latency_ms"]} for s in ring]
        newest["frame_b64"] = base64.b64encode(newest.pop("jpeg")).decode()
        newest["history"] = hist
        return newest

    def students(self) -> list[str]:
        with self._lock:
            return sorted(self._rings)
