"""SSE event bus (design §9.4): per-job + class topics, Last-Event-ID resume.

Publish/subscribe in memory with a per-topic history ring so a reconnecting
client (``Last-Event-ID``) replays what it missed. The service publishes
``state``/``metrics``/``val``/``log``/``artifacts``/``predict`` events;
``/stream/{job_id}`` and ``/stream/class`` consume them.

Delivery uses stdlib ``queue.Queue`` (thread-safe) rather than
``asyncio.Queue``: the service may publish from the app loop (production
tick task) or from an external driver (tests drive ticks with
``asyncio.run``), while the SSE consumers await on the app loop — a
cross-loop ``put_nowait`` would not wake a consumer blocked in
``select()`` until its next timer fired (up to one heartbeat of lag).
A slow consumer whose 1000-deep queue fills drops new events (logged
behaviour, §9.4: consumers resume via Last-Event-ID).
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

HISTORY = 500
MAX_QUEUE = 1000

Event = tuple[int, str, dict]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[queue.Queue[Event]]] = {}
        self._hist: dict[str, list[Event]] = {}
        self._seq: dict[str, int] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, etype: str, data: dict[str, Any]) -> int:
        """Append to history + fan out to subscribers. Returns the seq."""
        with self._lock:
            seq = self._seq.get(topic, 0) + 1
            self._seq[topic] = seq
            hist = self._hist.setdefault(topic, [])
            hist.append((seq, etype, data))
            del hist[:-HISTORY]
            subs = list(self._subs.get(topic, ()))
        for q in subs:
            try:
                q.put_nowait((seq, etype, data))
            except queue.Full:  # dead/slow subscriber — it resumes by Last-Event-ID
                pass
        return seq

    def subscribe(self, topic: str) -> queue.Queue[Event]:
        q: queue.Queue[Event] = queue.Queue(maxsize=MAX_QUEUE)
        with self._lock:
            self._subs.setdefault(topic, []).append(q)
        return q

    def unsubscribe(self, topic: str, q: queue.Queue[Event]) -> None:
        with self._lock:
            subs = self._subs.get(topic, [])
            if q in subs:
                subs.remove(q)

    def replay(self, topic: str, last_id: int) -> list[Event]:
        """Events with id > last_id (resume, §9.4)."""
        with self._lock:
            return [e for e in self._hist.get(topic, ()) if e[0] > last_id]

    def render(self, seq: int, etype: str, data: dict[str, Any]) -> str:
        return f"id: {seq}\nevent: {etype}\ndata: {json.dumps(data, sort_keys=True)}\n\n"
