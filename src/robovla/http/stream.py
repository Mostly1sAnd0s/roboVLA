"""SSE streams (design §9.4): /stream/{job_id} and /stream/class.

Events: ``state`` (queued|running|finishing|done|failed|killed + reason) ·
``metrics`` (JSONL rows) · ``val`` (held-out accuracy per checkpoint) ·
``log`` (tail lines) · ``artifacts`` · ``predict`` (smoke probe). Heartbeat
every 15 s; ``Last-Event-ID`` resume (history ring in the bus, 500 events).
"""

from __future__ import annotations

import asyncio
import json
import queue
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..sse import Event
from ..core.errors import RobovlaError
from .auth import ROLE_OBSERVER, ROLE_STUDENT, ROLE_TEACHER, current_user, svc

router = APIRouter(tags=["stream"])
_MEDIA = "text/event-stream"
_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive",
            "X-Accel-Buffering": "no"}
_POLL_S = 0.05  # consumer poll interval; worst-case event latency ~50 ms
_NO_EVENT = object()


def _last_event_id(request: Request) -> int:
    raw = request.headers.get("Last-Event-ID", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


async def _next(q: "queue.Queue[Event]", deadline: float):
    """Poll the thread-safe bus queue until an event or the heartbeat
    deadline (``_NO_EVENT``). Sleeps in small increments so the loop
    stays responsive and cancellation is prompt."""
    while True:
        try:
            return q.get_nowait()
        except queue.Empty:
            now = time.monotonic()
            if now >= deadline:
                return _NO_EVENT
            await asyncio.sleep(min(_POLL_S, deadline - now))


async def _stream(s, topic: str, last_id: int, heartbeat_s: float):
    q = s.bus.subscribe(topic)
    try:
        for seq, etype, data in s.bus.replay(topic, last_id):
            yield s.bus.render(seq, etype, data)
        next_heartbeat = time.monotonic() + heartbeat_s
        while True:
            item = await _next(q, next_heartbeat)
            if item is _NO_EVENT:
                yield ": heartbeat\n\n"
                next_heartbeat = time.monotonic() + heartbeat_s
            else:
                yield s.bus.render(*item)
    finally:
        s.bus.unsubscribe(topic, q)


@router.get("/stream/{job_id}")
async def stream_job(job_id: str, request: Request,
                     user: dict = Depends(current_user)) -> StreamingResponse:
    s = svc(request)
    row = s.storage.get_job(job_id)
    if row is None:
        raise RobovlaError("not_found", f"unknown job: {job_id}")
    if user["role"] not in (ROLE_TEACHER, ROLE_OBSERVER) \
            and row["spec"]["student"] != user["user_id"]:
        raise RobovlaError("forbidden", "not your run")
    return StreamingResponse(
        _stream(s, job_id, _last_event_id(request), s.heartbeat_s),
        media_type=_MEDIA, headers=_HEADERS)


@router.get("/stream/class")
async def stream_class(request: Request,
                       user: dict = Depends(current_user)) -> StreamingResponse:
    """Class-board feed: queue, node state, slots, activity — all roles,
    <= 1 msg/s (the service publishes only on change)."""
    s = svc(request)
    return StreamingResponse(
        _stream(s, "class", _last_event_id(request), max(s.heartbeat_s, 1.0)),
        media_type=_MEDIA, headers=_HEADERS)
