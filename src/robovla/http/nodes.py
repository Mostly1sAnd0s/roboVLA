"""/api/nodes — the training-endpoint board (§7.3/§8.2).

A student choosing where to train needs more than "is it online": measured
VRAM, both queues (jobs pinned to that node vs jobs submitted auto), the
slots in use, the model family the node actually runs, and the wait a job of
*their* shape would face there. `/api/health` stays the lean public view for
the classroom display (§8.5); this one is authenticated because it names the
students holding each slot.

The pick itself is still the submit-time `node_id` gate of §9.3 — this route
only tells the truth about why an endpoint is or is not selectable.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .. import nodes_ops
from .auth import ROLE_STUDENT, ROLE_TEACHER, ROLE_OBSERVER, require, svc

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


@router.get("")
async def board(request: Request, job_type: str = "train", batch: int | None = None,
                epochs: int | None = None,
                user: dict = Depends(require(ROLE_STUDENT, ROLE_TEACHER, ROLE_OBSERVER))) -> dict:
    """Per-node load + queue + the wait a job of this shape would face.

    `job_type`/`batch`/`epochs` are the caller's pending form values: the
    estimates are computed for THAT shape on each node, so the board and the
    submit form cannot disagree about what a run costs.
    """
    s = svc(request)
    return {"nodes": nodes_ops.endpoint_board(
        s, job_type=job_type, batch=batch, epochs=epochs,
        is_teacher=user["role"] == ROLE_TEACHER)}
