"""Datasets (design §9.2): chunked upload -> finalize -> prep/QA gate,
versioning, norm stats, revalidate, frame previews. Student: own ·
teacher: all (ownership in the service -> ``forbidden``)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from .. import datasets_ops
from .auth import ROLE_STUDENT, ROLE_TEACHER, current_user, require, svc

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
_ROLES = (ROLE_STUDENT, ROLE_TEACHER)


def _ds_view(ds: dict) -> dict:
    return {
        "dataset_id": ds["dataset_id"], "owner": ds["owner"], "status": ds["status"],
        "total_bytes": ds["total_bytes"], "created_at": ds["created_at"],
        "updated_at": ds["updated_at"], "rejected_reason": ds.get("rejected_reason"),
        "episode_count": (ds.get("qa") or {}).get("summary", {}).get("episode_count"),
    }


@router.post("/upload")
async def upload(request: Request,
                 user: dict = Depends(require(*_ROLES)),
                 dataset_id: str = Form(...),
                 total_chunks: int = Form(...),
                 chunk_index: int = Form(...),
                 sha256: str = Form(...),
                 file: UploadFile = File(...)) -> dict:
    data = await file.read()
    return datasets_ops.upload_chunk(svc(request), user, dataset_id,
                                     total_chunks, chunk_index, sha256, data)


@router.post("/upload/finalize")
async def finalize(request: Request,
                   user: dict = Depends(require(*_ROLES))) -> dict:
    body = await request.json()
    report = datasets_ops.finalize_dataset(
        svc(request), user, body.get("dataset_id", ""),
        int(body.get("total_chunks", 0)), body.get("sha256", ""))
    return report


@router.get("")
async def list_datasets(request: Request,
                        user: dict = Depends(require(*_ROLES))) -> dict:
    s = svc(request)
    owner = None if user["role"] == ROLE_TEACHER else user["user_id"]
    return {"datasets": [_ds_view(d) for d in s.storage.list_datasets(owner=owner)]}




@router.get("/{dataset_id:path}/qa")
async def dataset_qa(dataset_id: str, request: Request,
                     user: dict = Depends(require(*_ROLES))) -> dict:
    ds = svc(request).own_dataset(user, dataset_id)
    if ds.get("qa") is None:
        from ..core.errors import RobovlaError
        raise RobovlaError("conflict", "QA report not ready (dataset not prepared)")
    return {"dataset_id": dataset_id, "qa": ds["qa"]}


@router.get("/{dataset_id:path}/norm-stats")
async def dataset_norm_stats(dataset_id: str, request: Request,
                             user: dict = Depends(require(*_ROLES))) -> dict:
    ds = svc(request).own_dataset(user, dataset_id)
    if ds.get("norm_stats") is None:
        from ..core.errors import RobovlaError
        raise RobovlaError("conflict", "norm stats not ready (dataset not prepared)")
    return {"dataset_id": dataset_id, "norm_stats": ds["norm_stats"]}


@router.get("/{dataset_id:path}")
async def get_dataset(dataset_id: str, request: Request,
                      user: dict = Depends(require(*_ROLES))) -> dict:
    ds = svc(request).own_dataset(user, dataset_id)
    view = _ds_view(ds)
    view["qa"] = ds.get("qa")
    return view


@router.post("/{dataset_id:path}/revalidate")
async def revalidate(dataset_id: str, request: Request,
                     user: dict = Depends(require(*_ROLES))) -> dict:
    return datasets_ops.revalidate_dataset(svc(request), user, dataset_id)


@router.get("/{dataset_id:path}/episodes/{episode:[^/]+}/frames/{index}.jpg")
async def frame(dataset_id: str, episode: str, index: int, request: Request,
                user: dict = Depends(require(*_ROLES))) -> FileResponse:
    path = datasets_ops.frame(svc(request), user, dataset_id, episode, index)
    return FileResponse(path, media_type="image/jpeg")
