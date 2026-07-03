"""Photo pooling: upload, processing status, access-guarded read, host pool."""

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.access.guard import require_event_host, require_photo_read
from app.auth.users import current_active_user
from app.cv.image import dhash
from app.db.base import get_async_session
from app.db.models import Account, Membership, Photo
from app.schemas.photo import (
    GDriveIngestAccepted,
    GDriveIngestRequest,
    PhotoRead,
    ProcessingStatus,
    UploadAccepted,
)
from app.storage.base import get_storage
from app.workers import get_queue, job_status

router = APIRouter(tags=["photos"])


def _content_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


async def _require_membership(
    session: AsyncSession, account_id: uuid.UUID, event_id: uuid.UUID
) -> None:
    member = await session.scalar(
        select(Membership.id).where(
            Membership.event_id == event_id, Membership.account_id == account_id
        )
    )
    if member is None:
        raise HTTPException(status_code=404, detail="Event not found")


@router.post(
    "/events/{event_id}/photos",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_photos(
    event_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> UploadAccepted:
    """Accept a batch of photos, collapse duplicates by phash (FR-006), store
    bytes, and enqueue background processing (202 — never blocks, SC-006)."""
    await _require_membership(session, user.id, event_id)
    storage = get_storage()

    photo_ids: list[uuid.UUID] = []
    duplicates = 0
    # Track phashes seen in this batch as well as those already in the event.
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        phash = dhash(data)
        clash = await session.scalar(
            select(Photo.id).where(Photo.event_id == event_id, Photo.phash == phash)
        )
        if clash is not None:
            duplicates += 1
            continue

        photo = Photo(
            event_id=event_id,
            contributor_id=user.id,
            storage_key="",  # set below once we have the id
            source="upload",
            phash=phash,
            processing_status="pending",
        )
        session.add(photo)
        await session.flush()
        key = f"events/{event_id}/{photo.id}"
        storage.put(key, data)
        photo.storage_key = key
        photo_ids.append(photo.id)

    await session.commit()

    job_id = ""
    if photo_ids:
        job = get_queue().enqueue(
            "app.workers.pipeline.process_photos",
            str(event_id),
            [str(pid) for pid in photo_ids],
        )
        job_id = job.id

    return UploadAccepted(
        job_id=job_id,
        photo_ids=photo_ids,
        accepted=len(photo_ids),
        duplicates=duplicates,
    )


@router.post(
    "/events/{event_id}/ingest/gdrive",
    response_model=GDriveIngestAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_gdrive(
    event_id: uuid.UUID,
    payload: GDriveIngestRequest,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> GDriveIngestAccepted:
    """Pull images from a Google Drive folder/files into the pool (background job)."""
    await _require_membership(session, user.id, event_id)
    if not payload.folder_id and not payload.file_ids:
        raise HTTPException(status_code=422, detail="Provide a folder_id or file_ids")

    job = get_queue().enqueue(
        "app.workers.pipeline.ingest_gdrive",
        str(event_id),
        str(user.id),
        payload.access_token,
        payload.file_ids,
        payload.folder_id,
    )
    return GDriveIngestAccepted(job_id=job.id)


@router.get("/events/{event_id}/photos/processing", response_model=ProcessingStatus)
async def processing_status(
    event_id: uuid.UUID,
    job_id: str = Query(...),
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> ProcessingStatus:
    await _require_membership(session, user.id, event_id)
    return ProcessingStatus(**job_status(job_id))


@router.get("/photos/{photo_id}")
async def read_photo(
    photo_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Access-guarded source image: gallery member (verified in it) or host."""
    photo = await require_photo_read(session, user.id, photo_id)
    data = get_storage().get(photo.storage_key)
    return Response(content=data, media_type=_content_type(data))


@router.get("/events/{event_id}/pool", response_model=list[PhotoRead])
async def host_pool(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[Photo]:
    """Host-only full pool view; members are forbidden (FR-019)."""
    await require_event_host(session, user.id, event_id)
    photos = await session.scalars(
        select(Photo).where(Photo.event_id == event_id).order_by(Photo.created_at)
    )
    return list(photos)
