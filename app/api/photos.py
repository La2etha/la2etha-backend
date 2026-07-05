"""Photo pooling: upload, processing status, access-guarded read, host pool."""

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.access.guard import is_event_host, require_event_host, require_photo_read
from app.auth.users import current_active_user
from app.config import get_settings
from app.cv.image import dhash
from app.db.base import get_async_session
from app.db.models import Account, DetectedFace, Event, FaceCluster, Membership, Photo
from app.services.curation import auto_pick_cover_photo_id
from app.schemas.photo import (
    GDriveIngestAccepted,
    GDriveIngestRequest,
    PhotoFace,
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
            Membership.event_id == event_id,
            Membership.account_id == account_id,
            Membership.status == "active",
        )
    )
    if member is None:
        raise HTTPException(status_code=404, detail="Event not found")


async def _require_open_for_uploads(
    session: AsyncSession, account_id: uuid.UUID, event_id: uuid.UUID
) -> Event:
    """Spec 005 US5: gate uploads/enrollment on the event's toggles. Returns the
    event so callers can reuse it."""
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.status == "archived":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This event is archived — no new uploads or enrollments.",
        )
    if event.member_uploads == "host_only" and not await is_event_host(
        session, account_id, event_id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The host is managing photos for this event.",
        )
    return event


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
    await _require_open_for_uploads(session, user.id, event_id)
    storage = get_storage()
    max_bytes = get_settings().max_upload_bytes

    photo_ids: list[uuid.UUID] = []
    duplicates = 0
    # Track phashes seen in this batch as well as those already in the event.
    for upload in files:
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only image files can be uploaded.",
            )
        # Reject oversize before reading it fully into memory when the size is known.
        if upload.size is not None and upload.size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="A file exceeds the maximum upload size.",
            )
        data = await upload.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="A file exceeds the maximum upload size.",
            )
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
    await _require_open_for_uploads(session, user.id, event_id)
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


@router.get("/photos/{photo_id}/faces", response_model=list[PhotoFace])
async def read_photo_faces(
    photo_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[PhotoFace]:
    """Detected-face boxes for a photo the caller may view, each flagged `is_me`
    when its cluster is claimed by the caller — powers the trust overlay (FR-024).

    ``name`` (spec 005 FR-001/002) is populated only when the event's name_policy
    permits it for this viewer; the gate is server-side so a policy-hidden name
    is indistinguishable from an unclaimed guest — never a client-side hide."""
    photo = await require_photo_read(session, user.id, photo_id)
    if not photo.width or not photo.height:
        return []  # not yet processed → no boxes to normalize

    event = await session.get(Event, photo.event_id)
    caller_is_host = await is_event_host(session, user.id, photo.event_id)
    names_allowed = event is not None and (
        event.name_policy == "everyone" or (event.name_policy == "host_only" and caller_is_host)
    )

    rows = (
        await session.execute(
            select(DetectedFace.bbox, FaceCluster.claimed_by_account_id, Account.name)
            .outerjoin(FaceCluster, FaceCluster.id == DetectedFace.cluster_id)
            .outerjoin(Account, Account.id == FaceCluster.claimed_by_account_id)
            .where(DetectedFace.photo_id == photo_id)
        )
    ).all()

    w, h = float(photo.width), float(photo.height)
    faces = []
    for bbox, claimed_by, claimed_name in rows:
        is_me = claimed_by == user.id
        name = None
        if not is_me and claimed_by is not None and names_allowed:
            name = claimed_name
        faces.append(
            PhotoFace(
                x=bbox["x"] / w,
                y=bbox["y"] / h,
                w=bbox["w"] / w,
                h=bbox["h"] / h,
                is_me=is_me,
                name=name,
            )
        )
    return faces


@router.get("/events/{event_id}/pool", response_model=list[PhotoRead])
async def host_pool(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[Photo]:
    """Full pool view. Host-only by default (FR-019); widened to every member
    when the host sets gallery_visibility=everyone_sees_all (spec 005 US5) —
    browsing is visibility only, per-photo actions keep their own guards."""
    if await is_event_host(session, user.id, event_id):
        pass
    else:
        await _require_membership(session, user.id, event_id)
        event = await session.get(Event, event_id)
        if event is None or event.gallery_visibility != "everyone_sees_all":
            raise HTTPException(status_code=403, detail="Host access required")
    photos = await session.scalars(
        select(Photo).where(Photo.event_id == event_id).order_by(Photo.created_at)
    )
    return list(photos)


@router.delete("/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_photo(
    photo_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Delete a pool photo: the host always may; the uploader may when the event
    allows member_delete_own (spec 005 US5/FR-018). Cascades storage bytes plus
    every FK-dependent row (faces, gallery entries/claims)."""
    photo = await session.get(Photo, photo_id)
    if photo is None:
        raise HTTPException(status_code=404, detail="Photo not found")

    caller_is_host = await is_event_host(session, user.id, photo.event_id)
    if not caller_is_host:
        event = await session.get(Event, photo.event_id)
        is_uploader = photo.contributor_id == user.id
        if not (is_uploader and event is not None and event.member_delete_own):
            raise HTTPException(status_code=403, detail="You can't delete this photo.")

    key = photo.storage_key
    event_id = photo.event_id
    was_cover = (await session.get(Event, event_id)).cover_photo_id == photo.id
    await session.delete(photo)
    await session.commit()
    get_storage().delete(key)

    if was_cover:
        # The FK (ondelete=set null) already cleared cover_photo_id; re-pick
        # immediately rather than leaving the event with no cover (R6). The
        # deleted photo can no longer be a "host choice", so this reverts to auto.
        event = await session.get(Event, event_id)
        if event is not None:
            picked = await auto_pick_cover_photo_id(session, event_id)
            event.cover_photo_id = picked
            event.cover_source = "auto" if picked else None
            await session.commit()
