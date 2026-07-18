"""Multi-angle enrollment (F2) and own-identity deletion (FR-022)."""

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from rq.job import Job
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.cv.video import probe
from app.db.base import get_async_session
from app.db.models import Account, Event, FaceCluster, GalleryEntry, IdentityEnrollment, Membership
from app.schemas.enrollment import EnrollmentAccepted, EnrollmentStatus
from app.storage.base import get_storage
from app.workers import get_queue, get_redis

router = APIRouter(tags=["enrollment"])

MIN_SAMPLES = 3
MAX_SAMPLES = 8
MAX_ENROLL_VIDEO_S = 5.0


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


async def _require_not_archived(session: AsyncSession, event_id: uuid.UUID) -> None:
    """Spec 005 US5 uploads_closed toggle: an archived event accepts no new
    enrollments (galleries/search/export keep working)."""
    event = await session.get(Event, event_id)
    if event is not None and event.status == "archived":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This event is archived — no new enrollments.",
        )


async def _enroll_video(
    event_id: uuid.UUID, upload: UploadFile, user: Account
) -> EnrollmentAccepted:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=422, detail="No usable enrollment video")
    result = probe(data)
    if not result.ok or (result.duration_s or 0) > MAX_ENROLL_VIDEO_S:
        raise HTTPException(
            status_code=422,
            detail=f"Enrollment video must be a decodable clip up to {MAX_ENROLL_VIDEO_S:.0f}s",
        )
    storage = get_storage()
    key = f"enroll/{event_id}/{user.id}/{uuid.uuid4()}.mp4"
    storage.put(key, data)
    job = get_queue().enqueue(
        "app.workers.pipeline.process_video_enrollment",
        str(user.id),
        str(event_id),
        key,
    )
    return EnrollmentAccepted(job_id=job.id, sample_count=0)


@router.post(
    "/events/{event_id}/enroll",
    response_model=EnrollmentAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enroll(
    event_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> EnrollmentAccepted:
    """Upload 3–5 multi-angle photos, or one short selfie video → build an
    identity centroid in the worker."""
    await _require_membership(session, user.id, event_id)
    await _require_not_archived(session, event_id)

    if len(files) == 1 and (files[0].content_type or "").startswith("video/"):
        return await _enroll_video(event_id, files[0], user)

    if not (MIN_SAMPLES <= len(files) <= MAX_SAMPLES):
        raise HTTPException(
            status_code=422,
            detail=f"Provide {MIN_SAMPLES}–{MAX_SAMPLES} enrollment photos",
        )

    storage = get_storage()
    sample_keys: list[str] = []
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        key = f"enroll/{event_id}/{user.id}/{uuid.uuid4()}"
        storage.put(key, data)
        sample_keys.append(key)

    if not sample_keys:
        raise HTTPException(status_code=422, detail="No usable enrollment images")

    job = get_queue().enqueue(
        "app.workers.pipeline.process_enrollment",
        str(user.id),
        str(event_id),
        sample_keys,
    )
    return EnrollmentAccepted(job_id=job.id, sample_count=len(sample_keys))


@router.get("/events/{event_id}/enroll/status", response_model=EnrollmentStatus)
async def enroll_status(
    event_id: uuid.UUID,
    job_id: str = Query(...),
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> EnrollmentStatus:
    await _require_membership(session, user.id, event_id)
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:
        return EnrollmentStatus(job_id=job_id, status="unknown")

    job_status = job.get_status(refresh=True)
    result = job.return_value() if job.is_finished else None
    payload = {"job_id": job_id, "status": job_status}
    if isinstance(result, dict):
        payload.update(
            {k: result.get(k) for k in ("enrolled", "quality_ok", "sample_count", "gallery_entries", "reason")}
        )

    # Source-of-truth fallback. The RQ return value expires (500s default) and a
    # job that crashed (e.g. a transient DB/Redis blip) has no return value at all
    # — in both cases the caller is actually enrolled, yet the raw job signal would
    # read as a face-detection failure and the client shows "no face detected". So
    # once the job is no longer running, if we don't have a clear enrolled=True,
    # trust the persisted enrollment: if a centroid exists for this account+event,
    # the user IS enrolled regardless of what the ephemeral job says.
    if job_status in ("finished", "failed", "unknown") and not payload.get("enrolled"):
        enrollment = await session.scalar(
            select(IdentityEnrollment).where(
                IdentityEnrollment.account_id == user.id,
                IdentityEnrollment.event_id == event_id,
            )
        )
        if enrollment is not None:
            gallery_count = await session.scalar(
                select(func.count())
                .select_from(GalleryEntry)
                .where(
                    GalleryEntry.account_id == user.id,
                    GalleryEntry.event_id == event_id,
                )
            )
            payload["status"] = "finished"
            payload["enrolled"] = True
            payload["quality_ok"] = enrollment.quality_ok
            payload["sample_count"] = enrollment.sample_count
            payload["gallery_entries"] = int(gallery_count or 0)
            payload["reason"] = None

    return EnrollmentStatus(**payload)


@router.delete("/users/me/identity", status_code=status.HTTP_204_NO_CONTENT)
async def delete_own_identity(
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Remove the caller's enrollments + gallery entries and unlink any clusters
    they claimed (FR-022)."""
    await session.execute(
        update(FaceCluster)
        .where(FaceCluster.claimed_by_account_id == user.id)
        .values(claimed_by_account_id=None)
    )
    await session.execute(delete(GalleryEntry).where(GalleryEntry.account_id == user.id))
    await session.execute(
        delete(IdentityEnrollment).where(IdentityEnrollment.account_id == user.id)
    )
    await session.commit()
