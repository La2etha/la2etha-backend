"""Personal gallery reads (the caller's verified photos only)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.db.base import get_async_session
from app.db.models import Account, GalleryClaim, GalleryEntry, Membership, Photo
from app.schemas.gallery import EmptyState, GalleryPage, GalleryPhoto
from app.services.curation import best_scores_for_account

router = APIRouter(tags=["gallery"])

DEFAULT_PAGE_SIZE = 60


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


@router.get("/events/{event_id}/gallery", response_model=GalleryPage)
async def get_gallery(
    event_id: uuid.UUID,
    cursor: str | None = Query(None),
    limit: int = Query(DEFAULT_PAGE_SIZE, le=200),
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> GalleryPage:
    """Return only the caller's verified photos (materialized GalleryEntries)."""
    await _require_membership(session, user.id, event_id)

    stmt = (
        select(GalleryEntry)
        .where(GalleryEntry.account_id == user.id, GalleryEntry.event_id == event_id)
        .order_by(GalleryEntry.created_at, GalleryEntry.id)
        .limit(limit + 1)
    )
    if cursor:
        stmt = stmt.where(GalleryEntry.id > uuid.UUID(cursor))

    entries = list(await session.scalars(stmt))
    next_cursor = None
    if len(entries) > limit:
        next_cursor = str(entries[limit - 1].id)
        entries = entries[:limit]

    photo_rows = {
        row.id: row
        for row in (
            await session.execute(
                select(
                    Photo.id, Photo.contributor_id, Photo.media_type, Photo.duration_s
                ).where(Photo.id.in_([e.photo_id for e in entries]))
            )
        ).all()
    }
    # Best-shot ranking (spec 004 R1) — main-relevance items only.
    main_ids = [e.photo_id for e in entries if e.relevance != "low"]
    best_scores = await best_scores_for_account(session, user.id, main_ids)
    items = [
        GalleryPhoto(
            photo_id=e.photo_id,
            origin=e.origin,
            relevance=e.relevance,
            demote_reason=e.demote_reason,
            confidence=e.confidence,
            contributor_id=photo_rows[e.photo_id].contributor_id,
            best_score=best_scores.get(e.photo_id),
            media_type=photo_rows[e.photo_id].media_type,
            duration_s=photo_rows[e.photo_id].duration_s,
        )
        for e in entries
    ]
    return GalleryPage(items=items, next_cursor=next_cursor)


async def set_claim(
    session: AsyncSession, account_id: uuid.UUID, photo: Photo, *, claimed: bool
) -> None:
    """Record a manual 'this is me' / 'not me' correction and sync the gallery
    (FR-018). ``claimed`` produces a GalleryEntry(origin='claim') granting access;
    unclaim removes the caller's entry (auto or claim) for the photo. The
    GalleryClaim row is the durable record — a lasting 'unclaimed' tombstone keeps
    auto re-materialization from resurrecting a photo the caller rejected."""
    state = "claimed" if claimed else "unclaimed"
    row = await session.scalar(
        select(GalleryClaim).where(
            GalleryClaim.account_id == account_id, GalleryClaim.photo_id == photo.id
        )
    )
    if row is None:
        session.add(GalleryClaim(account_id=account_id, photo_id=photo.id, state=state))
    else:
        row.state = state

    if claimed:
        exists = await session.scalar(
            select(GalleryEntry.id).where(
                GalleryEntry.account_id == account_id, GalleryEntry.photo_id == photo.id
            )
        )
        if exists is None:
            session.add(
                GalleryEntry(
                    account_id=account_id,
                    event_id=photo.event_id,
                    photo_id=photo.id,
                    origin="claim",
                    relevance="main",
                )
            )
    else:
        await session.execute(
            delete(GalleryEntry).where(
                GalleryEntry.account_id == account_id, GalleryEntry.photo_id == photo.id
            )
        )


async def _member_photo(
    session: AsyncSession, account_id: uuid.UUID, photo_id: uuid.UUID
) -> Photo:
    """Load a photo the caller may correct: it must exist and they must be a
    member of its event. (Unlike a photo READ, claiming a MISSED photo is allowed
    even when the caller isn't yet verified in it — that's the whole point.)"""
    photo = await session.get(Photo, photo_id)
    if photo is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    await _require_membership(session, account_id, photo.event_id)
    return photo


@router.post("/photos/{photo_id}/claim", status_code=status.HTTP_204_NO_CONTENT)
async def claim_photo(
    photo_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Claim a photo as containing you (FR-018) — grants access + adds it to your gallery."""
    photo = await _member_photo(session, user.id, photo_id)
    await set_claim(session, user.id, photo, claimed=True)
    await session.commit()


@router.delete("/photos/{photo_id}/claim", status_code=status.HTTP_204_NO_CONTENT)
async def unclaim_photo(
    photo_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Retract a photo from your gallery (FR-018) — removes it and revokes access."""
    photo = await _member_photo(session, user.id, photo_id)
    await set_claim(session, user.id, photo, claimed=False)
    await session.commit()


@router.get("/events/{event_id}/gallery/empty-state", response_model=EmptyState)
async def gallery_empty_state(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> EmptyState:
    await _require_membership(session, user.id, event_id)
    count = await session.scalar(
        select(GalleryEntry.id)
        .where(GalleryEntry.account_id == user.id, GalleryEntry.event_id == event_id)
        .limit(1)
    )
    if count is None:
        return EmptyState(
            empty=True,
            message="No photos of you yet. Enroll your face, or you may not appear in any photos.",
        )
    return EmptyState(empty=False, message="")
