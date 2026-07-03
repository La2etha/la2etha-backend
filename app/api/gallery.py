"""Personal gallery reads (the caller's verified photos only)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.db.base import get_async_session
from app.db.models import Account, GalleryEntry, Membership
from app.schemas.gallery import EmptyState, GalleryPage, GalleryPhoto

router = APIRouter(tags=["gallery"])

DEFAULT_PAGE_SIZE = 60


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

    items = [
        GalleryPhoto(photo_id=e.photo_id, origin=e.origin, confidence=e.confidence)
        for e in entries
    ]
    return GalleryPage(items=items, next_cursor=next_cursor)


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
