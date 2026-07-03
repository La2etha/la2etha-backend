"""Access-isolation guard (FR-019) — the single choke point for photo reads.

A caller may read a photo IFF they have a GalleryEntry for it (their verified
identity is present in it) OR they are the event host. Unauthorized reads return
404 (not 403) so the API never reveals that a photo the caller can't access
exists (contract: avoid leaking presence).
"""

import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import GalleryEntry, Membership, Photo

_PHOTO_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found"
)


async def is_event_host(
    session: AsyncSession, account_id: uuid.UUID, event_id: uuid.UUID
) -> bool:
    host = await session.scalar(
        select(Membership.id).where(
            Membership.event_id == event_id,
            Membership.account_id == account_id,
            Membership.role == "host",
        )
    )
    return host is not None


async def require_photo_read(
    session: AsyncSession, account_id: uuid.UUID, photo_id: uuid.UUID
) -> Photo:
    """Return the Photo if the caller may read it, else raise 404."""
    photo = await session.get(Photo, photo_id)
    if photo is None:
        raise _PHOTO_NOT_FOUND

    if await is_event_host(session, account_id, photo.event_id):
        return photo

    entry = await session.scalar(
        select(GalleryEntry.id).where(
            GalleryEntry.account_id == account_id,
            GalleryEntry.photo_id == photo_id,
        )
    )
    if entry is not None:
        return photo

    raise _PHOTO_NOT_FOUND


async def require_event_host(
    session: AsyncSession, account_id: uuid.UUID, event_id: uuid.UUID
) -> None:
    """Raise 403 if the caller is not the host of ``event_id`` (host-only routes)."""
    if not await is_event_host(session, account_id, event_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Host access required"
        )
