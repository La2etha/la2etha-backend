"""Event lifecycle + membership routes."""

import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.access.guard import require_event_host
from app.auth.users import current_active_user
from app.config import get_settings
from app.db.base import get_async_session
from app.db.models import Account, Event, GalleryEntry, Membership, Photo
from app.storage.base import get_storage
from app.schemas.event import (
    DemotedItem,
    EventCreate,
    EventCreated,
    EventJoin,
    EventRead,
    EventSettingsUpdate,
    MemberRead,
)

router = APIRouter(prefix="/events", tags=["events"])
settings = get_settings()


def _app_base_url() -> str:
    origins = settings.cors_origin_list
    return origins[0] if origins else "http://localhost:5173"


async def _unique_join_code(session: AsyncSession) -> str:
    for _ in range(10):
        code = secrets.token_hex(3).upper()  # 6 hex chars
        exists = await session.scalar(select(Event.id).where(Event.join_code == code))
        if not exists:
            return code
    raise HTTPException(status_code=500, detail="Could not allocate a join code")


@router.post("", response_model=EventCreated, status_code=status.HTTP_201_CREATED)
async def create_event(
    payload: EventCreate,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> EventCreated:
    event = Event(
        name=payload.name,
        owner_id=user.id,
        join_code=await _unique_join_code(session),
        join_token=secrets.token_urlsafe(24),
    )
    session.add(event)
    await session.flush()
    session.add(Membership(event_id=event.id, account_id=user.id, role="host"))
    await session.commit()
    await session.refresh(event)

    link = f"{_app_base_url()}/events/join?code={event.join_code}"
    return EventCreated(**EventRead.model_validate(event).model_dump(), join_link=link)


@router.post("/join", response_model=EventRead)
async def join_event(
    payload: EventJoin,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Event:
    event = await session.scalar(
        select(Event).where(Event.join_code == payload.join_code.upper())
    )
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    existing = await session.scalar(
        select(Membership.id).where(
            Membership.event_id == event.id, Membership.account_id == user.id
        )
    )
    if existing is None:
        session.add(Membership(event_id=event.id, account_id=user.id, role="member"))
        await session.commit()
    return event


async def _require_membership(
    session: AsyncSession, account_id: uuid.UUID, event_id: uuid.UUID
) -> Membership:
    membership = await session.scalar(
        select(Membership).where(
            Membership.event_id == event_id, Membership.account_id == account_id
        )
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return membership


@router.get("/{event_id}", response_model=EventRead)
async def get_event(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Event:
    await _require_membership(session, user.id, event_id)
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.get("/{event_id}/members", response_model=list[MemberRead])
async def list_members(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[Membership]:
    await require_event_host(session, user.id, event_id)
    members = await session.scalars(
        select(Membership).where(Membership.event_id == event_id)
    )
    return list(members)


@router.patch("/{event_id}/settings", response_model=EventRead)
async def update_settings(
    event_id: uuid.UUID,
    payload: EventSettingsUpdate,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Event:
    await require_event_host(session, user.id, event_id)
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if payload.privacy_default_remove_strangers is not None:
        event.privacy_default_remove_strangers = payload.privacy_default_remove_strangers
    await session.commit()
    await session.refresh(event)
    return event


@router.get("/{event_id}/demoted", response_model=list[DemotedItem])
async def list_demoted(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[DemotedItem]:
    """Host: photos demoted to members' secondary sections + why (F3/F4, FR-014)."""
    await require_event_host(session, user.id, event_id)
    rows = await session.execute(
        select(GalleryEntry.photo_id, GalleryEntry.account_id, GalleryEntry.demote_reason)
        .where(GalleryEntry.event_id == event_id, GalleryEntry.relevance == "low")
        .order_by(GalleryEntry.created_at)
    )
    return [
        DemotedItem(photo_id=p, account_id=a, demote_reason=r) for p, a, r in rows.all()
    ]


@router.post("/{event_id}/demoted/{photo_id}/promote", status_code=status.HTTP_204_NO_CONTENT)
async def promote_demoted(
    event_id: uuid.UUID,
    photo_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Host promotes a demoted photo back to the main gallery for everyone in it.

    Clears the photo's cull verdict too, so a later re-materialization does not
    demote it again.
    """
    await require_event_host(session, user.id, event_id)
    photo = await session.get(Photo, photo_id)
    if photo is None or photo.event_id != event_id:
        raise HTTPException(status_code=404, detail="Photo not found")

    photo.quality_verdict = "ok"
    photo.cull_reason = None
    await session.execute(
        update(GalleryEntry)
        .where(
            GalleryEntry.event_id == event_id,
            GalleryEntry.photo_id == photo_id,
            GalleryEntry.relevance == "low",
        )
        .values(relevance="main", demote_reason=None)
    )
    await session.commit()


@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    await require_event_host(session, user.id, event_id)
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    # Collect storage keys BEFORE the cascade removes the photo rows (FR-022).
    keys = list(
        await session.scalars(select(Photo.storage_key).where(Photo.event_id == event_id))
    )
    # FK cascades delete photos, faces, clusters, galleries, memberships.
    await session.delete(event)
    await session.commit()
    # Then purge the bytes. Done after the DB commit so a storage hiccup leaves
    # orphaned files (harmless, sweepable) rather than DB rows pointing at gone bytes.
    storage = get_storage()
    for key in keys:
        storage.delete(key)
