"""Event lifecycle + membership routes."""

import secrets
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.access.guard import require_event_host
from app.api.photos import _content_type
from app.auth.users import current_active_user
from app.config import get_settings
from app.db.base import get_async_session
from app.db.models import Account, Event, GalleryEntry, Membership, Photo
from app.services.curation import auto_pick_cover_photo_id, event_stats
from app.services.membership import CannotRemoveHostError, list_members, remove_member
from app.storage.base import get_storage
from app.schemas.event import (
    DemotedItem,
    EventCreate,
    EventCreated,
    EventJoin,
    EventJoined,
    EventListItem,
    EventRead,
    EventSettingsUpdate,
    EventStats,
    HighlightItem,
    MemberRead,
    SetCover,
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
        event_type=payload.event_type,
    )
    session.add(event)
    await session.flush()
    session.add(Membership(event_id=event.id, account_id=user.id, role="host"))
    await session.commit()
    await session.refresh(event)

    link = f"{_app_base_url()}/events/join?code={event.join_code}"
    return EventCreated(**EventRead.model_validate(event).model_dump(), join_link=link)


@router.get("", response_model=list[EventListItem])
async def list_events(
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[EventListItem]:
    """Events the caller belongs to, newest first, with their role + light counts
    for the home-screen ticket stubs."""
    member_count = (
        select(func.count(Membership.id))
        .where(Membership.event_id == Event.id)
        .correlate(Event)
        .scalar_subquery()
    )
    photo_count = (
        select(func.count(Photo.id))
        .where(Photo.event_id == Event.id)
        .correlate(Event)
        .scalar_subquery()
    )
    rows = await session.execute(
        select(Event, Membership.role, member_count, photo_count)
        .join(Membership, Membership.event_id == Event.id)
        .where(Membership.account_id == user.id, Membership.status == "active")
        .order_by(Event.created_at.desc())
    )
    return [
        EventListItem(
            **EventRead.model_validate(event).model_dump(),
            role=role,
            member_count=members,
            photo_count=photos,
        )
        for event, role, members, photos in rows.all()
    ]


@router.post("/join", response_model=EventJoined)
async def join_event(
    payload: EventJoin,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> EventJoined:
    event = await session.scalar(
        select(Event).where(Event.join_code == payload.join_code.upper())
    )
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    existing = await session.scalar(
        select(Membership).where(
            Membership.event_id == event.id, Membership.account_id == user.id
        )
    )
    if existing is None:
        # join_approval (spec 005 US5): new members wait for the host instead of
        # joining instantly. Pending members have zero content access — every
        # guard requires status == "active".
        member_status = "pending" if event.join_approval else "active"
        session.add(
            Membership(
                event_id=event.id, account_id=user.id, role="member", status=member_status
            )
        )
        await session.commit()
        existing_status = member_status
    else:
        existing_status = existing.status

    if existing_status == "pending":
        return EventJoined(status="pending", event=None)
    return EventJoined(status="active", event=EventRead.model_validate(event))


async def _require_membership(
    session: AsyncSession, account_id: uuid.UUID, event_id: uuid.UUID
) -> Membership:
    membership = await session.scalar(
        select(Membership).where(
            Membership.event_id == event_id,
            Membership.account_id == account_id,
            Membership.status == "active",
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
async def get_members(
    event_id: uuid.UUID,
    status_filter: str | None = Query(None, alias="status"),
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[dict]:
    """Host-only: display name, enrolled flag, and photo-appearance count per
    member (spec 005 FR-012). ``?status=pending`` lists the join-approval queue."""
    await require_event_host(session, user.id, event_id)
    return await list_members(session, event_id, status=status_filter)


@router.delete("/{event_id}/members/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_member(
    event_id: uuid.UUID,
    account_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Host removes a member: revokes their event membership, enrollment, and
    gallery access for THIS event only (spec 005 FR-013)."""
    await require_event_host(session, user.id, event_id)
    try:
        await remove_member(session, event_id, account_id)
    except CannotRemoveHostError:
        raise HTTPException(status_code=409, detail="The host cannot remove themselves.")


@router.post("/{event_id}/members/{account_id}/approve", status_code=status.HTTP_204_NO_CONTENT)
async def approve_member(
    event_id: uuid.UUID,
    account_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    await require_event_host(session, user.id, event_id)
    membership = await session.scalar(
        select(Membership).where(
            Membership.event_id == event_id, Membership.account_id == account_id
        )
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Member not found")
    membership.status = "active"
    await session.commit()


@router.post("/{event_id}/members/{account_id}/reject", status_code=status.HTTP_204_NO_CONTENT)
async def reject_member(
    event_id: uuid.UUID,
    account_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    await require_event_host(session, user.id, event_id)
    membership = await session.scalar(
        select(Membership).where(
            Membership.event_id == event_id, Membership.account_id == account_id
        )
    )
    if membership is not None:
        await session.delete(membership)
        await session.commit()


@router.put("/{event_id}/cover", response_model=EventRead)
async def set_cover(
    event_id: uuid.UUID,
    file: UploadFile = File(...),
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Event:
    """Host-only: set/replace the event's cover image (boarding-pass variant)."""
    await require_event_host(session, user.id, event_id)
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Cover must be an image.",
        )
    data = await file.read()
    if len(data) > get_settings().max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Cover image exceeds the maximum upload size.",
        )
    key = f"events/{event_id}/cover"
    get_storage().put(key, data)
    event.cover_key = key
    await session.commit()
    await session.refresh(event)
    return event


@router.get("/{event_id}/cover")
async def read_cover(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Access-guarded cover bytes — any member of the event may view it.

    Fallback chain (spec 004 R6): host-uploaded cover_key (spec 002) → curated
    cover_photo_id (host-picked or auto-picked from the event's own photos) →
    404 (client falls back to the monogram placeholder).
    """
    await _require_membership(session, user.id, event_id)
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="No cover set")
    if event.cover_key is not None:
        data = get_storage().get(event.cover_key)
        return Response(content=data, media_type=_content_type(data))
    if event.cover_photo_id is not None:
        photo = await session.get(Photo, event.cover_photo_id)
        if photo is None:
            # Dangling reference (photo deleted without going through the
            # delete-photo path) — re-pick lazily rather than 404 forever.
            picked = await auto_pick_cover_photo_id(session, event_id)
            event.cover_photo_id = picked
            event.cover_source = "auto" if picked else None
            await session.commit()
            photo = await session.get(Photo, picked) if picked else None
        if photo is not None:
            data = get_storage().get(photo.storage_key)
            return Response(content=data, media_type=_content_type(data))
    raise HTTPException(status_code=404, detail="No cover set")


@router.put("/{event_id}/cover/photo", response_model=EventRead)
async def set_cover_photo(
    event_id: uuid.UUID,
    payload: SetCover,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Event:
    """Host picks an existing event photo as the cover (spec 004 US3/FR-009)."""
    await require_event_host(session, user.id, event_id)
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    photo = await session.get(Photo, payload.photo_id)
    if photo is None or photo.event_id != event_id:
        raise HTTPException(status_code=404, detail="Photo not found")
    event.cover_photo_id = photo.id
    event.cover_source = "host"
    await session.commit()
    await session.refresh(event)
    return event


@router.delete("/{event_id}/cover/photo", response_model=EventRead)
async def clear_cover_photo(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Event:
    """Host reverts to the auto-picked cover (FR-011): re-picks immediately."""
    await require_event_host(session, user.id, event_id)
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    picked = await auto_pick_cover_photo_id(session, event_id)
    event.cover_photo_id = picked
    event.cover_source = "auto" if picked else None
    await session.commit()
    await session.refresh(event)
    return event


@router.get("/{event_id}/highlights", response_model=list[HighlightItem])
async def get_highlights(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[HighlightItem]:
    """Highlight photos intersected with the caller's accessible set (R5): the
    host sees every highlight, a member only the ones they can already open."""
    membership = await _require_membership(session, user.id, event_id)
    stmt = select(Photo.id, Photo.highlight_rank).where(
        Photo.event_id == event_id, Photo.is_highlight.is_(True)
    )
    if membership.role != "host":
        stmt = stmt.where(
            Photo.id.in_(
                select(GalleryEntry.photo_id).where(GalleryEntry.account_id == user.id)
            )
        )
    rows = await session.execute(stmt.order_by(Photo.highlight_rank))
    return [HighlightItem(photo_id=pid, highlight_rank=rank) for pid, rank in rows.all()]


@router.get("/{event_id}/stats", response_model=EventStats)
async def get_stats(
    event_id: uuid.UUID,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> EventStats:
    """Host-only aggregates (spec 004 US4, R7)."""
    await require_event_host(session, user.id, event_id)
    return EventStats(**await event_stats(session, event_id))


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
    if payload.name_policy is not None:
        event.name_policy = payload.name_policy
    if payload.event_type is not None:
        event.event_type = payload.event_type
    if payload.gallery_visibility is not None:
        event.gallery_visibility = payload.gallery_visibility
    if payload.ai_edit_scope is not None:
        event.ai_edit_scope = payload.ai_edit_scope
    if payload.member_uploads is not None:
        event.member_uploads = payload.member_uploads
    if payload.member_delete_own is not None:
        event.member_delete_own = payload.member_delete_own
    if payload.join_approval is not None:
        event.join_approval = payload.join_approval
    if payload.member_list_visible is not None:
        event.member_list_visible = payload.member_list_visible
    if payload.uploads_closed is not None:
        # Reuses the existing status column rather than a parallel boolean.
        event.status = "archived" if payload.uploads_closed else "active"
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
    keys.extend(
        await session.scalars(
            select(Photo.poster_key).where(
                Photo.event_id == event_id, Photo.poster_key.is_not(None)
            )
        )
    )
    if event.cover_key:
        keys.append(event.cover_key)
    # FK cascades delete photos, faces, clusters, galleries, memberships.
    await session.delete(event)
    await session.commit()
    # Then purge the bytes. Done after the DB commit so a storage hiccup leaves
    # orphaned files (harmless, sweepable) rather than DB rows pointing at gone bytes.
    storage = get_storage()
    for key in keys:
        storage.delete(key)
