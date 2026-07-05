"""Spec 005 US5 — host event settings & permissions. Each toggle is verified
server-side (SC-007): rejected when off, permitted when on. Defaults reproduce
today's behavior exactly (SC-008)."""

import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api.enrollment import _require_not_archived
from app.api.events import join_event
from app.api.export import EditRequest, edit_photo
from app.api.photos import _require_open_for_uploads, delete_photo, host_pool
from app.config import get_settings
from app.db.models import Account, DetectedFace, Event, FaceCluster, Membership, Photo
from app.schemas.event import EventJoin

pytestmark = pytest.mark.asyncio

EMBED = [0.0] * 512


async def _account(session, email: str, name: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=name, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def _event(session, host, **overrides) -> Event:
    event = Event(
        id=uuid.uuid4(),
        name="Test",
        owner_id=host.id,
        join_code=f"S{uuid.uuid4().hex[:6].upper()}",
        join_token=str(uuid.uuid4()),
        **overrides,
    )
    session.add(event)
    await session.flush()
    session.add(Membership(event_id=event.id, account_id=host.id, role="host"))
    await session.flush()
    return event


# --- Defaults reproduce today's behavior (SC-008) ---


async def test_defaults_match_pre_feature_behavior(db_session):
    host = await _account(db_session, "hdef@example.com", "Hdef")
    event = Event(name="x", owner_id=host.id, join_code="DEFAULT1", join_token=str(uuid.uuid4()))
    db_session.add(event)
    await db_session.flush()
    assert event.status == "active"
    assert event.gallery_visibility == "own_only"
    assert event.ai_edit_scope == "solo_only"
    assert event.member_uploads == "enabled"
    assert event.member_delete_own is True
    assert event.join_approval is False
    assert event.member_list_visible is False


# --- uploads_closed (archived) ---


async def test_archived_event_blocks_uploads_and_enrollment(db_session):
    host = await _account(db_session, "h1@example.com", "H1")
    event = await _event(db_session, host, status="archived")

    with pytest.raises(HTTPException) as exc:
        await _require_open_for_uploads(db_session, host.id, event.id)
    assert exc.value.status_code == 403

    with pytest.raises(HTTPException):
        await _require_not_archived(db_session, event.id)


async def test_active_event_allows_uploads_and_enrollment(db_session):
    host = await _account(db_session, "h2@example.com", "H2")
    event = await _event(db_session, host)  # default status=active
    await _require_open_for_uploads(db_session, host.id, event.id)  # no raise
    await _require_not_archived(db_session, event.id)  # no raise


# --- member_uploads (enabled | host_only) ---


async def test_member_uploads_host_only_blocks_members_allows_host(db_session):
    host = await _account(db_session, "h3@example.com", "H3")
    member = await _account(db_session, "m3@example.com", "M3")
    event = await _event(db_session, host, member_uploads="host_only")
    db_session.add(Membership(event_id=event.id, account_id=member.id, role="member"))
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await _require_open_for_uploads(db_session, member.id, event.id)
    assert exc.value.status_code == 403

    await _require_open_for_uploads(db_session, host.id, event.id)  # host unaffected


# --- gallery_visibility (own_only | everyone_sees_all) ---


async def test_gallery_visibility_widens_pool_view_to_members(db_session):
    host = await _account(db_session, "h4@example.com", "H4")
    member = await _account(db_session, "m4@example.com", "M4")
    event = await _event(db_session, host)  # default own_only
    db_session.add(Membership(event_id=event.id, account_id=member.id, role="member"))
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await host_pool(event.id, user=member, session=db_session)
    assert exc.value.status_code == 403

    event.gallery_visibility = "everyone_sees_all"
    await db_session.commit()
    result = await host_pool(event.id, user=member, session=db_session)
    assert result == []  # widened access granted, just no photos seeded


# --- member_delete_own ---


async def test_member_delete_own_gates_uploader_delete(db_session):
    host = await _account(db_session, "h5@example.com", "H5")
    uploader = await _account(db_session, "u5@example.com", "U5")
    event = await _event(db_session, host, member_delete_own=False)
    db_session.add(Membership(event_id=event.id, account_id=uploader.id, role="member"))
    photo = Photo(id=uuid.uuid4(), event_id=event.id, contributor_id=uploader.id, storage_key="k")
    db_session.add(photo)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await delete_photo(photo.id, user=uploader, session=db_session)
    assert exc.value.status_code == 403

    event.member_delete_own = True
    await db_session.commit()
    await delete_photo(photo.id, user=uploader, session=db_session)  # now allowed
    assert await db_session.get(Photo, photo.id) is None


async def test_host_can_always_delete_any_photo(db_session):
    host = await _account(db_session, "h6@example.com", "H6")
    uploader = await _account(db_session, "u6@example.com", "U6")
    event = await _event(db_session, host, member_delete_own=False)
    db_session.add(Membership(event_id=event.id, account_id=uploader.id, role="member"))
    photo = Photo(id=uuid.uuid4(), event_id=event.id, contributor_id=uploader.id, storage_key="k")
    db_session.add(photo)
    await db_session.flush()

    await delete_photo(photo.id, user=host, session=db_session)  # FR-018
    assert await db_session.get(Photo, photo.id) is None


# --- ai_edit_scope (solo_only | any_photo) ---


async def test_ai_edit_scope_any_photo_bypasses_solo_guard(db_session):
    host = await _account(db_session, "h7@example.com", "H7")
    other = await _account(db_session, "o7@example.com", "O7")
    event = await _event(db_session, host, ai_edit_scope="any_photo")

    photo = Photo(
        id=uuid.uuid4(), event_id=event.id, contributor_id=host.id, storage_key="k",
        width=100, height=100, processing_status="done",
    )
    db_session.add(photo)
    await db_session.flush()
    cluster_host = FaceCluster(id=uuid.uuid4(), event_id=event.id, centroid=EMBED, claimed_by_account_id=host.id)
    cluster_other = FaceCluster(id=uuid.uuid4(), event_id=event.id, centroid=EMBED, claimed_by_account_id=other.id)
    db_session.add_all([cluster_host, cluster_other])
    await db_session.flush()
    db_session.add_all(
        [
            DetectedFace(photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 1, "h": 1}, det_score=0.9,
                         embedding=EMBED, cluster_id=cluster_host.id),
            DetectedFace(photo_id=photo.id, bbox={"x": 2, "y": 0, "w": 1, "h": 1}, det_score=0.9,
                         embedding=EMBED, cluster_id=cluster_other.id),
        ]
    )
    await db_session.flush()

    # any_photo bypasses the solo guard entirely — the request proceeds to the
    # Gemini-availability check, which 503s in this test env (no API key), NOT
    # the 403 solo guard. That distinguishes "bypassed" from "blocked".
    with pytest.raises(HTTPException) as exc:
        await edit_photo(
            photo.id, EditRequest(prompt="make it warm", consent=True), user=host, session=db_session
        )
    assert exc.value.status_code == 503


async def test_ai_edit_scope_solo_only_still_blocks_group_photo(db_session):
    host = await _account(db_session, "h8@example.com", "H8")
    other = await _account(db_session, "o8@example.com", "O8")
    event = await _event(db_session, host)  # default solo_only

    photo = Photo(
        id=uuid.uuid4(), event_id=event.id, contributor_id=host.id, storage_key="k",
        width=100, height=100, processing_status="done",
    )
    db_session.add(photo)
    await db_session.flush()
    cluster_host = FaceCluster(id=uuid.uuid4(), event_id=event.id, centroid=EMBED, claimed_by_account_id=host.id)
    cluster_other = FaceCluster(id=uuid.uuid4(), event_id=event.id, centroid=EMBED, claimed_by_account_id=other.id)
    db_session.add_all([cluster_host, cluster_other])
    await db_session.flush()
    db_session.add_all(
        [
            DetectedFace(photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 1, "h": 1}, det_score=0.9,
                         embedding=EMBED, cluster_id=cluster_host.id),
            DetectedFace(photo_id=photo.id, bbox={"x": 2, "y": 0, "w": 1, "h": 1}, det_score=0.9,
                         embedding=EMBED, cluster_id=cluster_other.id),
        ]
    )
    await db_session.flush()

    if not get_settings().edit_solo_only:
        pytest.skip("edit_solo_only disabled in this environment")

    with pytest.raises(HTTPException) as exc:
        await edit_photo(
            photo.id, EditRequest(prompt="make it warm", consent=True), user=host, session=db_session
        )
    assert exc.value.status_code == 403


# --- join_approval ---


async def test_join_approval_pending_then_approved(db_session):
    host = await _account(db_session, "h9@example.com", "H9")
    newcomer = await _account(db_session, "n9@example.com", "N9")
    event = await _event(db_session, host, join_approval=True)

    result = await join_event(EventJoin(join_code=event.join_code), user=newcomer, session=db_session)
    assert result.status == "pending"
    assert result.event is None

    membership = await db_session.scalar(
        select(Membership).where(
            Membership.event_id == event.id, Membership.account_id == newcomer.id
        )
    )
    assert membership.status == "pending"


async def test_join_approval_off_joins_instantly(db_session):
    host = await _account(db_session, "h10@example.com", "H10")
    newcomer = await _account(db_session, "n10@example.com", "N10")
    event = await _event(db_session, host)  # default join_approval=False

    result = await join_event(EventJoin(join_code=event.join_code), user=newcomer, session=db_session)
    assert result.status == "active"
    assert result.event is not None
