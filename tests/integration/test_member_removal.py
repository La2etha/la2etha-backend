"""Spec 005 US4 — host removes a member: full event-scoped revocation, others
unaffected, faces revert to anonymous (SC-004).
"""

import uuid

import pytest
from sqlalchemy import select

from app.access.guard import require_photo_read
from app.api.photos import read_photo_faces
from app.db.models import (
    Account,
    DetectedFace,
    Event,
    FaceCluster,
    GalleryClaim,
    GalleryEntry,
    IdentityEnrollment,
    Membership,
    Photo,
)
from app.services.membership import CannotRemoveHostError, remove_member

pytestmark = pytest.mark.asyncio

EMBED = [0.0] * 512


async def _account(session, email: str, name: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=name, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def test_removal_revokes_event_scoped_access_and_reverts_face_to_guest(db_session):
    host = await _account(db_session, "host@example.com", "Host")
    a = await _account(db_session, "a@example.com", "Alice")
    b = await _account(db_session, "b@example.com", "Bob")

    event = Event(
        id=uuid.uuid4(), name="Grad", owner_id=host.id,
        join_code="REM001", join_token=str(uuid.uuid4()), name_policy="everyone",
    )
    db_session.add(event)
    await db_session.flush()
    db_session.add_all(
        [
            Membership(event_id=event.id, account_id=host.id, role="host"),
            Membership(event_id=event.id, account_id=a.id, role="member"),
            Membership(event_id=event.id, account_id=b.id, role="member"),
        ]
    )

    photo = Photo(
        id=uuid.uuid4(), event_id=event.id, contributor_id=a.id, storage_key="k",
        width=100, height=100, processing_status="done",
    )
    db_session.add(photo)
    await db_session.flush()

    cluster = FaceCluster(id=uuid.uuid4(), event_id=event.id, centroid=EMBED, claimed_by_account_id=a.id)
    db_session.add(cluster)
    await db_session.flush()
    db_session.add(
        DetectedFace(
            photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 10, "h": 10}, det_score=0.9,
            embedding=EMBED, cluster_id=cluster.id,
        )
    )
    db_session.add(
        IdentityEnrollment(account_id=a.id, event_id=event.id, centroid=EMBED, sample_count=5)
    )
    db_session.add(
        GalleryEntry(account_id=a.id, event_id=event.id, photo_id=photo.id, origin="auto")
    )
    db_session.add(GalleryClaim(account_id=a.id, photo_id=photo.id, state="claimed"))
    await db_session.flush()

    # a can read the photo (verified in it) before removal.
    got = await require_photo_read(db_session, a.id, photo.id)
    assert got.id == photo.id

    await remove_member(db_session, event.id, a.id)

    # Membership, enrollment, gallery entry/claim all gone for `a` in this event.
    assert await db_session.scalar(
        select(Membership.id).where(Membership.event_id == event.id, Membership.account_id == a.id)
    ) is None
    assert await db_session.scalar(
        select(IdentityEnrollment.id).where(
            IdentityEnrollment.account_id == a.id, IdentityEnrollment.event_id == event.id
        )
    ) is None
    assert await db_session.scalar(
        select(GalleryEntry.id).where(GalleryEntry.account_id == a.id, GalleryEntry.event_id == event.id)
    ) is None
    assert await db_session.scalar(
        select(GalleryClaim.id).where(GalleryClaim.account_id == a.id, GalleryClaim.photo_id == photo.id)
    ) is None

    # Access denied post-removal.
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        await require_photo_read(db_session, a.id, photo.id)

    # Her face reverts to anonymous for everyone (cluster unclaimed). Host
    # always has photo access, so use it to check the label without needing a
    # separate gallery grant for the viewer.
    faces = await read_photo_faces(photo.id, user=host, session=db_session)
    assert faces[0].name is None
    assert faces[0].is_me is False

    # b (untouched) still a member.
    assert await db_session.scalar(
        select(Membership.id).where(Membership.event_id == event.id, Membership.account_id == b.id)
    ) is not None

    # Photo she uploaded stays in the pool.
    still_there = await db_session.get(Photo, photo.id)
    assert still_there is not None


async def test_host_cannot_be_removed(db_session):
    host = await _account(db_session, "host2@example.com", "Host2")
    event = Event(
        id=uuid.uuid4(), name="Iftar", owner_id=host.id,
        join_code="REM002", join_token=str(uuid.uuid4()),
    )
    db_session.add(event)
    await db_session.flush()
    db_session.add(Membership(event_id=event.id, account_id=host.id, role="host"))
    await db_session.flush()

    with pytest.raises(CannotRemoveHostError):
        await remove_member(db_session, event.id, host.id)
