"""Spec 005 SC-003 — one account in two events: nothing about event 2 (faces,
membership, names) is retrievable by a member of event 1 who isn't in event 2.
"""

import uuid

import pytest
from fastapi import HTTPException

from app.access.guard import require_photo_read
from app.api.events import get_event
from app.api.photos import read_photo_faces
from app.db.models import Account, DetectedFace, Event, FaceCluster, GalleryEntry, Membership, Photo

pytestmark = pytest.mark.asyncio

EMBED = [0.0] * 512


async def _account(session, email: str, name: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=name, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def test_cross_event_isolation(db_session):
    # `shared` belongs to both events; `outsider` only belongs to event 1.
    shared = await _account(db_session, "shared@example.com", "Shared")
    outsider = await _account(db_session, "outsider@example.com", "Outsider")
    host = await _account(db_session, "host@example.com", "Host")

    event1 = Event(
        id=uuid.uuid4(), name="Event1", owner_id=host.id,
        join_code="EV1AAA", join_token=str(uuid.uuid4()), name_policy="everyone",
    )
    event2 = Event(
        id=uuid.uuid4(), name="Event2", owner_id=host.id,
        join_code="EV2AAA", join_token=str(uuid.uuid4()), name_policy="everyone",
    )
    db_session.add_all([event1, event2])
    await db_session.flush()

    db_session.add_all(
        [
            Membership(event_id=event1.id, account_id=host.id, role="host"),
            Membership(event_id=event1.id, account_id=shared.id, role="member"),
            Membership(event_id=event1.id, account_id=outsider.id, role="member"),
            Membership(event_id=event2.id, account_id=host.id, role="host"),
            Membership(event_id=event2.id, account_id=shared.id, role="member"),
        ]
    )
    await db_session.flush()

    photo2 = Photo(
        id=uuid.uuid4(), event_id=event2.id, contributor_id=host.id, storage_key="k2",
        width=100, height=100, processing_status="done",
    )
    db_session.add(photo2)
    await db_session.flush()
    cluster = FaceCluster(id=uuid.uuid4(), event_id=event2.id, centroid=EMBED, claimed_by_account_id=shared.id)
    db_session.add(cluster)
    await db_session.flush()
    db_session.add(
        DetectedFace(
            photo_id=photo2.id, bbox={"x": 0, "y": 0, "w": 10, "h": 10}, det_score=0.9,
            embedding=EMBED, cluster_id=cluster.id,
        )
    )
    # `shared`'s claim materializes as gallery access, same as the real pipeline.
    db_session.add(
        GalleryEntry(account_id=shared.id, event_id=event2.id, photo_id=photo2.id, origin="auto")
    )
    await db_session.flush()

    # Outsider (event 1 only) gets event 2's event record denied.
    with pytest.raises(HTTPException):
        await get_event(event2.id, user=outsider, session=db_session)

    # Outsider cannot read event 2's photo or its faces at all.
    with pytest.raises(HTTPException):
        await require_photo_read(db_session, outsider.id, photo2.id)
    with pytest.raises(HTTPException):
        await read_photo_faces(photo2.id, user=outsider, session=db_session)

    # Shared (in both) can see event 2's photo/faces fine — proves the guard is
    # event-membership-based, not blanket-blocking the account.
    faces = await read_photo_faces(photo2.id, user=shared, session=db_session)
    assert faces[0].is_me is True
