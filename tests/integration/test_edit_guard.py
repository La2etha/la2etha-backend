"""US5 (FR-017, Constitution IV) — AI edit is solo-photo-only.

The privacy boundary: a photo may be sent to the cloud editor only if it contains
no one but the requesting user. Any other member, or any unclaimed face, blocks it.
"""

import uuid

import numpy as np
import pytest

from app.db.models import Account, DetectedFace, Event, FaceCluster, Photo
from app.services.export import is_solo_editable

pytestmark = pytest.mark.asyncio


def _zeros() -> list[float]:
    return np.zeros(512, dtype=np.float32).tolist()


async def _account(session, email: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=email, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def _photo(session, event_id, owner_id) -> Photo:
    p = Photo(id=uuid.uuid4(), event_id=event_id, contributor_id=owner_id, storage_key="k")
    session.add(p)
    await session.flush()
    return p


async def _setup(session):
    me = await _account(session, "me@example.com")
    event = Event(id=uuid.uuid4(), name="E", owner_id=me.id, join_code="ED01", join_token="t")
    session.add(event)
    await session.flush()
    mine = FaceCluster(
        id=uuid.uuid4(), event_id=event.id, centroid=_zeros(), claimed_by_account_id=me.id
    )
    other = FaceCluster(
        id=uuid.uuid4(), event_id=event.id, centroid=_zeros(), claimed_by_account_id=None
    )
    session.add_all([mine, other])
    await session.flush()
    return me, event, mine, other


async def test_own_solo_photo_is_editable(db_session):
    me, event, mine, _ = await _setup(db_session)
    photo = await _photo(db_session, event.id, me.id)
    db_session.add(
        DetectedFace(
            photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 9, "h": 9}, det_score=0.9,
            cluster_id=mine.id, embedding=_zeros(),
        )
    )
    await db_session.flush()
    assert await is_solo_editable(db_session, photo.id, me.id) is True


async def test_photo_with_no_faces_is_editable(db_session):
    me, event, _, _ = await _setup(db_session)
    photo = await _photo(db_session, event.id, me.id)  # scenery, no people
    assert await is_solo_editable(db_session, photo.id, me.id) is True


async def test_multi_person_photo_blocked(db_session):
    me, event, mine, other = await _setup(db_session)
    photo = await _photo(db_session, event.id, me.id)
    db_session.add_all(
        [
            DetectedFace(
                photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 9, "h": 9}, det_score=0.9,
                cluster_id=mine.id, embedding=_zeros(),
            ),
            DetectedFace(
                photo_id=photo.id, bbox={"x": 20, "y": 0, "w": 9, "h": 9}, det_score=0.9,
                cluster_id=other.id, embedding=_zeros(),  # someone else
            ),
        ]
    )
    await db_session.flush()
    assert await is_solo_editable(db_session, photo.id, me.id) is False


async def test_unclaimed_face_blocks_edit(db_session):
    me, event, _, _ = await _setup(db_session)
    photo = await _photo(db_session, event.id, me.id)
    # A single face that isn't linked to any account (stranger / unconfirmed).
    db_session.add(
        DetectedFace(
            photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 9, "h": 9}, det_score=0.9,
            cluster_id=None, embedding=_zeros(),
        )
    )
    await db_session.flush()
    assert await is_solo_editable(db_session, photo.id, me.id) is False
