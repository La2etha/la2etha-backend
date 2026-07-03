"""US4 (FR-016) — export removal targets only unclaimed background people.

Tests the privacy-selection logic (which faces get inpainted out), independent of
LaMa: a claimed member must be preserved; only an unclaimed background face is
selected for removal.
"""

import uuid

import numpy as np
import pytest

from app.db.models import Account, Event, FaceCluster, Photo
from app.services.export import faces_to_remove

pytestmark = pytest.mark.asyncio


def _zeros() -> list[float]:
    return np.zeros(512, dtype=np.float32).tolist()


async def _account(session, email: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=email, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def test_only_unclaimed_background_faces_removed(db_session):
    from app.db.models import DetectedFace

    member = await _account(db_session, "member@example.com")
    event = Event(
        id=uuid.uuid4(), name="Party", owner_id=member.id, join_code="EXP01", join_token="tk"
    )
    db_session.add(event)
    await db_session.flush()

    photo = Photo(id=uuid.uuid4(), event_id=event.id, contributor_id=member.id, storage_key="k")
    db_session.add(photo)

    # Cluster A is a claimed member; cluster B is an unclaimed stranger.
    claimed = FaceCluster(
        id=uuid.uuid4(), event_id=event.id, centroid=_zeros(), claimed_by_account_id=member.id
    )
    unclaimed = FaceCluster(
        id=uuid.uuid4(), event_id=event.id, centroid=_zeros(), claimed_by_account_id=None
    )
    db_session.add_all([claimed, unclaimed])
    await db_session.flush()

    # 1) claimed member, foreground  → keep
    # 2) claimed member, background  → keep (they're a real member)
    # 3) unclaimed stranger, background → REMOVE
    # 4) unclaimed stranger, foreground → keep (prominent subject, not a bystander)
    db_session.add_all(
        [
            DetectedFace(
                photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 50, "h": 50}, det_score=0.9,
                is_background=False, cluster_id=claimed.id, embedding=_zeros(),
            ),
            DetectedFace(
                photo_id=photo.id, bbox={"x": 60, "y": 0, "w": 8, "h": 8}, det_score=0.9,
                is_background=True, cluster_id=claimed.id, embedding=_zeros(),
            ),
            DetectedFace(
                photo_id=photo.id, bbox={"x": 200, "y": 10, "w": 6, "h": 6}, det_score=0.9,
                is_background=True, cluster_id=unclaimed.id, embedding=_zeros(),
            ),
            DetectedFace(
                photo_id=photo.id, bbox={"x": 300, "y": 0, "w": 60, "h": 60}, det_score=0.9,
                is_background=False, cluster_id=unclaimed.id, embedding=_zeros(),
            ),
        ]
    )
    await db_session.flush()

    remove = await faces_to_remove(db_session, photo.id)

    assert len(remove) == 1, remove
    assert remove[0] == {"x": 200, "y": 10, "w": 6, "h": 6}  # only the unclaimed background face


async def test_noise_faces_without_cluster_count_as_unclaimed(db_session):
    from app.db.models import DetectedFace

    owner = await _account(db_session, "owner@example.com")
    event = Event(
        id=uuid.uuid4(), name="E", owner_id=owner.id, join_code="EXP02", join_token="tk2"
    )
    db_session.add(event)
    await db_session.flush()
    photo = Photo(id=uuid.uuid4(), event_id=event.id, contributor_id=owner.id, storage_key="k2")
    db_session.add(photo)
    await db_session.flush()

    # A background face that never got clustered (cluster_id NULL) is unclaimed → remove.
    db_session.add(
        DetectedFace(
            photo_id=photo.id, bbox={"x": 5, "y": 5, "w": 7, "h": 7}, det_score=0.7,
            is_background=True, cluster_id=None, embedding=_zeros(),
        )
    )
    await db_session.flush()

    remove = await faces_to_remove(db_session, photo.id)
    assert remove == [{"x": 5, "y": 5, "w": 7, "h": 7}]
