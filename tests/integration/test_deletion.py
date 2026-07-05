"""Retention/deletion sweep (FR-022): deleting an event cascades its DB rows AND
purges the photos' stored bytes — no orphaned images left behind."""

import uuid

import pytest
from sqlalchemy import select

from app.api.events import delete_event
from app.db.models import Account, DetectedFace, Event, Membership, Photo
from app.storage.base import get_storage

pytestmark = pytest.mark.asyncio


async def test_delete_event_cascades_rows_and_bytes(db_session):
    host = Account(
        id=uuid.uuid4(),
        email="host@example.com",
        hashed_password="x",
        name="host",
        is_active=True,
    )
    db_session.add(host)
    await db_session.flush()

    event = Event(
        id=uuid.uuid4(),
        name="Gala",
        owner_id=host.id,
        join_code="DELETE",
        join_token="tok",
    )
    db_session.add(event)
    await db_session.flush()
    db_session.add(Membership(event_id=event.id, account_id=host.id, role="host"))

    # A real stored object so we can prove the bytes get purged, not just the row.
    storage = get_storage()
    key = f"test-deletion/{uuid.uuid4()}.jpg"
    storage.put(key, b"\xff\xd8\xff fake jpeg")

    photo = Photo(
        id=uuid.uuid4(),
        event_id=event.id,
        contributor_id=host.id,
        storage_key=key,
    )
    db_session.add(photo)
    await db_session.flush()
    db_session.add(
        DetectedFace(
            id=uuid.uuid4(),
            photo_id=photo.id,
            bbox={"x": 0, "y": 0, "w": 1, "h": 1},
            det_score=0.9,
            embedding=[0.0] * 512,
        )
    )
    await db_session.flush()
    assert storage.get(key) == b"\xff\xd8\xff fake jpeg"

    # Capture ids before expiry — the ORM objects go stale after the cascade.
    event_id, photo_id = event.id, photo.id
    await delete_event(event_id, user=host, session=db_session)
    # The FK cascade runs in the DB; the test session caches rows (expire_on_commit
    # is off), so drop the identity map to force fresh reads. Real requests each get
    # a new session, so this staleness is a test-only artifact.
    db_session.expire_all()

    # DB rows gone (cascade).
    assert await db_session.get(Event, event_id) is None
    assert await db_session.get(Photo, photo_id) is None
    remaining_faces = await db_session.scalar(
        select(DetectedFace.id).where(DetectedFace.photo_id == photo_id)
    )
    assert remaining_faces is None

    # Stored bytes gone (retention sweep).
    with pytest.raises(FileNotFoundError):
        storage.get(key)


async def test_delete_event_cascades_video_and_poster_bytes(db_session):
    """Spec 003 T015: a video's poster_key isn't an FK, so the DB cascade alone
    won't purge it — delete_event must collect and delete it explicitly."""
    host = Account(
        id=uuid.uuid4(),
        email="host2@example.com",
        hashed_password="x",
        name="host2",
        is_active=True,
    )
    db_session.add(host)
    await db_session.flush()

    event = Event(
        id=uuid.uuid4(), name="Gala", owner_id=host.id, join_code="VIDDEL", join_token="tok2"
    )
    db_session.add(event)
    await db_session.flush()
    db_session.add(Membership(event_id=event.id, account_id=host.id, role="host"))

    storage = get_storage()
    video_key = f"test-deletion/{uuid.uuid4()}.mp4"
    poster_key = f"test-deletion/{uuid.uuid4()}-poster.jpg"
    storage.put(video_key, b"fake mp4 bytes")
    storage.put(poster_key, b"\xff\xd8\xff fake poster jpeg")

    photo = Photo(
        id=uuid.uuid4(),
        event_id=event.id,
        contributor_id=host.id,
        storage_key=video_key,
        media_type="video",
        poster_key=poster_key,
    )
    db_session.add(photo)
    await db_session.flush()

    event_id = event.id
    await delete_event(event_id, user=host, session=db_session)

    with pytest.raises(FileNotFoundError):
        storage.get(video_key)
    with pytest.raises(FileNotFoundError):
        storage.get(poster_key)
