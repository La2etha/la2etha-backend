"""US2 — video pool access (spec 003 T010): the generic access guard already
covers videos (a video is just a Photo row with media_type="video"), and the
content_hash duplicate-detection query used by the video upload path in
`app/api/photos.py` behaves as expected.
"""

import hashlib
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.access.guard import require_photo_read
from app.db.models import Account, Event, GalleryEntry, Membership, Photo

pytestmark = pytest.mark.asyncio


async def _make_account(session, email: str) -> Account:
    account = Account(
        id=uuid.uuid4(), email=email, hashed_password="x", name=email.split("@")[0], is_active=True
    )
    session.add(account)
    await session.flush()
    return account


async def test_member_not_in_the_video_cannot_fetch_it(db_session):
    host = await _make_account(db_session, "host@example.com")
    member = await _make_account(db_session, "member@example.com")

    event = Event(id=uuid.uuid4(), name="Wedding", owner_id=host.id, join_code="VID001", join_token="tok")
    db_session.add(event)
    await db_session.flush()
    db_session.add_all(
        [
            Membership(event_id=event.id, account_id=host.id, role="host"),
            Membership(event_id=event.id, account_id=member.id, role="member"),
        ]
    )

    video = Photo(
        id=uuid.uuid4(),
        event_id=event.id,
        contributor_id=host.id,
        storage_key="k1.mp4",
        media_type="video",
    )
    db_session.add(video)
    await db_session.flush()

    # Member not verified in the video → 404, same as a still photo.
    with pytest.raises(HTTPException) as exc:
        await require_photo_read(db_session, member.id, video.id)
    assert exc.value.status_code == 404

    # Once verified (GalleryEntry), the member can read it.
    db_session.add(
        GalleryEntry(account_id=member.id, event_id=event.id, photo_id=video.id, origin="auto")
    )
    await db_session.flush()
    got = await require_photo_read(db_session, member.id, video.id)
    assert got.id == video.id

    # Host can always read it.
    got_host = await require_photo_read(db_session, host.id, video.id)
    assert got_host.id == video.id


async def test_content_hash_dedupe(db_session):
    host = await _make_account(db_session, "host2@example.com")
    event = Event(id=uuid.uuid4(), name="Gala", owner_id=host.id, join_code="VID002", join_token="tok2")
    db_session.add(event)
    await db_session.flush()
    db_session.add(Membership(event_id=event.id, account_id=host.id, role="host"))

    content_hash = hashlib.sha256(b"same clip bytes").hexdigest()
    db_session.add(
        Photo(
            id=uuid.uuid4(),
            event_id=event.id,
            contributor_id=host.id,
            storage_key="k2.mp4",
            media_type="video",
            content_hash=content_hash,
        )
    )
    await db_session.flush()

    # Same bytes re-uploaded → the dedupe query (used verbatim in upload_photos)
    # finds the existing row.
    clash = await db_session.scalar(
        select(Photo.id).where(Photo.event_id == event.id, Photo.content_hash == content_hash)
    )
    assert clash is not None

    # Different bytes → no clash.
    other_hash = hashlib.sha256(b"different clip bytes").hexdigest()
    no_clash = await db_session.scalar(
        select(Photo.id).where(Photo.event_id == event.id, Photo.content_hash == other_hash)
    )
    assert no_clash is None
