"""US1 — access isolation (FR-019, Principle IV): a member is denied a photo
they are not verified in (404), while the host may read any event photo."""

import uuid

import pytest
from fastapi import HTTPException

from app.access.guard import require_photo_read
from app.db.models import Account, Event, GalleryEntry, Membership, Photo

pytestmark = pytest.mark.asyncio


async def _make_account(session, email: str) -> Account:
    account = Account(
        id=uuid.uuid4(),
        email=email,
        hashed_password="x",
        name=email.split("@")[0],
        is_active=True,
    )
    session.add(account)
    await session.flush()
    return account


async def test_member_denied_photo_they_are_not_in(db_session):
    host = await _make_account(db_session, "host@example.com")
    member = await _make_account(db_session, "member@example.com")

    event = Event(
        id=uuid.uuid4(),
        name="Wedding",
        owner_id=host.id,
        join_code="ABC123",
        join_token="tok",
    )
    db_session.add(event)
    await db_session.flush()

    db_session.add_all(
        [
            Membership(event_id=event.id, account_id=host.id, role="host"),
            Membership(event_id=event.id, account_id=member.id, role="member"),
        ]
    )

    photo_in = Photo(id=uuid.uuid4(), event_id=event.id, contributor_id=host.id, storage_key="k1")
    photo_out = Photo(id=uuid.uuid4(), event_id=event.id, contributor_id=host.id, storage_key="k2")
    db_session.add_all([photo_in, photo_out])
    await db_session.flush()

    # The member is verified only in photo_in.
    db_session.add(
        GalleryEntry(
            account_id=member.id, event_id=event.id, photo_id=photo_in.id, origin="auto"
        )
    )
    await db_session.flush()

    # Member CAN read the photo they're verified in.
    got = await require_photo_read(db_session, member.id, photo_in.id)
    assert got.id == photo_in.id

    # Member CANNOT read a photo they're not in — 404 (never leak presence).
    with pytest.raises(HTTPException) as exc:
        await require_photo_read(db_session, member.id, photo_out.id)
    assert exc.value.status_code == 404

    # Host CAN read any event photo.
    got_host = await require_photo_read(db_session, host.id, photo_out.id)
    assert got_host.id == photo_out.id


async def test_unknown_photo_returns_404(db_session):
    stranger = await _make_account(db_session, "stranger@example.com")
    with pytest.raises(HTTPException) as exc:
        await require_photo_read(db_session, stranger.id, uuid.uuid4())
    assert exc.value.status_code == 404
