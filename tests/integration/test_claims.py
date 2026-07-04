"""US6 — manual 'this is me' correction (FR-018): claiming a photo grants access
and adds it to the gallery; unclaiming revokes access and leaves an 'unclaimed'
tombstone so an auto re-match can't resurrect it."""

import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.access.guard import require_photo_read
from app.api.gallery import set_claim
from app.db.models import Account, Event, GalleryClaim, Membership, Photo

pytestmark = pytest.mark.asyncio


async def _seed(session):
    account = Account(
        id=uuid.uuid4(),
        email="me@example.com",
        hashed_password="x",
        name="me",
        is_active=True,
    )
    session.add(account)
    await session.flush()

    event = Event(
        id=uuid.uuid4(),
        name="Party",
        owner_id=account.id,
        join_code="CLAIM1",
        join_token="tok",
    )
    session.add(event)
    await session.flush()
    # Member (not host) — so access hinges on the gallery entry, not the host bypass.
    session.add(Membership(event_id=event.id, account_id=account.id, role="member"))

    photo = Photo(
        id=uuid.uuid4(),
        event_id=event.id,
        contributor_id=account.id,
        storage_key="k/1.jpg",
    )
    session.add(photo)
    await session.flush()
    return account, photo


async def test_claim_grants_access_then_unclaim_revokes(db_session):
    account, photo = await _seed(db_session)

    # Before claiming, a member not verified in the photo can't read it (404).
    with pytest.raises(HTTPException) as before:
        await require_photo_read(db_session, account.id, photo.id)
    assert before.value.status_code == 404

    # Claim → access granted.
    await set_claim(db_session, account.id, photo, claimed=True)
    got = await require_photo_read(db_session, account.id, photo.id)
    assert got.id == photo.id

    # Unclaim → access revoked again...
    await set_claim(db_session, account.id, photo, claimed=False)
    with pytest.raises(HTTPException) as after:
        await require_photo_read(db_session, account.id, photo.id)
    assert after.value.status_code == 404

    # ...and a durable 'unclaimed' tombstone remains for the materializer to honor.
    claim = await db_session.scalar(
        select(GalleryClaim).where(GalleryClaim.photo_id == photo.id)
    )
    assert claim.state == "unclaimed"


async def test_claim_is_idempotent(db_session):
    account, photo = await _seed(db_session)
    await set_claim(db_session, account.id, photo, claimed=True)
    await set_claim(db_session, account.id, photo, claimed=True)  # no duplicate entry/row
    # The (account, photo) unique constraint on GalleryEntry would raise on a
    # second insert; reaching here means the guard held.
    got = await require_photo_read(db_session, account.id, photo.id)
    assert got.id == photo.id
