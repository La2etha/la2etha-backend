"""Host sets an event cover; any member can read it back; a non-member cannot."""

import uuid

import pytest

from app.api.events import read_cover, set_cover
from app.db.models import Account, Event, Membership

pytestmark = pytest.mark.asyncio


class _FakeUpload:
    def __init__(self, data: bytes, content_type: str = "image/jpeg") -> None:
        self._data = data
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._data


async def _account(session, email: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=email, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def test_host_sets_cover_and_member_reads_it_back(db_session):
    host = await _account(db_session, "host@example.com")
    member = await _account(db_session, "member@example.com")
    stranger = await _account(db_session, "stranger@example.com")

    event = Event(id=uuid.uuid4(), name="Gala", owner_id=host.id, join_code="COVER1", join_token="t")
    db_session.add(event)
    await db_session.flush()
    db_session.add_all([
        Membership(event_id=event.id, account_id=host.id, role="host"),
        Membership(event_id=event.id, account_id=member.id, role="member"),
    ])
    await db_session.flush()

    updated = await set_cover(
        event.id, file=_FakeUpload(b"\xff\xd8\xff fake jpeg"), user=host, session=db_session
    )
    assert updated.has_cover is True

    resp = await read_cover(event.id, user=member, session=db_session)
    assert resp.body == b"\xff\xd8\xff fake jpeg"
    assert resp.media_type == "image/jpeg"

    with pytest.raises(Exception):
        await read_cover(event.id, user=stranger, session=db_session)


async def test_non_host_cannot_set_cover(db_session):
    host = await _account(db_session, "host2@example.com")
    member = await _account(db_session, "member2@example.com")

    event = Event(id=uuid.uuid4(), name="Gala2", owner_id=host.id, join_code="COVER2", join_token="t2")
    db_session.add(event)
    await db_session.flush()
    db_session.add_all([
        Membership(event_id=event.id, account_id=host.id, role="host"),
        Membership(event_id=event.id, account_id=member.id, role="member"),
    ])
    await db_session.flush()

    with pytest.raises(Exception):
        await set_cover(event.id, file=_FakeUpload(b"x"), user=member, session=db_session)
