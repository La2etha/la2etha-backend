"""GET /events lists only the caller's events, with role + counts (home screen)."""

import uuid

import pytest

from app.api.events import list_events
from app.db.models import Account, Event, Membership, Photo

pytestmark = pytest.mark.asyncio


async def _account(session, email: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=email, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def test_list_returns_only_my_events_with_role_and_counts(db_session):
    host = await _account(db_session, "host@example.com")
    guest = await _account(db_session, "guest@example.com")

    mine = Event(id=uuid.uuid4(), name="Mine", owner_id=host.id, join_code="AA1", join_token="t1")
    theirs = Event(id=uuid.uuid4(), name="Theirs", owner_id=guest.id, join_code="BB2", join_token="t2")
    db_session.add_all([mine, theirs])
    await db_session.flush()

    # host: hosts "mine" and is a member of "theirs"; guest hosts "theirs" only.
    db_session.add_all([
        Membership(event_id=mine.id, account_id=host.id, role="host"),
        Membership(event_id=mine.id, account_id=guest.id, role="member"),
        Membership(event_id=theirs.id, account_id=guest.id, role="host"),
        Membership(event_id=theirs.id, account_id=host.id, role="member"),
    ])
    db_session.add_all([
        Photo(id=uuid.uuid4(), event_id=mine.id, contributor_id=host.id, storage_key="p1"),
        Photo(id=uuid.uuid4(), event_id=mine.id, contributor_id=guest.id, storage_key="p2"),
    ])
    await db_session.flush()

    out = await list_events(user=host, session=db_session)

    by_name = {e.name: e for e in out}
    assert set(by_name) == {"Mine", "Theirs"}  # both events host belongs to
    assert by_name["Mine"].role == "host"
    assert by_name["Mine"].member_count == 2
    assert by_name["Mine"].photo_count == 2
    assert by_name["Theirs"].role == "member"
    assert by_name["Theirs"].photo_count == 0

    # A user with no memberships sees nothing.
    loner = await _account(db_session, "loner@example.com")
    assert await list_events(user=loner, session=db_session) == []
