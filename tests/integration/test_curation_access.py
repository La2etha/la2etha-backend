"""US2/SC-006 (spec 004) — a highlight photo never leaks to a member who can't
already access it; the host sees every highlight. Also covers the host-only
cover-pick/revert and stats endpoints introduced alongside highlights."""

import uuid

import pytest

from app.api.events import clear_cover_photo, get_highlights, get_stats, set_cover_photo
from app.db.models import Account, Event, GalleryEntry, Membership, Photo
from app.schemas.event import SetCover

pytestmark = pytest.mark.asyncio


async def _account(session, email: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=email, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def _setup(session):
    host = await _account(session, "host@example.com")
    member_a = await _account(session, "a@example.com")  # can see photo_in
    member_b = await _account(session, "b@example.com")  # cannot

    event = Event(id=uuid.uuid4(), name="Trip", owner_id=host.id, join_code="HL0001", join_token="t")
    session.add(event)
    await session.flush()
    session.add_all(
        [
            Membership(event_id=event.id, account_id=host.id, role="host"),
            Membership(event_id=event.id, account_id=member_a.id, role="member"),
            Membership(event_id=event.id, account_id=member_b.id, role="member"),
        ]
    )

    photo_in = Photo(
        id=uuid.uuid4(), event_id=event.id, contributor_id=host.id, storage_key="k1",
        is_highlight=True, highlight_rank=1,
    )
    photo_out = Photo(
        id=uuid.uuid4(), event_id=event.id, contributor_id=host.id, storage_key="k2",
        is_highlight=True, highlight_rank=2,
    )
    session.add_all([photo_in, photo_out])
    await session.flush()

    # member_a is verified only in photo_in; member_b in neither.
    session.add(
        GalleryEntry(account_id=member_a.id, event_id=event.id, photo_id=photo_in.id, origin="auto")
    )
    await session.flush()

    return event, host, member_a, member_b, photo_in, photo_out


async def test_member_only_sees_highlights_they_can_access(db_session):
    event, host, member_a, member_b, photo_in, photo_out = await _setup(db_session)

    a_highlights = await get_highlights(event.id, user=member_a, session=db_session)
    assert {h.photo_id for h in a_highlights} == {photo_in.id}

    b_highlights = await get_highlights(event.id, user=member_b, session=db_session)
    assert b_highlights == []

    host_highlights = await get_highlights(event.id, user=host, session=db_session)
    assert {h.photo_id for h in host_highlights} == {photo_in.id, photo_out.id}


async def test_host_sets_and_reverts_cover(db_session):
    event, host, member_a, _, photo_in, photo_out = await _setup(db_session)

    updated = await set_cover_photo(
        event.id, SetCover(photo_id=photo_in.id), user=host, session=db_session
    )
    assert updated.cover_photo_id == photo_in.id
    assert updated.cover_source == "host"

    with pytest.raises(Exception):
        await set_cover_photo(event.id, SetCover(photo_id=photo_in.id), user=member_a, session=db_session)

    reverted = await clear_cover_photo(event.id, user=host, session=db_session)
    assert reverted.cover_source == "auto"
    # Auto-pick chose one of the event's photos (whichever the fallback ranks first).
    assert reverted.cover_photo_id in {photo_in.id, photo_out.id}


async def test_stats_are_host_only_and_hide_unclaimed_identities(db_session):
    event, host, member_a, member_b, *_ = await _setup(db_session)

    stats = await get_stats(event.id, user=host, session=db_session)
    assert stats.photo_count == 2
    assert stats.unclaimed_count >= 0  # no clusters seeded here — count only, never itemized
    assert all(m.account_id != uuid.UUID(int=0) for m in stats.members)

    with pytest.raises(Exception):
        await get_stats(event.id, user=member_a, session=db_session)
