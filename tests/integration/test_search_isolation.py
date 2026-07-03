"""US3 (FR-015) — search results are limited to the caller's verified photos.

The isolation rule is enforced in the query, so we test it with a synthetic query
vector (no SigLIP needed): a photo the caller is NOT verified in must never appear
in their search results, even if it's the closest match to the query.
"""

import uuid

import numpy as np
import pytest

from app.db.models import (
    Account,
    Event,
    GalleryEntry,
    Membership,
    Photo,
)
from app.services.search import search_gallery

pytestmark = pytest.mark.asyncio

DIM = 768


def _unit(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


async def _account(session, email: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=email, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def test_search_only_returns_callers_verified_photos(db_session):
    alice = await _account(db_session, "alice@example.com")
    bob = await _account(db_session, "bob@example.com")
    event = Event(
        id=uuid.uuid4(), name="Party", owner_id=alice.id, join_code="SRCH01", join_token="tk"
    )
    db_session.add(event)
    await db_session.flush()
    db_session.add_all(
        [
            Membership(event_id=event.id, account_id=alice.id, role="host"),
            Membership(event_id=event.id, account_id=bob.id, role="member"),
        ]
    )

    query = _unit(1)
    # Alice's verified photo has a *weaker* match to the query than Bob's photo,
    # to prove ranking never overrides isolation.
    alice_photo = Photo(
        id=uuid.uuid4(), event_id=event.id, contributor_id=alice.id,
        storage_key="a", search_embedding=_unit(2),
    )
    bob_photo = Photo(
        id=uuid.uuid4(), event_id=event.id, contributor_id=bob.id,
        storage_key="b", search_embedding=query,  # near-perfect match to the query
    )
    # A photo nobody indexed for Alice, and one with no embedding.
    unindexed = Photo(
        id=uuid.uuid4(), event_id=event.id, contributor_id=alice.id,
        storage_key="c", search_embedding=None,
    )
    db_session.add_all([alice_photo, bob_photo, unindexed])
    await db_session.flush()

    # Alice is verified only in alice_photo and the unindexed one.
    db_session.add_all(
        [
            GalleryEntry(
                account_id=alice.id, event_id=event.id, photo_id=alice_photo.id, origin="auto"
            ),
            GalleryEntry(
                account_id=alice.id, event_id=event.id, photo_id=unindexed.id, origin="auto"
            ),
        ]
    )
    # Bob is verified in his photo (the strong match), but Alice must never see it.
    db_session.add(
        GalleryEntry(
            account_id=bob.id, event_id=event.id, photo_id=bob_photo.id, origin="auto"
        )
    )
    await db_session.flush()

    hits = await search_gallery(
        db_session, alice.id, event.id, np.asarray(query, dtype=np.float32), limit=40
    )
    returned = {h.photo_id for h in hits}

    assert bob_photo.id not in returned  # never leak Bob's photo, despite best match
    assert unindexed.id not in returned  # no embedding → not searchable
    assert returned == {alice_photo.id}  # only Alice's verified, indexed photo


async def test_search_excludes_other_events(db_session):
    alice = await _account(db_session, "solo@example.com")
    e1 = Event(id=uuid.uuid4(), name="E1", owner_id=alice.id, join_code="EV1", join_token="t1")
    e2 = Event(id=uuid.uuid4(), name="E2", owner_id=alice.id, join_code="EV2", join_token="t2")
    db_session.add_all([e1, e2])
    await db_session.flush()
    db_session.add_all(
        [
            Membership(event_id=e1.id, account_id=alice.id, role="host"),
            Membership(event_id=e2.id, account_id=alice.id, role="host"),
        ]
    )
    p2 = Photo(
        id=uuid.uuid4(), event_id=e2.id, contributor_id=alice.id,
        storage_key="x", search_embedding=_unit(5),
    )
    db_session.add(p2)
    await db_session.flush()
    db_session.add(
        GalleryEntry(account_id=alice.id, event_id=e2.id, photo_id=p2.id, origin="auto")
    )
    await db_session.flush()

    # Searching event 1 must not return a photo verified in event 2.
    hits = await search_gallery(
        db_session, alice.id, e1.id, np.asarray(_unit(5), dtype=np.float32), limit=40
    )
    assert hits == []

    # Sanity: it IS findable when searching event 2.
    hits2 = await search_gallery(
        db_session, alice.id, e2.id, np.asarray(_unit(5), dtype=np.float32), limit=40
    )
    assert {h.photo_id for h in hits2} == {p2.id}
