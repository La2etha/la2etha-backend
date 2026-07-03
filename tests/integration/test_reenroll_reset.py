"""Re-enrollment must fully replace a prior identity match (not accumulate it).

``reset_auto_match`` drops the account's auto gallery entries and unlinks the
clusters it claimed, while preserving manual "this is me" claims (FR-018).
"""

import uuid

import numpy as np
import pytest
from sqlalchemy import select

from app.db.models import (
    Account,
    Event,
    FaceCluster,
    GalleryEntry,
    Membership,
    Photo,
)
from app.services.gallery import reset_auto_match

pytestmark = pytest.mark.asyncio


def _zeros() -> list[float]:
    return np.zeros(512, dtype=np.float32).tolist()


async def _account(session, email: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=email, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def test_reset_drops_auto_keeps_claims(db_session):
    # reset_auto_match is sync; drive it against the async test session's sync bind.
    account = await _account(db_session, "u@example.com")
    event = Event(
        id=uuid.uuid4(), name="E", owner_id=account.id, join_code="Z1", join_token="t1"
    )
    db_session.add(event)
    await db_session.flush()
    db_session.add(Membership(event_id=event.id, account_id=account.id, role="member"))

    p_auto, p_claim = uuid.uuid4(), uuid.uuid4()
    db_session.add_all(
        [
            Photo(id=p_auto, event_id=event.id, contributor_id=account.id, storage_key="a"),
            Photo(id=p_claim, event_id=event.id, contributor_id=account.id, storage_key="c"),
        ]
    )
    await db_session.flush()
    cluster = FaceCluster(
        id=uuid.uuid4(), event_id=event.id, centroid=_zeros(), claimed_by_account_id=account.id
    )
    db_session.add(cluster)
    db_session.add_all(
        [
            GalleryEntry(
                account_id=account.id, event_id=event.id, photo_id=p_auto, origin="auto"
            ),
            GalleryEntry(
                account_id=account.id, event_id=event.id, photo_id=p_claim, origin="claim"
            ),
        ]
    )
    await db_session.flush()

    # run_sync gives reset_auto_match a synchronous Session over the same connection.
    await db_session.run_sync(lambda s: reset_auto_match(s, account.id, event.id))

    remaining = (
        await db_session.scalars(
            select(GalleryEntry.origin).where(GalleryEntry.account_id == account.id)
        )
    ).all()
    assert remaining == ["claim"]  # auto dropped, manual claim preserved

    # Re-select the value (the bulk UPDATE bypassed the identity map).
    unlinked = await db_session.scalar(
        select(FaceCluster.claimed_by_account_id).where(FaceCluster.id == cluster.id)
    )
    assert unlinked is None  # cluster unlinked
