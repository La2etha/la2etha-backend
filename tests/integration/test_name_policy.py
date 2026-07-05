"""Spec 005 US1/US2 — name-on-faces gated by the event's name_policy (SC-002).

Server-side only: when the policy forbids it, a named member's face is
indistinguishable from an unclaimed guest (no name, no hint). The viewer's own
face is always "self" regardless of policy.
"""

import uuid

import pytest

from app.api.photos import read_photo_faces
from app.db.models import Account, DetectedFace, Event, FaceCluster, GalleryEntry, Membership, Photo

pytestmark = pytest.mark.asyncio

EMBED = [0.0] * 512


async def _account(session, email: str, name: str) -> Account:
    acc = Account(id=uuid.uuid4(), email=email, hashed_password="x", name=name, is_active=True)
    session.add(acc)
    await session.flush()
    return acc


async def _seed(session, *, name_policy: str):
    host = await _account(session, "host@example.com", "Host")
    a = await _account(session, "a@example.com", "Alice")
    b = await _account(session, "b@example.com", "Bob")

    event = Event(
        id=uuid.uuid4(),
        name="Wedding",
        owner_id=host.id,
        join_code=f"NP{uuid.uuid4().hex[:6].upper()}",
        join_token=str(uuid.uuid4()),
        name_policy=name_policy,
    )
    session.add(event)
    await session.flush()
    session.add_all(
        [
            Membership(event_id=event.id, account_id=host.id, role="host"),
            Membership(event_id=event.id, account_id=a.id, role="member"),
            Membership(event_id=event.id, account_id=b.id, role="member"),
        ]
    )

    photo = Photo(
        id=uuid.uuid4(),
        event_id=event.id,
        contributor_id=host.id,
        storage_key="k",
        width=100,
        height=100,
        processing_status="done",
    )
    session.add(photo)
    await session.flush()

    cluster_a = FaceCluster(id=uuid.uuid4(), event_id=event.id, centroid=EMBED, claimed_by_account_id=a.id)
    cluster_stranger = FaceCluster(id=uuid.uuid4(), event_id=event.id, centroid=EMBED)
    session.add_all([cluster_a, cluster_stranger])
    await session.flush()

    session.add_all(
        [
            DetectedFace(
                photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 10, "h": 10}, det_score=0.9,
                embedding=EMBED, cluster_id=cluster_a.id,
            ),
            DetectedFace(
                photo_id=photo.id, bbox={"x": 20, "y": 0, "w": 10, "h": 10}, det_score=0.9,
                embedding=EMBED, cluster_id=cluster_stranger.id,
            ),
        ]
    )
    # Both Alice and Bob are verified in this group photo (gallery access) so
    # either can legitimately open the face overlay — the thing under test is
    # what NAME they see, not whether they can see the photo at all.
    session.add_all(
        [
            GalleryEntry(account_id=a.id, event_id=event.id, photo_id=photo.id, origin="auto"),
            GalleryEntry(account_id=b.id, event_id=event.id, photo_id=photo.id, origin="auto"),
        ]
    )
    await session.flush()
    return host, a, b, photo


def _by_x(faces):
    return sorted(faces, key=lambda f: f.x)


async def test_policy_nobody_hides_names_from_everyone(db_session):
    host, a, b, photo = await _seed(db_session, name_policy="nobody")

    for viewer in (host, b):
        faces = _by_x(await read_photo_faces(photo.id, user=viewer, session=db_session))
        assert faces[0].name is None  # Alice's claimed face — hidden by policy
        assert faces[0].is_me is False
        assert faces[1].name is None  # stranger — always anonymous

    faces_as_a = _by_x(await read_photo_faces(photo.id, user=a, session=db_session))
    assert faces_as_a[0].is_me is True
    assert faces_as_a[0].name is None  # self doesn't need a name field


async def test_policy_host_only_shows_names_to_host_alone(db_session):
    host, a, b, photo = await _seed(db_session, name_policy="host_only")

    host_faces = _by_x(await read_photo_faces(photo.id, user=host, session=db_session))
    assert host_faces[0].name == "Alice"

    member_faces = _by_x(await read_photo_faces(photo.id, user=b, session=db_session))
    assert member_faces[0].name is None


async def test_policy_everyone_shows_names_to_all_members(db_session):
    host, a, b, photo = await _seed(db_session, name_policy="everyone")

    for viewer in (host, b):
        faces = _by_x(await read_photo_faces(photo.id, user=viewer, session=db_session))
        assert faces[0].name == "Alice"
        assert faces[1].name is None  # unclaimed stays anonymous under every policy


async def test_new_event_defaults_to_nobody(db_session):
    host = await _account(db_session, "hd@example.com", "Hd")
    event = Event(name="x", owner_id=host.id, join_code="DEFPOL", join_token=str(uuid.uuid4()))
    db_session.add(event)
    await db_session.flush()
    assert event.name_policy == "nobody"
