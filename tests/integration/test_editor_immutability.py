"""Spec 006 (SC-003) — classic/AI photo editing never mutates the pool original.

Editing is entirely client-side (classic) or reads-then-returns-new-bytes
(AI/export) — the backend's `storage_key` for a photo must never be
overwritten by an edit session. This pins that invariant at the storage layer
so a future change that adds a `storage.put(...)` to the read path would fail
the test rather than silently corrupting every contributor's original.
"""

import hashlib
import uuid

import pytest

from app.access.guard import require_photo_read
from app.api.export import ExportRequest, export_photo
from app.db.models import Account, Event, Membership, Photo
from app.storage.local_fs import LocalFilesystemStorage

pytestmark = pytest.mark.asyncio


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def test_export_default_path_leaves_the_original_byte_identical(db_session, tmp_path, monkeypatch):
    storage = LocalFilesystemStorage(str(tmp_path))
    monkeypatch.setattr("app.api.export.get_storage", lambda: storage)

    owner = Account(id=uuid.uuid4(), email="owner@example.com", hashed_password="x", name="Owner", is_active=True)
    db_session.add(owner)
    await db_session.flush()
    event = Event(id=uuid.uuid4(), name="E", owner_id=owner.id, join_code="IMMU1", join_token="t")
    db_session.add(event)
    await db_session.flush()
    db_session.add(Membership(event_id=event.id, account_id=owner.id, role="host", status="active"))

    original = b"\xff\xd8\xff\xe0 pretend-jpeg-bytes for a pool original"
    key = f"{uuid.uuid4()}.jpg"
    storage.put(key, original)
    before = _checksum(storage.get(key))

    photo = Photo(id=uuid.uuid4(), event_id=event.id, contributor_id=owner.id, storage_key=key)
    db_session.add(photo)
    await db_session.flush()

    # Sanity check the access guard resolves (host reading their own event's photo).
    resolved = await require_photo_read(db_session, owner.id, photo.id)
    assert resolved.id == photo.id

    response = await export_photo(
        photo_id=photo.id,
        payload=ExportRequest(remove_strangers=False),
        user=owner,
        session=db_session,
    )
    assert response.body == original  # default path: original bytes, untouched

    after = _checksum(storage.get(key))
    assert after == before, "the pool original must be byte-identical after an export/edit session (SC-003)"
