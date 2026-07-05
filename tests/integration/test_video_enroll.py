"""US1 — video enrollment (spec 003): success writes one identity centroid from
>=3 quality-surviving frames; a low-quality clip fails without any partial state.

Runs ``process_video_enrollment`` for real against the same Postgres the async
``db_session`` fixture migrated (via a sync engine pointed at that test
database), with ``detect_faces``/``sample_frames`` faked so the test doesn't
need the real InsightFace model — same "model-free where possible" convention
as the rest of this suite.
"""

import uuid
from dataclasses import dataclass

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.db.models import Account, Event, IdentityEnrollment, Membership
from app.storage.base import get_storage
from app.workers import pipeline

pytestmark = pytest.mark.asyncio


@dataclass
class _FakeFrame:
    image_bgr: np.ndarray
    frame_index: int
    ts_s: float


@dataclass
class _FakeDetection:
    bbox: dict
    face_area_ratio: float
    det_score: float


def _frames(n: int) -> list[_FakeFrame]:
    return [
        _FakeFrame(image_bgr=np.zeros((4, 4, 3), dtype=np.uint8), frame_index=i, ts_s=i / 3.0)
        for i in range(n)
    ]


@pytest.fixture
def sync_pipeline_session(monkeypatch):
    """Point the worker's sync session at the same `<db>_test` database the
    async db_session fixture just migrated, instead of the dev database."""
    url = make_url(get_settings().alembic_database_url)
    test_url = url.set(database=f"{url.database}_test")
    engine = create_engine(test_url.render_as_string(hide_password=False))
    monkeypatch.setattr(pipeline, "SyncSessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    yield
    engine.dispose()


async def _setup_account_event(session):
    account = Account(
        id=uuid.uuid4(), email="v@example.com", hashed_password="x", name="v", is_active=True
    )
    session.add(account)
    await session.flush()
    event = Event(id=uuid.uuid4(), name="E", owner_id=account.id, join_code="VID1", join_token="t")
    session.add(event)
    await session.flush()
    session.add(Membership(event_id=event.id, account_id=account.id, role="member"))
    await session.commit()
    return account, event


async def test_video_enrollment_success(db_session, monkeypatch, sync_pipeline_session):
    account, event = await _setup_account_event(db_session)

    storage = get_storage()
    key = f"test-enroll-video/{uuid.uuid4()}"
    storage.put(key, b"fake mp4 bytes")

    monkeypatch.setattr(pipeline, "sample_frames", lambda data, fps, max_frames: _frames(5))
    monkeypatch.setattr(
        pipeline,
        "detect_faces",
        lambda bgr: [_FakeDetection(bbox={"x": 0, "y": 0, "w": 10, "h": 10}, face_area_ratio=0.2, det_score=0.9)],
    )
    monkeypatch.setattr(pipeline, "face_crop_sharpness", lambda bgr, bbox: 999.0)
    monkeypatch.setattr(
        pipeline, "normalized_embedding", lambda det: np.ones(512, dtype=np.float32) / np.sqrt(512)
    )

    account_id = account.id  # capture before expiry (test session, async ORM)
    result = pipeline.process_video_enrollment(str(account_id), str(event.id), key)

    assert result["enrolled"] is True
    assert result["sample_count"] == 5
    with pytest.raises(FileNotFoundError):
        storage.get(key)

    db_session.expire_all()
    enrollment = await db_session.scalar(
        select(IdentityEnrollment).where(IdentityEnrollment.account_id == account_id)
    )
    assert enrollment is not None
    assert enrollment.sample_count == 5


async def test_video_enrollment_low_quality_leaves_no_partial_state(
    db_session, monkeypatch, sync_pipeline_session
):
    account, event = await _setup_account_event(db_session)

    storage = get_storage()
    key = f"test-enroll-video/{uuid.uuid4()}"
    storage.put(key, b"fake mp4 bytes")

    monkeypatch.setattr(pipeline, "sample_frames", lambda data, fps, max_frames: _frames(5))
    # No detections in any frame → zero survivors, well under the 3-sample floor.
    monkeypatch.setattr(pipeline, "detect_faces", lambda bgr: [])

    account_id = account.id  # capture before expiry (test session, async ORM)
    result = pipeline.process_video_enrollment(str(account_id), str(event.id), key)

    assert result == {"enrolled": False, "reason": "low_quality_video", "gallery_entries": 0}
    with pytest.raises(FileNotFoundError):
        storage.get(key)

    db_session.expire_all()
    enrollment = await db_session.scalar(
        select(IdentityEnrollment).where(IdentityEnrollment.account_id == account_id)
    )
    assert enrollment is None
