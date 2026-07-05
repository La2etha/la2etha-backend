"""Curation layer (spec 004): best-shot ranking, event highlights, auto cover
pick, and host stats. Signals-only — composes columns the pipeline already
writes (``photo.quality_score``, ``detected_face.det_score/face_area_ratio/
face_sharpness``, cluster membership). No new models, no new inference.

The pure scoring functions (``best_shot_score``, ``highlight_score``,
``passes_quality_bar``) take plain numbers so they're testable with seeded
fixtures, no DB required (research R1/R2). The ``async def`` functions below
them do the DB reads/writes for the read-time gallery ranking, the
post-pipeline highlights/cover refresh, and the host stats aggregate.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import DetectedFace, Event, FaceCluster, IdentityEnrollment, Photo

settings = get_settings()


def _norm(value: float | None, reference: float) -> float:
    """Clamp ``value / reference`` to [0, 1]. ``reference <= 0`` never happens
    (settings are positive by construction); a ``None`` signal scores 0."""
    if value is None or reference <= 0:
        return 0.0
    return max(0.0, min(value / reference, 1.0))


def best_shot_score(
    *,
    quality_score: float | None,
    det_score: float | None,
    face_area_ratio: float | None,
    face_sharpness: float | None,
    w_q: float = settings.curation_w_quality,
    w_d: float = settings.curation_w_det,
    w_a: float = settings.curation_w_area,
    w_s: float = settings.curation_w_sharpness,
    area_ref: float = settings.curation_area_ref,
    quality_ref: float | None = None,
) -> float:
    """R1: per-gallery-entry score for "best photos of you". Each signal is
    normalized to [0, 1] so the weights are directly interpretable."""
    quality_ref = quality_ref if quality_ref is not None else 3 * settings.quality_blur_min
    return (
        w_q * _norm(quality_score, quality_ref)
        + w_d * (det_score or 0.0)
        + w_a * _norm(face_area_ratio, area_ref)
        + w_s * _norm(face_sharpness, quality_ref)
    )


def passes_quality_bar(quality_verdict: str, quality_score: float | None, blur_min: float) -> bool:
    return quality_verdict == "ok" and (quality_score or 0.0) >= blur_min


def highlight_score(
    *,
    quality_score: float | None,
    n_confident_faces: int,
    quality_ref: float | None = None,
    group_bonus: float = settings.curation_group_bonus,
    group_min_faces: int = settings.curation_group_min_faces,
) -> float:
    """R2: event-level "highlight-worthiness" — favors sharp group moments over
    solo close-ups, with log damping so a crowd of blurry faces can't out-rank
    a crisp trio."""
    quality_ref = quality_ref if quality_ref is not None else 3 * settings.quality_blur_min
    bonus = group_bonus if n_confident_faces >= group_min_faces else 1.0
    return _norm(quality_score, quality_ref) * (1 + math.log2(1 + n_confident_faces)) * bonus


@dataclass
class HighlightCandidate:
    photo_id: uuid.UUID
    quality_score: float | None
    quality_verdict: str
    n_confident_faces: int
    created_at: datetime


def compute_highlights(candidates: list[HighlightCandidate], *, top_n: int = 10) -> list[uuid.UUID]:
    """R2/R4: rank candidates that clear the quality bar, ties broken by
    recency (newest first), return up to ``top_n`` photo ids in rank order."""
    eligible = [
        c for c in candidates if passes_quality_bar(c.quality_verdict, c.quality_score, settings.quality_blur_min)
    ]
    eligible.sort(key=lambda c: (highlight_score(quality_score=c.quality_score, n_confident_faces=c.n_confident_faces), c.created_at), reverse=True)
    return [c.photo_id for c in eligible[:top_n]]


# --------------------------------------------------------------------------- #
# DB-backed helpers                                                           #
# --------------------------------------------------------------------------- #
async def best_scores_for_account(
    session: AsyncSession, account_id: uuid.UUID, photo_ids: list[uuid.UUID]
) -> dict[uuid.UUID, float]:
    """Best-shot score per photo for one account's own matched face (R1). A
    photo with no matched face for this account (shouldn't happen for a gallery
    entry, but defensive) is omitted rather than scored 0."""
    if not photo_ids:
        return {}
    rows = (
        await session.execute(
            select(
                DetectedFace.photo_id,
                DetectedFace.det_score,
                DetectedFace.face_area_ratio,
                DetectedFace.face_sharpness,
                Photo.quality_score,
            )
            .join(Photo, Photo.id == DetectedFace.photo_id)
            .join(FaceCluster, FaceCluster.id == DetectedFace.cluster_id)
            .where(
                FaceCluster.claimed_by_account_id == account_id,
                DetectedFace.photo_id.in_(photo_ids),
            )
        )
    ).all()
    scores: dict[uuid.UUID, float] = {}
    for photo_id, det_score, face_area_ratio, face_sharpness, quality_score in rows:
        score = best_shot_score(
            quality_score=quality_score,
            det_score=det_score,
            face_area_ratio=face_area_ratio,
            face_sharpness=face_sharpness,
        )
        # A photo can have >1 matched face (rare); keep the best one.
        if photo_id not in scores or score > scores[photo_id]:
            scores[photo_id] = score
    return scores


def _sync_confident_face_counts(session: Session, photo_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    rows = session.execute(
        select(DetectedFace.photo_id, func.count(DetectedFace.id))
        .where(
            DetectedFace.photo_id.in_(photo_ids),
            DetectedFace.det_score >= settings.curation_confident_det_min,
            DetectedFace.face_area_ratio >= settings.curation_confident_area_min,
        )
        .group_by(DetectedFace.photo_id)
    ).all()
    return {photo_id: count for photo_id, count in rows}


def refresh_curation(session: Session, event_id: uuid.UUID) -> None:
    """Post-pipeline curation step (R3/T006): rewrite highlight flags for the
    whole event and auto-pick a cover, atomically (no partial strips). Never
    overwrites a host-chosen cover (FR-010)."""
    photos = session.scalars(select(Photo).where(Photo.event_id == event_id)).all()
    if not photos:
        return
    photo_ids = [p.id for p in photos]
    confident_counts = _sync_confident_face_counts(session, photo_ids)

    candidates = [
        HighlightCandidate(
            photo_id=p.id,
            quality_score=p.quality_score,
            quality_verdict=p.quality_verdict,
            n_confident_faces=confident_counts.get(p.id, 0),
            created_at=p.created_at,
        )
        for p in photos
    ]
    ranked = compute_highlights(candidates)
    ranked_set = {photo_id: rank for rank, photo_id in enumerate(ranked, start=1)}

    for p in photos:
        rank = ranked_set.get(p.id)
        p.is_highlight = rank is not None
        p.highlight_rank = rank

    event = session.get(Event, event_id)
    if event is not None and event.cover_source != "host":
        picked = ranked[0] if ranked else _best_quality_photo_id(photos)
        event.cover_photo_id = picked
        event.cover_source = "auto" if picked is not None else None
    session.flush()


def _best_quality_photo_id(photos: list[Photo]) -> uuid.UUID | None:
    ok_photos = [p for p in photos if p.quality_verdict == "ok"]
    pool = ok_photos or photos
    if not pool:
        return None
    return max(pool, key=lambda p: (p.quality_score or 0.0, p.created_at)).id


async def auto_pick_cover_photo_id(session: AsyncSession, event_id: uuid.UUID) -> uuid.UUID | None:
    """Async counterpart used by the cover-revert/dangling-fallback paths."""
    row = (
        await session.execute(
            select(Photo.id, Photo.highlight_rank)
            .where(Photo.event_id == event_id, Photo.is_highlight.is_(True))
            .order_by(Photo.highlight_rank)
            .limit(1)
        )
    ).first()
    if row is not None:
        return row[0]
    photos = (
        await session.execute(
            select(Photo.id, Photo.quality_score, Photo.quality_verdict, Photo.created_at).where(
                Photo.event_id == event_id
            )
        )
    ).all()
    if not photos:
        return None
    ok_photos = [p for p in photos if p.quality_verdict == "ok"] or list(photos)
    return max(ok_photos, key=lambda p: (p.quality_score or 0.0, p.created_at)).id


async def event_stats(session: AsyncSession, event_id: uuid.UUID) -> dict:
    """R7: host-only aggregate. Unclaimed clusters are counted, never itemized
    (anonymity by construction)."""
    photo_count = await session.scalar(
        select(func.count(Photo.id)).where(Photo.event_id == event_id)
    )
    processing = await session.scalar(
        select(func.count(Photo.id)).where(
            Photo.event_id == event_id, Photo.processing_status.in_(("pending", "processing"))
        )
    )
    cluster_count = await session.scalar(
        select(func.count(FaceCluster.id)).where(FaceCluster.event_id == event_id)
    )
    unclaimed_count = await session.scalar(
        select(func.count(FaceCluster.id)).where(
            FaceCluster.event_id == event_id, FaceCluster.claimed_by_account_id.is_(None)
        )
    )
    enrolled_count = await session.scalar(
        select(func.count(IdentityEnrollment.id)).where(IdentityEnrollment.event_id == event_id)
    )

    from app.services.membership import list_members

    members = await list_members(session, event_id)

    return {
        "photo_count": photo_count or 0,
        "cluster_count": cluster_count or 0,
        "enrolled_count": enrolled_count or 0,
        "unclaimed_count": unclaimed_count or 0,
        "processing": bool(processing),
        "members": [
            {
                "account_id": m["account_id"],
                "name": m["name"],
                "enrolled": m["enrolled"],
                "appearance_count": m["appearance_count"],
            }
            for m in members
            if m["role"] == "member"
        ],
    }


if __name__ == "__main__":  # runnable self-check (Constitution V)
    now = datetime.now()
    sharp_group = HighlightCandidate(uuid.uuid4(), 200.0, "ok", 3, now)
    sharp_solo = HighlightCandidate(uuid.uuid4(), 200.0, "ok", 1, now)
    blurry = HighlightCandidate(uuid.uuid4(), 10.0, "culled", 3, now)

    ranked = compute_highlights([sharp_group, sharp_solo, blurry])
    assert blurry.photo_id not in ranked, "culled photo must never be a highlight"
    assert ranked[0] == sharp_group.photo_id, "a sharp group shot should out-rank a solo shot"

    solo_score = best_shot_score(quality_score=200.0, det_score=0.9, face_area_ratio=0.05, face_sharpness=200.0)
    blurry_score = best_shot_score(quality_score=10.0, det_score=0.9, face_area_ratio=0.05, face_sharpness=10.0)
    assert solo_score > blurry_score, "a sharp close-up should score above a blurry one"
    print("curation self-check OK")
