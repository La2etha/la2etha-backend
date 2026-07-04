"""Gallery materialization (sync) — link matched clusters and create entries.

Given an account's enrollment for an event, match its identity centroid against
the event's cluster centroids (O(M+K)) and materialize GalleryEntry rows for the
photos containing those clusters' faces. Culled photos (F3) and background-only
presence (F4) are DEMOTED to ``relevance='low'`` with a ``demote_reason`` rather
than excluded (FR-012/013). Used by both the enrollment job and the photo pipeline.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.cv.enroll import match_centroid_to_clusters
from app.db.models import (
    DetectedFace,
    FaceCluster,
    GalleryClaim,
    GalleryEntry,
    IdentityEnrollment,
    Photo,
)

settings = get_settings()


def reset_auto_match(session: Session, account_id: uuid.UUID, event_id: uuid.UUID) -> None:
    """Undo an account's automatic identity match in an event.

    Deletes its auto-created gallery entries and unlinks any clusters it claimed,
    so a re-enrollment fully REPLACES the previous match instead of accumulating
    stale photos from a wrongly-matched person. Manual claims (origin="claim",
    FR-018) are preserved — those are explicit user assertions, not auto matches.
    """
    session.query(GalleryEntry).filter(
        GalleryEntry.account_id == account_id,
        GalleryEntry.event_id == event_id,
        GalleryEntry.origin == "auto",
    ).delete(synchronize_session=False)
    session.query(FaceCluster).filter(
        FaceCluster.event_id == event_id,
        FaceCluster.claimed_by_account_id == account_id,
    ).update({FaceCluster.claimed_by_account_id: None}, synchronize_session=False)
    session.flush()


def materialize_gallery(session: Session, account_id: uuid.UUID, event_id: uuid.UUID) -> int:
    """Match one account's enrollment to event clusters and create gallery
    entries for the photos it appears in. Returns the number of new entries."""
    enrollment = session.scalar(
        select(IdentityEnrollment).where(
            IdentityEnrollment.account_id == account_id,
            IdentityEnrollment.event_id == event_id,
        )
    )
    if enrollment is None:
        return 0

    clusters = session.scalars(
        select(FaceCluster).where(FaceCluster.event_id == event_id)
    ).all()
    if not clusters:
        return 0

    cluster_centroids = {c.id: np.asarray(c.centroid, dtype=np.float32) for c in clusters}
    per_angle = [np.asarray(e, dtype=np.float32) for e in (enrollment.per_angle_embeddings or [])]

    matches = match_centroid_to_clusters(
        enroll_centroid=np.asarray(enrollment.centroid, dtype=np.float32),
        per_angle=per_angle,
        cluster_centroids=cluster_centroids,
        threshold=settings.enroll_match_threshold,
    )
    if not matches:
        return 0

    matched_confidence = {cid: sim for cid, sim in matches}

    # Link the matched clusters to this account (anonymous cluster → identity).
    for cluster in clusters:
        if cluster.id in matched_confidence:
            cluster.claimed_by_account_id = account_id

    # Collect EVERY photo containing a matched face — including culled and
    # background ones. F3/F4 no longer exclude; they DEMOTE (FR-012/013), so we
    # keep per-face is_background and the photo's cull verdict to decide relevance.
    rows = session.execute(
        select(
            DetectedFace.photo_id,
            DetectedFace.cluster_id,
            DetectedFace.is_background,
            Photo.quality_verdict,
            Photo.cull_reason,
        )
        .join(Photo, Photo.id == DetectedFace.photo_id)
        .where(DetectedFace.cluster_id.in_(list(matched_confidence.keys())))
    ).all()
    if not rows:
        return 0

    # Aggregate the matched faces per photo: a photo is a "main" subject photo iff
    # it passed quality culling AND at least one of this person's matched faces in
    # it is a foreground subject. Otherwise it is demoted with the strongest reason.
    per_photo: dict[uuid.UUID, dict] = {}
    for photo_id, cluster_id, is_background, quality_verdict, cull_reason in rows:
        conf = matched_confidence.get(cluster_id, 0.0)
        agg = per_photo.setdefault(
            photo_id,
            {"conf": 0.0, "foreground": False, "culled": False, "cull_reason": None},
        )
        agg["conf"] = max(agg["conf"], conf)
        if not is_background:
            agg["foreground"] = True
        if quality_verdict == "culled":
            agg["culled"] = True
            agg["cull_reason"] = cull_reason

    def _relevance(agg: dict) -> tuple[str, str | None]:
        if agg["culled"]:
            return "low", agg["cull_reason"] or "low_quality"
        if not agg["foreground"]:
            return "low", "background"
        return "main", None

    existing = set(
        session.scalars(
            select(GalleryEntry.photo_id).where(
                GalleryEntry.account_id == account_id,
                GalleryEntry.photo_id.in_(list(per_photo.keys())),
            )
        ).all()
    )
    # Photos the caller manually rejected ("not me", FR-018) stay out — never
    # resurrected by an auto match.
    unclaimed = set(
        session.scalars(
            select(GalleryClaim.photo_id).where(
                GalleryClaim.account_id == account_id,
                GalleryClaim.state == "unclaimed",
                GalleryClaim.photo_id.in_(list(per_photo.keys())),
            )
        ).all()
    )

    created = 0
    for photo_id, agg in per_photo.items():
        if photo_id in existing or photo_id in unclaimed:
            continue
        relevance, reason = _relevance(agg)
        session.add(
            GalleryEntry(
                account_id=account_id,
                event_id=event_id,
                photo_id=photo_id,
                origin="auto",
                relevance=relevance,
                demote_reason=reason,
                confidence=agg["conf"],
            )
        )
        created += 1

    session.flush()
    return created


def rematerialize_event(session: Session, event_id: uuid.UUID) -> int:
    """Re-run gallery materialization for every enrollment in the event
    (called after new photos are clustered). Returns total new entries."""
    account_ids = session.scalars(
        select(IdentityEnrollment.account_id).where(
            IdentityEnrollment.event_id == event_id
        )
    ).all()
    total = 0
    for account_id in account_ids:
        total += materialize_gallery(session, account_id, event_id)
    return total
