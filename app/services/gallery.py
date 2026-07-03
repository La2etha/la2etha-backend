"""Gallery materialization (sync) — link matched clusters and create entries.

Given an account's enrollment for an event, match its identity centroid against
the event's cluster centroids (O(M+K)) and materialize GalleryEntry rows for the
photos containing those clusters' faces — excluding background faces (F4) and
culled photos (F3). Used by both the enrollment job and the photo pipeline.
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
    GalleryEntry,
    IdentityEnrollment,
    Photo,
)

settings = get_settings()


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

    # Collect photos containing non-background faces from the matched clusters.
    rows = session.execute(
        select(DetectedFace.photo_id, DetectedFace.cluster_id)
        .join(Photo, Photo.id == DetectedFace.photo_id)
        .where(
            DetectedFace.cluster_id.in_(list(matched_confidence.keys())),
            DetectedFace.is_background.is_(False),
            Photo.quality_verdict != "culled",
        )
    ).all()

    photo_to_conf: dict[uuid.UUID, float] = {}
    for photo_id, cluster_id in rows:
        conf = matched_confidence.get(cluster_id, 0.0)
        if photo_id not in photo_to_conf or conf > photo_to_conf[photo_id]:
            photo_to_conf[photo_id] = conf

    if not photo_to_conf:
        return 0

    existing = set(
        session.scalars(
            select(GalleryEntry.photo_id).where(
                GalleryEntry.account_id == account_id,
                GalleryEntry.photo_id.in_(list(photo_to_conf.keys())),
            )
        ).all()
    )

    created = 0
    for photo_id, conf in photo_to_conf.items():
        if photo_id in existing:
            continue
        session.add(
            GalleryEntry(
                account_id=account_id,
                event_id=event_id,
                photo_id=photo_id,
                origin="auto",
                confidence=conf,
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
