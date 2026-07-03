"""RQ pipeline jobs: photo processing (detect→embed→cluster) and enrollment.

All heavy CV runs here on the worker (GPU, sync). The API only enqueues jobs and
reports progress via ``job.meta`` (SC-006). After clustering, galleries are
re-materialized so newly added photos flow into the right people's galleries.
"""

from __future__ import annotations

import uuid

import numpy as np
from rq import get_current_job
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cv.detect import detect_faces
from app.cv.embed import normalized_embedding
from app.cv.enroll import aggregate_enrollment
from app.cv.image import load_image, rgb_to_bgr
from app.db.models import DetectedFace, FaceCluster, IdentityEnrollment, Photo
from app.db.sync import SyncSessionLocal
from app.services.gallery import materialize_gallery, rematerialize_event
from app.storage.base import get_storage


def _set_progress(processed: int, total: int) -> None:
    job = get_current_job()
    if job is None:
        return
    job.meta["processed"] = processed
    job.meta["total"] = total
    job.meta["progress"] = round(processed / total, 3) if total else 1.0
    job.save_meta()


def process_photos(event_id: str, photo_ids: list[str]) -> dict:
    """Detect + embed faces for the given photos, re-cluster the event, and
    re-materialize galleries. Returns a small summary dict."""
    event_uuid = uuid.UUID(event_id)
    storage = get_storage()
    total = len(photo_ids)
    faces_written = 0

    with SyncSessionLocal() as session:
        for idx, pid in enumerate(photo_ids, start=1):
            photo = session.get(Photo, uuid.UUID(pid))
            if photo is None:
                _set_progress(idx, total)
                continue
            photo.processing_status = "processing"
            session.flush()
            try:
                data = storage.get(photo.storage_key)
                rgb, w, h, oriented = load_image(data)
                photo.width, photo.height, photo.orientation_applied = w, h, oriented
                for det in detect_faces(rgb_to_bgr(rgb)):
                    session.add(
                        DetectedFace(
                            photo_id=photo.id,
                            bbox=det.bbox,
                            landmarks=det.landmarks,
                            det_score=det.det_score,
                            face_area_ratio=det.face_area_ratio,
                            embedding=normalized_embedding(det).tolist(),
                        )
                    )
                    faces_written += 1
                photo.processing_status = "done"
            except Exception:
                photo.processing_status = "failed"
            session.flush()
            _set_progress(idx, total)

        session.commit()

        _recluster_event(session, event_uuid)
        session.commit()

        new_entries = rematerialize_event(session, event_uuid)
        session.commit()

    return {"photos": total, "faces": faces_written, "gallery_entries": new_entries}


def _recluster_event(session: Session, event_id: uuid.UUID) -> None:
    """Full re-cluster of an event's faces. ponytail: full recluster per batch
    is fine at event scale; incremental clustering is the scale-up path."""
    from app.cv.cluster import cluster_embeddings

    faces = session.scalars(
        select(DetectedFace)
        .join(Photo, Photo.id == DetectedFace.photo_id)
        .where(Photo.event_id == event_id)
    ).all()
    if not faces:
        return

    embeddings = np.stack([np.asarray(f.embedding, dtype=np.float32) for f in faces])
    result = cluster_embeddings(embeddings)

    # Replace clusters for this event; DetectedFace.cluster_id resets to null.
    session.query(FaceCluster).filter(FaceCluster.event_id == event_id).delete()
    session.flush()

    label_to_cluster: dict[int, FaceCluster] = {}
    for label, centroid_vec in result.centroids.items():
        cluster = FaceCluster(
            event_id=event_id,
            centroid=centroid_vec.tolist(),
            member_count=int((result.labels == label).sum()),
            algo="hdbscan",
        )
        session.add(cluster)
        label_to_cluster[label] = cluster
    session.flush()

    for face, label in zip(faces, result.labels):
        face.cluster_id = label_to_cluster[int(label)].id if int(label) != -1 else None
    session.flush()


def process_enrollment(account_id: str, event_id: str, sample_keys: list[str]) -> dict:
    """Build an identity centroid from enrollment samples, store it, and
    materialize the caller's gallery. Deletes the temporary sample bytes."""
    account_uuid = uuid.UUID(account_id)
    event_uuid = uuid.UUID(event_id)
    storage = get_storage()

    embeddings: list[np.ndarray] = []
    for key in sample_keys:
        try:
            rgb, _, _, _ = load_image(storage.get(key))
        except Exception:
            continue
        detections = detect_faces(rgb_to_bgr(rgb))
        if not detections:
            continue
        # The subject is the largest, most-confident face in an enrollment shot.
        subject = max(detections, key=lambda d: (d.face_area_ratio, d.det_score))
        embeddings.append(normalized_embedding(subject))

    if not embeddings:
        return {"enrolled": False, "reason": "no_face_detected", "gallery_entries": 0}

    aggregate = aggregate_enrollment(embeddings)

    with SyncSessionLocal() as session:
        enrollment = session.scalar(
            select(IdentityEnrollment).where(
                IdentityEnrollment.account_id == account_uuid,
                IdentityEnrollment.event_id == event_uuid,
            )
        )
        if enrollment is None:
            enrollment = IdentityEnrollment(account_id=account_uuid, event_id=event_uuid)
            session.add(enrollment)
        enrollment.centroid = aggregate.centroid.tolist()
        enrollment.per_angle_embeddings = [e.tolist() for e in aggregate.per_angle]
        enrollment.sample_count = aggregate.sample_count
        enrollment.quality_ok = aggregate.quality_ok
        session.flush()

        new_entries = materialize_gallery(session, account_uuid, event_uuid)
        session.commit()

    for key in sample_keys:
        storage.delete(key)

    return {
        "enrolled": True,
        "quality_ok": aggregate.quality_ok,
        "sample_count": aggregate.sample_count,
        "gallery_entries": new_entries,
    }
