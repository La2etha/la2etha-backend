"""RQ pipeline jobs: photo processing (detect→embed→cluster) and enrollment.

All heavy CV runs here on the worker (GPU, sync). The API only enqueues jobs and
reports progress via ``job.meta`` (SC-006). After clustering, galleries are
re-materialized so newly added photos flow into the right people's galleries.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np
from rq import get_current_job
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.cv.detect import detect_faces
from app.cv.embed import normalized_embedding
from app.cv.enroll import aggregate_enrollment
from app.cv.image import load_image, rgb_to_bgr
from app.cv.proximity import face_crop_sharpness, is_background_face
from app.cv.quality import assess_photo_quality
from app.cv.search_embed import embed_image, search_available
from app.cv.video import pick_poster, probe, sample_frames
from app.db.models import DetectedFace, FaceCluster, IdentityEnrollment, Photo
from app.db.sync import SyncSessionLocal
from app.services.curation import refresh_curation
from app.services.gallery import materialize_gallery, rematerialize_event, reset_auto_match
from app.storage.base import get_storage

settings = get_settings()


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
    index_search = search_available()  # checked once; SigLIP is optional (F5)

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
                if photo.media_type == "video":
                    faces_written += _process_video_photo(session, storage, photo, data)
                else:
                    faces_written += _process_still_photo(session, photo, data, index_search)
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

        refresh_curation(session, event_uuid)
        session.commit()

    return {"photos": total, "faces": faces_written, "gallery_entries": new_entries}


def _process_still_photo(session: Session, photo: Photo, data: bytes, index_search: bool) -> int:
    """Existing image path: detect→embed→quality-cull→(optional) search-index.
    Returns the number of faces written."""
    rgb, w, h, oriented = load_image(data)
    photo.width, photo.height, photo.orientation_applied = w, h, oriented
    bgr = rgb_to_bgr(rgb)

    detections = detect_faces(bgr)
    for det in detections:
        _write_detected_face(session, photo.id, bgr, det)

    # Quality culling (F3) — records a verdict; never removes the photo.
    verdict = assess_photo_quality(bgr, [d.det_score for d in detections], settings.quality_blur_min)
    photo.quality_score = verdict.score
    photo.quality_verdict = verdict.verdict
    photo.cull_reason = verdict.reason

    # Semantic-search index (F5) — best-effort: if SigLIP isn't installed, or
    # fails, the photo just won't be searchable.
    if index_search:
        try:
            photo.search_embedding = embed_image(rgb).tolist()
        except Exception:
            pass

    return len(detections)


def _process_video_photo(session: Session, storage, photo: Photo, data: bytes) -> int:
    """Video branch (spec 003 US2): sample frames, detect/embed each (faces carry
    frame_index/frame_ts_s so they roll up to this one photo row exactly like a
    still's faces do), write a poster frame, and quality-cull off that poster.
    Returns the number of faces written."""
    frames = sample_frames(data, fps=2.0, max_frames=120)
    if not frames:
        raise ValueError("undecodable video")

    h, w = frames[0].image_bgr.shape[:2]
    photo.width, photo.height = w, h

    faces_written = 0
    for frame in frames:
        for det in detect_faces(frame.image_bgr):
            _write_detected_face(
                session, photo.id, frame.image_bgr, det, frame_index=frame.frame_index, frame_ts_s=frame.ts_s
            )
            faces_written += 1

    poster = pick_poster(frames)
    if poster is not None:
        ok, encoded = cv2.imencode(".jpg", poster.image_bgr)
        if ok:
            poster_key = f"events/{photo.event_id}/{photo.id}/poster"
            storage.put(poster_key, encoded.tobytes())
            photo.poster_key = poster_key

        verdict = assess_photo_quality(poster.image_bgr, None, settings.quality_blur_min)
        photo.quality_score = verdict.score
        photo.quality_verdict = verdict.verdict
        photo.cull_reason = verdict.reason

    return faces_written


def _write_detected_face(
    session: Session,
    photo_id: uuid.UUID,
    image_bgr: np.ndarray,
    det,
    frame_index: int | None = None,
    frame_ts_s: float | None = None,
) -> None:
    sharpness = face_crop_sharpness(image_bgr, det.bbox)
    background = is_background_face(
        det.face_area_ratio, sharpness, settings.proximity_area_min, settings.proximity_sharpness_min
    )
    session.add(
        DetectedFace(
            photo_id=photo_id,
            bbox=det.bbox,
            landmarks=det.landmarks,
            det_score=det.det_score,
            face_area_ratio=det.face_area_ratio,
            face_sharpness=sharpness,
            is_background=background,
            embedding=normalized_embedding(det).tolist(),
            frame_index=frame_index,
            frame_ts_s=frame_ts_s,
        )
    )


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

    result = _finish_enrollment(account_uuid, event_uuid, embeddings)
    for key in sample_keys:
        storage.delete(key)
    return result


def process_video_enrollment(account_id: str, event_id: str, video_key: str) -> dict:
    """Video variant of enrollment (spec 003 US1): sample ~10 frames from a short
    selfie clip, reuse the same per-frame face/quality path as photo enrollment,
    and require >=3 surviving frames else fail ``low_quality_video``. The clip
    itself is always deleted after extraction (privacy) — only the resulting
    embeddings persist, exactly like photo samples."""
    account_uuid = uuid.UUID(account_id)
    event_uuid = uuid.UUID(event_id)
    storage = get_storage()

    embeddings: list[np.ndarray] = []
    try:
        data = storage.get(video_key)
        for frame in sample_frames(data, fps=3.0, max_frames=10):
            detections = detect_faces(frame.image_bgr)
            if not detections:
                continue
            subject = max(detections, key=lambda d: (d.face_area_ratio, d.det_score))
            sharpness = face_crop_sharpness(frame.image_bgr, subject.bbox)
            if sharpness < settings.quality_blur_min:
                continue
            embeddings.append(normalized_embedding(subject))
    finally:
        storage.delete(video_key)

    if len(embeddings) < 3:
        return {"enrolled": False, "reason": "low_quality_video", "gallery_entries": 0}

    return _finish_enrollment(account_uuid, event_uuid, embeddings)


def _finish_enrollment(
    account_uuid: uuid.UUID, event_uuid: uuid.UUID, embeddings: list[np.ndarray]
) -> dict:
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

        # Re-enrollment must fully REPLACE the prior identity match, not add to it.
        reset_auto_match(session, account_uuid, event_uuid)

        new_entries = materialize_gallery(session, account_uuid, event_uuid)
        session.commit()

    return {
        "enrolled": True,
        "quality_ok": aggregate.quality_ok,
        "sample_count": aggregate.sample_count,
        "gallery_entries": new_entries,
    }


def ingest_gdrive(
    event_id: str,
    contributor_id: str,
    access_token: str,
    file_ids: list[str] | None = None,
    folder_id: str | None = None,
) -> dict:
    """Download images from Google Drive into the event pool, dedup by phash
    (FR-006), store them, then enqueue the normal processing pipeline.

    ponytail: the short-lived Drive token rides in the job args (Redis); fine for
    the demo — a hardening pass would keep it out of the queue payload.
    """
    from app.cv.image import dhash
    from app.storage.gdrive import download_file, list_folder_images
    from app.workers import get_queue

    event_uuid = uuid.UUID(event_id)
    contributor_uuid = uuid.UUID(contributor_id)
    storage = get_storage()

    ids = list(file_ids or [])
    if folder_id:
        ids.extend(f.id for f in list_folder_images(folder_id, access_token))

    new_photo_ids: list[uuid.UUID] = []
    duplicates = 0
    with SyncSessionLocal() as session:
        for fid in ids:
            try:
                data = download_file(fid, access_token)
            except Exception:
                continue
            if not data:
                continue
            phash = dhash(data)
            clash = session.scalar(
                select(Photo.id).where(Photo.event_id == event_uuid, Photo.phash == phash)
            )
            if clash is not None:
                duplicates += 1
                continue
            photo = Photo(
                event_id=event_uuid,
                contributor_id=contributor_uuid,
                storage_key="",
                source="gdrive",
                phash=phash,
                processing_status="pending",
            )
            session.add(photo)
            session.flush()
            key = f"events/{event_uuid}/{photo.id}"
            storage.put(key, data)
            photo.storage_key = key
            new_photo_ids.append(photo.id)
        session.commit()

    process_job_id = ""
    if new_photo_ids:
        job = get_queue().enqueue(
            "app.workers.pipeline.process_photos",
            str(event_uuid),
            [str(pid) for pid in new_photo_ids],
        )
        process_job_id = job.id

    return {
        "ingested": len(new_photo_ids),
        "duplicates": duplicates,
        "process_job_id": process_job_id,
    }
