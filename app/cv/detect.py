"""Face detection + alignment (SCRFD, via InsightFace).

InsightFace's ``buffalo_l`` runs SCRFD detection and ArcFace embedding in a single
pass, so each detection already carries its embedding; ``embed.py`` normalizes it.
This module owns the detection outputs written to DetectedFace rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.cv.models import get_face_app


@dataclass
class FaceDetection:
    bbox: dict  # {x, y, w, h}
    landmarks: list  # 5x2 keypoints
    det_score: float
    face_area_ratio: float
    embedding: np.ndarray  # raw 512-d ArcFace embedding (pre-normalization)


def detect_faces(image_bgr: np.ndarray) -> list[FaceDetection]:
    """Detect faces in a BGR image and return detections with embeddings."""
    face_app = get_face_app()
    img_h, img_w = image_bgr.shape[:2]
    img_area = float(img_h * img_w) or 1.0

    detections: list[FaceDetection] = []
    for face in face_app.get(image_bgr):
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        detections.append(
            FaceDetection(
                bbox={"x": x1, "y": y1, "w": w, "h": h},
                landmarks=face.kps.tolist() if getattr(face, "kps", None) is not None else [],
                det_score=float(face.det_score),
                face_area_ratio=(w * h) / img_area,
                embedding=np.asarray(face.embedding, dtype=np.float32),
            )
        )
    return detections
