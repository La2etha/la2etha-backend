"""Video decode + frame sampling (spec 003) via ``cv2.VideoCapture`` — no new
heavy dependency. OpenCV can't decode from an in-memory buffer for arbitrary
container formats, so each call spills bytes to a temp file for the capture's
lifetime.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

from app.cv.quality import variance_of_laplacian


@dataclass
class ProbeResult:
    ok: bool
    duration_s: float | None = None


@dataclass
class SampledFrame:
    image_bgr: np.ndarray
    frame_index: int
    ts_s: float


@contextlib.contextmanager
def _temp_video(data: bytes):
    fd, path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        yield path
    finally:
        os.unlink(path)


def probe(data: bytes) -> ProbeResult:
    """Decodability + duration check; ``ok=False`` for anything unreadable."""
    with _temp_video(data) as path:
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return ProbeResult(ok=False)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if fps <= 0 or frame_count <= 0:
                return ProbeResult(ok=False)
            return ProbeResult(ok=True, duration_s=frame_count / fps)
        finally:
            cap.release()


def sample_frames(data: bytes, fps: float = 2.0, max_frames: int = 120) -> list[SampledFrame]:
    """Decode ~``fps`` frames per second of source video, capped at ``max_frames``."""
    with _temp_video(data) as path:
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return []
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if src_fps <= 0:
                return []
            step = max(1, round(src_fps / fps))

            frames: list[SampledFrame] = []
            idx = 0
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    frames.append(
                        SampledFrame(image_bgr=frame, frame_index=idx, ts_s=idx / src_fps)
                    )
                idx += 1
            return frames
        finally:
            cap.release()


def pick_poster(frames: list[SampledFrame]) -> SampledFrame | None:
    """Sharpest frame in the middle third of the clip (avoids fade-in/out ends)."""
    if not frames:
        return None
    n = len(frames)
    middle = frames[n // 3 : n - n // 3] or frames
    return max(middle, key=lambda f: variance_of_laplacian(f.image_bgr, normalize=True))
