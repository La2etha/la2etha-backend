"""US1/US2 — video decode + frame sampling on a tiny synthetic clip.

Generates a real, tiny MP4 with ``cv2.VideoWriter`` so the test exercises the
actual OpenCV decode path rather than mocking it.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from app.cv.video import pick_poster, probe, sample_frames

WIDTH, HEIGHT, SRC_FPS, N_FRAMES = 64, 64, 10.0, 20


def _make_clip_bytes() -> bytes:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), SRC_FPS, (WIDTH, HEIGHT)
        )
        rng = np.random.default_rng(0)
        for _ in range(N_FRAMES):
            frame = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def test_probe_reads_duration():
    result = probe(_make_clip_bytes())
    assert result.ok
    assert result.duration_s == pytest.approx(N_FRAMES / SRC_FPS, abs=0.15)


def test_probe_rejects_garbage():
    result = probe(b"not a video")
    assert not result.ok
    assert result.duration_s is None


def test_sample_frames_respects_fps_and_cap():
    frames = sample_frames(_make_clip_bytes(), fps=2.0, max_frames=120)
    # ~10fps source sampled at 2fps over 2s → ~4 frames.
    assert 1 <= len(frames) <= 6
    assert all(f.image_bgr.shape == (HEIGHT, WIDTH, 3) for f in frames)


def test_sample_frames_max_frames_cap():
    frames = sample_frames(_make_clip_bytes(), fps=100.0, max_frames=5)
    assert len(frames) <= 5


def test_pick_poster_returns_a_middle_frame():
    frames = sample_frames(_make_clip_bytes(), fps=10.0, max_frames=120)
    poster = pick_poster(frames)
    assert poster is not None
    assert poster.frame_index in [f.frame_index for f in frames]


def test_pick_poster_empty_list():
    assert pick_poster([]) is None
