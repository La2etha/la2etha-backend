"""US2 (F4) — proximity filter marks small/soft faces as background bystanders."""

import numpy as np

from app.cv.proximity import face_crop_sharpness, is_background_face

AREA_MIN = 0.012
SHARP_MIN = 30.0


def test_large_sharp_face_is_subject():
    assert not is_background_face(0.09, 600.0, AREA_MIN, SHARP_MIN)


def test_tiny_face_is_background():
    # A distant bystander occupying < 1.2% of the frame.
    assert is_background_face(0.003, 900.0, AREA_MIN, SHARP_MIN)


def test_smallish_and_soft_face_is_background():
    # Mid-small AND clearly out of focus → background depth-of-field.
    assert is_background_face(0.018, 8.0, AREA_MIN, SHARP_MIN)


def test_smallish_but_sharp_face_is_subject():
    assert not is_background_face(0.018, 500.0, AREA_MIN, SHARP_MIN)


def test_missing_area_defaults_to_subject():
    # Older rows without a measured area are conservatively treated as subjects.
    assert not is_background_face(None, None, AREA_MIN, SHARP_MIN)


def test_face_crop_sharpness_handles_empty_and_oob_boxes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert face_crop_sharpness(img, {"x": 0, "y": 0, "w": 0, "h": 0}) == 0.0
    # Out-of-bounds box is clamped; a flat crop has zero variance.
    assert face_crop_sharpness(img, {"x": 90, "y": 90, "w": 40, "h": 40}) == 0.0
