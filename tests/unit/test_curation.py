"""US1/US2 (spec 004) — best-shot ranking and highlights scoring, over seeded
deterministic fixtures (no DB, no InsightFace — pure arithmetic per research
R1/R2)."""

import uuid
from datetime import datetime, timedelta

from app.services.curation import (
    HighlightCandidate,
    best_shot_score,
    compute_highlights,
    highlight_score,
    passes_quality_bar,
)

NOW = datetime(2026, 1, 1, 12, 0, 0)


def test_sharp_close_up_outscores_blurry_appearance():
    sharp = best_shot_score(quality_score=200.0, det_score=0.95, face_area_ratio=0.06, face_sharpness=200.0)
    blurry = best_shot_score(quality_score=20.0, det_score=0.6, face_area_ratio=0.02, face_sharpness=15.0)
    assert sharp > blurry


def test_best_shot_score_bounded_zero_to_one():
    score = best_shot_score(quality_score=1e6, det_score=1.0, face_area_ratio=1.0, face_sharpness=1e6)
    assert 0.0 <= score <= 1.0


def test_missing_signal_scores_as_zero_not_error():
    score = best_shot_score(quality_score=None, det_score=None, face_area_ratio=None, face_sharpness=None)
    assert score == 0.0


def test_passes_quality_bar_requires_ok_verdict_and_score_floor():
    assert passes_quality_bar("ok", 100.0, blur_min=55.0) is True
    assert passes_quality_bar("culled", 100.0, blur_min=55.0) is False
    assert passes_quality_bar("ok", 10.0, blur_min=55.0) is False


def test_group_shot_outranks_solo_shot_at_equal_quality():
    solo = highlight_score(quality_score=200.0, n_confident_faces=1)
    group = highlight_score(quality_score=200.0, n_confident_faces=3)
    assert group > solo


def test_log_damping_caps_a_crowd_from_beating_a_crisp_trio():
    trio = highlight_score(quality_score=200.0, n_confident_faces=3)
    huge_crowd = highlight_score(quality_score=200.0, n_confident_faces=20)
    # Log damping: going from 3 to 20 faces should not multiply the score by
    # anywhere near 20/3 — the group bonus is flat past the threshold.
    assert huge_crowd < trio * 2


def _candidate(quality_score, verdict, n_faces, offset_seconds=0) -> HighlightCandidate:
    return HighlightCandidate(
        photo_id=uuid.uuid4(),
        quality_score=quality_score,
        quality_verdict=verdict,
        n_confident_faces=n_faces,
        created_at=NOW + timedelta(seconds=offset_seconds),
    )


def test_compute_highlights_excludes_culled_photos():
    group_shot = _candidate(200.0, "ok", 3)
    culled = _candidate(5.0, "culled", 3)
    ranked = compute_highlights([group_shot, culled])
    assert culled.photo_id not in ranked
    assert ranked == [group_shot.photo_id]


def test_compute_highlights_caps_at_top_n():
    candidates = [_candidate(200.0 + i, "ok", 1, offset_seconds=i) for i in range(15)]
    ranked = compute_highlights(candidates, top_n=10)
    assert len(ranked) == 10


def test_compute_highlights_ties_broken_by_recency():
    older = _candidate(200.0, "ok", 1, offset_seconds=0)
    newer = _candidate(200.0, "ok", 1, offset_seconds=100)
    ranked = compute_highlights([older, newer])
    assert ranked[0] == newer.photo_id


if __name__ == "__main__":  # runnable self-check (Constitution V) alongside pytest
    test_sharp_close_up_outscores_blurry_appearance()
    test_compute_highlights_excludes_culled_photos()
    print("test_curation self-check OK")
