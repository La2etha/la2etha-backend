"""CV evaluation harness scaffold + ground-truth loader.

Ground truth is a "who is in which photo" map — the basis for per-user gallery
recall/precision (SC-002/003), clustering quality, and enrollment ablations that
land in Phase 5 (eval/metrics_*.py). This module only loads and shapes the
ground truth; the metric computations are added later.

Ground-truth file (JSON):
    {
      "photos": { "<photo_filename>": ["alice", "bob"], ... },
      "people": ["alice", "bob", "carol"]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GroundTruth:
    # photo filename -> set of person labels present in it
    photo_people: dict[str, set[str]] = field(default_factory=dict)
    people: set[str] = field(default_factory=set)

    def photos_for(self, person: str) -> set[str]:
        """The set of photos a given person truly appears in."""
        return {photo for photo, people in self.photo_people.items() if person in people}


def load_ground_truth(path: str | Path) -> GroundTruth:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    photo_people = {photo: set(people) for photo, people in raw.get("photos", {}).items()}
    people = set(raw.get("people", [])) or {p for ps in photo_people.values() for p in ps}
    return GroundTruth(photo_people=photo_people, people=people)


def recall_precision(predicted: set[str], truth: set[str]) -> tuple[float, float]:
    """Recall/precision of a predicted photo set against ground truth."""
    if not truth:
        precision = 1.0 if not predicted else 0.0
        return 1.0, precision
    true_positives = len(predicted & truth)
    recall = true_positives / len(truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    return recall, precision


if __name__ == "__main__":  # simple runnable self-check (Constitution V)
    gt = GroundTruth(
        photo_people={"a.jpg": {"alice"}, "b.jpg": {"alice", "bob"}},
        people={"alice", "bob"},
    )
    assert gt.photos_for("alice") == {"a.jpg", "b.jpg"}
    assert gt.photos_for("bob") == {"b.jpg"}
    assert recall_precision({"a.jpg"}, {"a.jpg", "b.jpg"}) == (0.5, 1.0)
    print("eval harness self-check OK")
