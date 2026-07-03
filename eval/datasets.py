"""Loaders for public benchmark datasets (component sanity checks).

Recognition verification: LFW, CFP-FP, AgeDB-30 (pair lists → same/different).
Detection: WIDER FACE (val bbox ground truth → AP).

These loaders READ already-downloaded data; they do not download anything and
they do not ship any images. Point them at the official releases you obtained
yourself. Missing files raise a clear FileNotFoundError with the expected path.

Deliberately avoids retracted / withdrawn sets (MS-Celeb-1M, MegaFace,
DukeMTMC) — see research-cv-ai.md ("avoid retracted sets"); use the sets above,
which remain the standard, citable face benchmarks.

Run: ``python -m eval.datasets`` (self-check on inline sample formats).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VerificationPair:
    path_a: str
    path_b: str
    same: bool  # True = genuine (same identity), False = impostor


def load_verification_pairs(path: str | Path) -> list[VerificationPair]:
    """Generic pair list: one pair per line as ``path_a  path_b  label``.

    ``label`` is 1/0 (or true/false, same/diff). Blank lines and ``#`` comments
    are skipped. CFP-FP and AgeDB-30 pair lists convert cleanly to this format.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Verification pair list not found: {p}")
    truthy = {"1", "same", "true", "genuine", "pos"}
    pairs: list[VerificationPair] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        a, b, label = parts[0], parts[1], parts[2].lower()
        pairs.append(VerificationPair(a, b, same=label in truthy))
    return pairs


def load_lfw_official_pairs(
    pairs_txt: str | Path, image_root: str | Path, ext: str = "jpg"
) -> list[VerificationPair]:
    """Parse the canonical LFW ``pairs.txt``.

    Matched line:  ``Name  n1  n2``  → Name/Name_000n1.jpg vs Name/Name_000n2.jpg
    Mismatched:    ``Name1 n1 Name2 n2``.
    The header line (set/pair counts) is ignored.
    """
    p = Path(pairs_txt)
    if not p.exists():
        raise FileNotFoundError(f"LFW pairs.txt not found: {p}")
    root = Path(image_root)

    def img(name: str, idx: str) -> str:
        return str(root / name / f"{name}_{int(idx):04d}.{ext}")

    pairs: list[VerificationPair] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            pairs.append(VerificationPair(img(parts[0], parts[1]), img(parts[0], parts[2]), True))
        elif len(parts) == 4:
            pairs.append(VerificationPair(img(parts[0], parts[1]), img(parts[2], parts[3]), False))
    return pairs


def load_wider_face(gt_file: str | Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Parse WIDER FACE ``wider_face_*_bbx_gt.txt`` → {image_path: [(x,y,w,h), ...]}.

    Format per image: a path line, a face-count line, then that many bbox lines
    (``x y w h`` plus attribute columns we ignore). A count of 0 is followed by a
    single placeholder line, which we skip.
    """
    p = Path(gt_file)
    if not p.exists():
        raise FileNotFoundError(f"WIDER FACE ground-truth file not found: {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    i = 0
    while i < len(lines):
        image = lines[i].strip()
        i += 1
        if not image or i >= len(lines):
            break
        count = int(lines[i].strip() or "0")
        i += 1
        boxes: list[tuple[int, int, int, int]] = []
        rows = max(count, 1)  # a 0-face image still has one placeholder row
        for _ in range(rows):
            if i >= len(lines):
                break
            vals = lines[i].split()
            i += 1
            if count and len(vals) >= 4:
                boxes.append((int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])))
        out[image] = boxes
    return out


if __name__ == "__main__":  # runnable self-check (Constitution V)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        pairs_file = Path(d) / "pairs.tsv"
        pairs_file.write_text(
            "# a b label\n"
            "img/a1.jpg img/a2.jpg 1\n"
            "img/a1.jpg,img/b1.jpg,0\n"
            "\n",
            encoding="utf-8",
        )
        pairs = load_verification_pairs(pairs_file)
        assert len(pairs) == 2
        assert pairs[0].same is True and pairs[1].same is False

        wider = Path(d) / "gt.txt"
        wider.write_text(
            "0--Parade/img_1.jpg\n2\n10 20 30 40 0 0 0 0 0 0\n50 60 15 15 0 0 0 0 0 0\n"
            "1--Handshake/img_2.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n",
            encoding="utf-8",
        )
        boxes = load_wider_face(wider)
        assert boxes["0--Parade/img_1.jpg"] == [(10, 20, 30, 40), (50, 60, 15, 15)]
        assert boxes["1--Handshake/img_2.jpg"] == []

        try:
            load_verification_pairs(Path(d) / "nope.txt")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass

    print("datasets loader self-check OK")
