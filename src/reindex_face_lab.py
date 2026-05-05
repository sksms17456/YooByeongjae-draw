"""Fill face_l/face_a/face_b columns from a fresh Haar detection on each
256×256 tile. avg_l/a/b (full-image LAB) is left untouched.

Why: mosaic.py uses face_*  for face-region cells (skin/painted face matching)
and falls back to avg_* for background/body cells (whole-image color).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from utils import (
    INDEX_CSV,
    PROJECT_ROOT,
    avg_lab,
    load_image_rgb,
)


_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(path)
    return _CASCADE


def _detect_face_256(rgb):
    """Haar detection on 256x256 tile, return largest face bbox or None."""
    cascade = _cascade()
    if cascade.empty():
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x), int(y), int(w), int(h))


def _crop_face(rgb, x: int, y: int, w: int, h: int):
    H, W = rgb.shape[:2]
    pad = int(0.10 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return rgb[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser(description="얼굴 영역 LAB 재인덱싱 (Haar 신규 감지)")
    parser.add_argument("--csv", type=Path, default=INDEX_CSV)
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV 없음: {args.csv}")

    rows = []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    n_face_lab = 0
    n_full = 0
    n_skip_unread = 0

    for row in tqdm(rows, desc="reindex"):
        rel = row["path"]
        rgb = load_image_rgb(PROJECT_ROOT / rel)
        if rgb is None:
            n_skip_unread += 1
            continue

        bbox = _detect_face_256(rgb)
        if bbox is None:
            n_full += 1
            continue
        x, y, w, h = bbox

        face = _crop_face(rgb, x, y, w, h)
        if face is None or face.size == 0:
            n_full += 1
            continue

        L, A, B = avg_lab(face)
        row["face_l"] = round(L, 3)
        row["face_a"] = round(A, 3)
        row["face_b"] = round(B, 3)
        row["face_x"] = x
        row["face_y"] = y
        row["face_w"] = w
        row["face_h"] = h
        row["has_face"] = 1
        n_face_lab += 1

    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[reindex] 얼굴 LAB 적용: {n_face_lab}장")
    print(f"[reindex] 전체 LAB 유지(얼굴 없음): {n_full}장")
    print(f"[reindex] 스킵(읽기 실패): {n_skip_unread}장")
    print(f"[reindex] CSV 갱신 → {args.csv}")


if __name__ == "__main__":
    main()
