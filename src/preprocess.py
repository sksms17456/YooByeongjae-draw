"""Preprocess - normalize tiles to 256, generate 50px thumbnails, build numeric index CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from utils import (
    INDEX_CSV,
    ORIGINAL_SIZE,
    THUMB_SIZE,
    THUMBS_DIR,
    TILES_DIR,
    avg_lab,
    dominant_hue,
    ensure_dirs,
    iter_tile_paths,
    laplacian_sharpness,
    load_image_rgb,
    rel_to_project,
    resize_pad,
    save_jpg,
)

PATCH_N = 2  # 2x2 patches gave +6 avg score; 3x3 was noisier on small cells

CSV_FIELDS = [
    "path",
    "avg_l", "avg_a", "avg_b",
    "face_l", "face_a", "face_b",
    *[f"q{i}_{c}" for i in range(PATCH_N * PATCH_N) for c in ("l", "a", "b")],
    "dominant_hue",
    "brightness",
    "saturation",
    "sharpness",
    "has_face",
    "face_x", "face_y", "face_w", "face_h",
]


def _thumb_path_for(tile_path: Path) -> Path:
    rel = tile_path.relative_to(TILES_DIR)
    return THUMBS_DIR / rel


def process_tile(path: Path, do_thumb: bool = True) -> dict | None:
    rgb = load_image_rgb(path)
    if rgb is None:
        return None

    # Normalize source to 256x256 (overwrite original — saves disk, browser-friendly)
    if rgb.shape[0] != ORIGINAL_SIZE or rgb.shape[1] != ORIGINAL_SIZE:
        rgb = resize_pad(rgb, ORIGINAL_SIZE)
        save_jpg(rgb, path, quality=92)

    if do_thumb:
        thumb = resize_pad(rgb, THUMB_SIZE)
        save_jpg(thumb, _thumb_path_for(path), quality=85)

    L, A, B = avg_lab(rgb)
    sharpness = laplacian_sharpness(rgb)
    hue = dominant_hue(rgb)

    # NxN patch LAB
    H, W = rgb.shape[:2]
    q_labs = []
    for i in range(PATCH_N):
        y0 = i * H // PATCH_N
        y1 = (i + 1) * H // PATCH_N
        for j in range(PATCH_N):
            x0 = j * W // PATCH_N
            x1 = (j + 1) * W // PATCH_N
            q_labs.append(avg_lab(rgb[y0:y1, x0:x1]))

    # HSV brightness/saturation
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    brightness = float(hsv[..., 2].mean()) / 255.0
    saturation = float(hsv[..., 1].mean()) / 255.0

    out = {
        "path": rel_to_project(path),
        "avg_l": round(L, 3),
        "avg_a": round(A, 3),
        "avg_b": round(B, 3),
        "face_l": -1.0,  # filled by reindex_face_lab.py
        "face_a": -1.0,
        "face_b": -1.0,
        "dominant_hue": hue,
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "sharpness": round(sharpness, 2),
        "has_face": 0,  # filled by classify.py
        "face_x": -1,
        "face_y": -1,
        "face_w": -1,
        "face_h": -1,
    }
    for i, (ql, qa, qb) in enumerate(q_labs):
        out[f"q{i}_l"] = round(ql, 3)
        out[f"q{i}_a"] = round(qa, 3)
        out[f"q{i}_b"] = round(qb, 3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess tiles + build numeric index")
    parser.add_argument("--tiles", type=Path, default=TILES_DIR)
    parser.add_argument("--out-csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--no-thumbs", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    paths = iter_tile_paths(args.tiles)
    print(f"[preprocess] {len(paths)}장 처리 시작")

    rows = []
    for p in tqdm(paths):
        row = process_tile(p, do_thumb=not args.no_thumbs)
        if row:
            rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[preprocess] {len(rows)}장 인덱싱 완료 → {args.out_csv}")
    if not args.no_thumbs:
        print(f"[preprocess] 썸네일 생성: {THUMBS_DIR}")


if __name__ == "__main__":
    main()
