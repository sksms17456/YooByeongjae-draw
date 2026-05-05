"""Curation - filter low quality tiles into _rejected/ (no deletion)."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2

from utils import (
    REJECTED_DIR,
    TILES_DIR,
    ensure_dirs,
    iter_tile_paths,
    laplacian_sharpness,
    load_image_rgb,
)


def _is_too_small(rgb, min_side: int) -> bool:
    h, w = rgb.shape[:2]
    return min(h, w) < min_side


def _is_too_blurry(rgb, threshold: float) -> bool:
    return laplacian_sharpness(rgb) < threshold


def _is_text_heavy(rgb, edge_threshold: float = 0.18) -> bool:
    """에지 밀도가 비정상적으로 높으면 텍스트/포스터로 추정."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    density = edges.mean() / 255.0
    return density > edge_threshold


def _move_to_rejected(path: Path, reason: str) -> None:
    sub = REJECTED_DIR / reason
    sub.mkdir(parents=True, exist_ok=True)
    target = sub / path.name
    if target.exists():
        target = sub / f"{path.stem}_{path.parent.name}{path.suffix}"
    shutil.move(str(path), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="타일 품질 필터")
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--blur-threshold", type=float, default=80.0, help="라플라시안 분산 임계")
    parser.add_argument("--no-text-filter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    rejects: Counter[str] = Counter()
    keep = 0
    for p in iter_tile_paths():
        rgb = load_image_rgb(p)
        if rgb is None:
            (rejects.update(["unreadable"]),)
            if not args.dry_run:
                _move_to_rejected(p, "unreadable")
            continue

        if _is_too_small(rgb, args.min_side):
            rejects.update(["too_small"])
            if not args.dry_run:
                _move_to_rejected(p, "too_small")
            continue

        if _is_too_blurry(rgb, args.blur_threshold):
            rejects.update(["blurry"])
            if not args.dry_run:
                _move_to_rejected(p, "blurry")
            continue

        if not args.no_text_filter and _is_text_heavy(rgb):
            rejects.update(["text_heavy"])
            if not args.dry_run:
                _move_to_rejected(p, "text_heavy")
            continue

        keep += 1

    print(f"[curate] 유지 {keep}장")
    for reason, n in rejects.most_common():
        print(f"  rejected[{reason}]: {n}장")
    if args.dry_run:
        print("[curate] dry-run 모드 — 실제 이동 없음")


if __name__ == "__main__":
    main()
