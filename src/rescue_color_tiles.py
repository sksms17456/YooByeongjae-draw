"""Rescue tiles from _rejected/tiny_byunjae and byunjae_not_main with looser thresholds.

Use case: vivid color tiles (stage scenes, lighting shots) often have 유병재
small in frame but are valuable for color diversity. Default identity filter
rejects them. This script re-applies a looser filter on those rejection bins
and moves passing tiles back to tiles/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (
    REJECTED_DIR,
    TILES_DIR,
    ensure_dirs,
    load_image_rgb,
)
from identity_filter import _embeddings, REF_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescue color tiles from _rejected with looser thresholds")
    parser.add_argument("--sim-threshold", type=float, default=0.30,
                        help="reference 유사도 임계 (기본 0.30, 원본 필터 0.35)")
    parser.add_argument("--min-face-ratio", type=float, default=0.005,
                        help="얼굴 면적 비율 임계 (기본 0.005 = 매우 작은 얼굴도 OK)")
    parser.add_argument("--bins", nargs="+", default=["tiny_byunjae", "byunjae_not_main"],
                        help="처리할 _rejected 서브폴더")
    parser.add_argument("--target-source", default="misc",
                        help="구제된 타일 이동할 소스 폴더 (tiles/<source>/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if not REF_PATH.exists():
        raise SystemExit(f"reference 없음: {REF_PATH}")
    ref = np.load(REF_PATH).astype(np.float32)

    target_dir = TILES_DIR / args.target_source
    target_dir.mkdir(parents=True, exist_ok=True)

    rescued = 0
    rejected = 0
    for bin_name in args.bins:
        bin_dir = REJECTED_DIR / bin_name
        if not bin_dir.exists():
            continue
        files = [p for p in bin_dir.iterdir() if p.is_file()]
        print(f"[rescue] {bin_name}: {len(files)} candidates")

        for p in tqdm(files, desc=bin_name):
            rgb = load_image_rgb(p)
            if rgb is None:
                continue
            H, W = rgb.shape[:2]
            faces = _embeddings(rgb)
            if not faces:
                rejected += 1
                continue
            sims = [(area, ref @ emb) for area, emb, _ in faces]
            best = max(sims, key=lambda t: t[1])
            area, sim = best
            ratio = area / (H * W)
            if sim < args.sim_threshold or ratio < args.min_face_ratio:
                rejected += 1
                continue
            # passes — move to target_dir
            dest = target_dir / f"rescued_{bin_name}_{p.name}"
            if dest.exists():
                continue
            if not args.dry_run:
                shutil.move(str(p), str(dest))
            rescued += 1

    print(f"\n[rescue] 구제: {rescued}장 → {target_dir}")
    print(f"[rescue] 여전히 거절: {rejected}장 (그대로 유지)")
    if args.dry_run:
        print("[rescue] dry-run 모드 — 실제 이동 없음")


if __name__ == "__main__":
    main()
