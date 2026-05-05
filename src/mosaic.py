"""Mosaic builder.

For each cell of the target grid:
  1. compute target cell mean LAB
  2. for each tile, compute color distance (LAB ΔE)
  3. apply content_type weight by region (face/body/background)
  4. apply usage penalty (avoid over-use of one tile)
  5. apply neighbor penalty (avoid same tile in adjacent cell)
  6. pick lowest-cost tile

Outputs:
  - output/{stem}_layout.json  (cells: [{x,y,tile}])
  - output/_score_render/{stem}.png  (1024x1024 rendered, scoring use)
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import (
    INDEX_CSV,
    META_JSON,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SCORE_RENDER_DIR,
    TILES_DIR,
    ensure_dirs,
    load_image_rgb,
    rel_to_project,
)

SCORE_RENDER_SIZE = 1024

REGION_WEIGHTS = {
    # text_poster/multi_person 페널티 강화 — 정체성 필터 후에도 안전 장치
    "face_region": {"face": 0, "half_body": 5, "full_body": 15, "scene": 25, "text_poster": 60, "multi_person": 50, "unknown": 12},
    "body_region": {"face": 10, "half_body": 0, "full_body": 5, "scene": 15, "text_poster": 50, "multi_person": 15, "unknown": 8},
    "background_region": {"face": 20, "half_body": 10, "full_body": 5, "scene": 0, "text_poster": 40, "multi_person": 20, "unknown": 8},
}


def _load_index(csv_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns (paths, full_lab[N,3], face_lab[N,3], patch_lab[N,P*P,3], P).

    P inferred from CSV columns (count of q*_l fields).
    """
    paths: list[str] = []
    full: list[list[float]] = []
    face: list[list[float]] = []
    patch: list[list[list[float]]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n_patches = sum(1 for h in reader.fieldnames if h.endswith("_l") and h.startswith("q"))
        side = int(round(n_patches**0.5))
        if side * side != n_patches or side < 1:
            raise SystemExit(f"patch column count {n_patches} not a perfect square")
        for row in reader:
            paths.append(row["path"])
            avl = [float(row["avg_l"]), float(row["avg_a"]), float(row["avg_b"])]
            full.append(avl)
            fl = float(row.get("face_l", -1) or -1)
            fa = float(row.get("face_a", -1) or -1)
            fb = float(row.get("face_b", -1) or -1)
            face.append(avl if fl < 0 else [fl, fa, fb])
            qs = []
            for i in range(n_patches):
                ql = float(row.get(f"q{i}_l", -1) or -1)
                qa = float(row.get(f"q{i}_a", -1) or -1)
                qb = float(row.get(f"q{i}_b", -1) or -1)
                qs.append(avl if ql < 0 else [ql, qa, qb])
            patch.append(qs)
    return (
        paths,
        np.array(full, dtype=np.float32),
        np.array(face, dtype=np.float32),
        np.array(patch, dtype=np.float32),
        side,
    )


def _load_meta(meta_path: Path) -> dict:
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"[mosaic] WARN: {meta_path} 없음. content_type 가중치 비활성")
    return {}


def _detect_target_face(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """OpenCV haarcascade로 타겟의 얼굴 영역 감지."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x), int(y), int(w), int(h))


def _classify_cell_region(
    cx: int, cy: int, cw: int, ch: int, face_bbox: tuple[int, int, int, int] | None, img_h: int
) -> str:
    if not face_bbox:
        return "unknown"
    fx, fy, fw, fh = face_bbox
    cell_cx = cx + cw / 2
    cell_cy = cy + ch / 2
    if fx <= cell_cx <= fx + fw and fy <= cell_cy <= fy + fh:
        return "face_region"
    body_top = fy + fh
    body_bottom = min(img_h, fy + fh + int(fh * 3))
    if body_top <= cell_cy <= body_bottom:
        return "body_region"
    return "background_region"


def _build_layout(
    target_rgb: np.ndarray,
    paths: list[str],
    tile_full_labs: np.ndarray,
    tile_face_labs: np.ndarray,
    tile_patch_labs: np.ndarray,
    patch_side: int,
    meta: dict,
    grid: int,
    neighbor_penalty: float,
    usage_penalty: float,
    use_region: bool,
    color_top_k: int = 50,
    ab_weight: float = 3.0,
) -> list[dict]:
    H, W = target_rgb.shape[:2]
    cell_h = H // grid
    cell_w = W // grid

    face_bbox = _detect_target_face(target_rgb) if use_region else None
    if face_bbox:
        print(f"[mosaic] target face bbox: {face_bbox}")

    # Compute LAB per cell
    bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
    lab_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    layout = [[None] * grid for _ in range(grid)]
    usage = np.zeros(len(paths), dtype=np.float32)

    for r in tqdm(range(grid), desc="rows"):
        for c in range(grid):
            y0 = r * cell_h
            x0 = c * cell_w
            cell = lab_full[y0 : y0 + cell_h, x0 : x0 + cell_w]
            target_lab = cell.reshape(-1, 3).mean(axis=0)

            # PxP sub-cell LABs (row-major: 0=TL, P-1=TR, P*P-1=BR)
            sub_labs_list = []
            for pi in range(patch_side):
                ry0 = pi * cell_h // patch_side
                ry1 = (pi + 1) * cell_h // patch_side
                for pj in range(patch_side):
                    rx0 = pj * cell_w // patch_side
                    rx1 = (pj + 1) * cell_w // patch_side
                    patch = cell[ry0:ry1, rx0:rx1]
                    sub_labs_list.append(patch.reshape(-1, 3).mean(axis=0))
            sub_labs = np.stack(sub_labs_list)  # (P*P, 3)

            region = "unknown"
            if use_region and face_bbox:
                region = _classify_cell_region(x0, y0, cell_w, cell_h, face_bbox, H)

            # === STEP 1: COLOR PRE-FILTER (chroma-weighted) ===
            # a/b (chroma) 채널을 ab_weight 배 가중 → 검정 셀(중립 a≈b≈128)에서
            # 푸른 타일(b<120) 같은 색감 미스매치를 강력히 페널티
            ab_w = np.array([1.0, ab_weight, ab_weight], dtype=np.float32)
            diff_full = (tile_full_labs - target_lab) * ab_w
            color_dist = np.sqrt((diff_full * diff_full).sum(axis=1))
            k = min(color_top_k, len(color_dist))
            top_idx = np.argpartition(color_dist, k - 1)[:k]

            # === STEP 2: PATCH MATCHING (chroma-weighted) on subset ===
            patch_subset = tile_patch_labs[top_idx]  # (k, P*P, 3)
            diff_q = (patch_subset - sub_labs[None, :, :]) * ab_w[None, None, :]
            per_q_dist = np.sqrt((diff_q * diff_q).sum(axis=2))  # (k, P*P)
            cost = per_q_dist.mean(axis=1)  # (k,)

            # 색 정확도 우선 — face_lab/region_weights 비활성화 (사용자: 모든 색
            # 영역에서 가장 매칭되는 타일을 그대로 넣고 싶음)
            # usage penalty만 약하게 유지 (한 타일 도배 방지)
            cost += usage[top_idx] * usage_penalty

            # neighbor penalty — only if neighbor's tile is in current subset
            for dr, dc in ((-1, 0), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid and 0 <= nc < grid and layout[nr][nc] is not None:
                    nbr_full_i = layout[nr][nc]
                    pos = np.where(top_idx == nbr_full_i)[0]
                    if len(pos):
                        cost[pos[0]] += neighbor_penalty * 50.0

            best_sub = int(cost.argmin())
            best = int(top_idx[best_sub])
            layout[r][c] = best
            usage[best] += 1.0

    cells = []
    for r in range(grid):
        for c in range(grid):
            cells.append({"x": c, "y": r, "tile": paths[layout[r][c]]})
    return cells


def _render_score_image(
    cells: list[dict], grid: int, blend: float, target_rgb: np.ndarray, size: int = SCORE_RENDER_SIZE
) -> np.ndarray:
    """Render a {size}x{size} mosaic image for scoring."""
    cell_px = size // grid
    canvas = np.zeros((cell_px * grid, cell_px * grid, 3), dtype=np.uint8)

    target_resized = cv2.resize(target_rgb, (cell_px * grid, cell_px * grid), interpolation=cv2.INTER_AREA)

    cache: dict[str, np.ndarray] = {}
    for cell in cells:
        x, y, tile_rel = cell["x"], cell["y"], cell["tile"]
        if tile_rel not in cache:
            tile_path = PROJECT_ROOT / tile_rel
            img = load_image_rgb(tile_path)
            if img is None:
                cache[tile_rel] = np.full((cell_px, cell_px, 3), 128, dtype=np.uint8)
            else:
                cache[tile_rel] = cv2.resize(img, (cell_px, cell_px), interpolation=cv2.INTER_AREA)
        tile_img = cache[tile_rel]
        y0 = y * cell_px
        x0 = x * cell_px
        target_patch = target_resized[y0 : y0 + cell_px, x0 : x0 + cell_px]
        if blend > 0:
            blended = (target_patch * blend + tile_img * (1 - blend)).clip(0, 255).astype(np.uint8)
        else:
            blended = tile_img
        canvas[y0 : y0 + cell_px, x0 : x0 + cell_px] = blended
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Mosaic builder")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--blend", type=float, default=0.75)
    parser.add_argument("--neighbor-penalty", type=float, default=2.0)
    parser.add_argument("--usage-penalty", type=float, default=5.0)
    parser.add_argument("--no-region", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--index-csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--meta-json", type=Path, default=META_JSON)
    parser.add_argument("--score-render-size", type=int, default=SCORE_RENDER_SIZE)
    parser.add_argument(
        "--color-top-k", type=int, default=50,
        help="셀 매칭 시 색거리 기준 사전 필터 후보 수 (작을수록 색 정확↑, 다양성↓)"
    )
    parser.add_argument(
        "--ab-weight", type=float, default=3.0,
        help="LAB의 a/b 채널 가중치 (큰 값일수록 색상 차이에 더 민감)"
    )
    args = parser.parse_args()

    ensure_dirs()

    if not args.index_csv.exists():
        raise SystemExit(f"인덱스 없음: {args.index_csv}. /classify-tiles 또는 /build-index 먼저 실행")
    if not args.target.exists():
        raise SystemExit(f"타겟 없음: {args.target}")

    paths, tile_full_labs, tile_face_labs, tile_patch_labs, patch_side = _load_index(args.index_csv)
    if len(paths) < 100:
        print(f"[mosaic] WARN: 인덱스 {len(paths)}장. 100장 미만은 모자이크 품질 낮음")
    print(f"[mosaic] patch grid: {patch_side}x{patch_side} = {patch_side*patch_side} sub-LABs/tile")
    meta = _load_meta(args.meta_json)

    target_rgb = load_image_rgb(args.target)
    if target_rgb is None:
        raise SystemExit(f"타겟 로드 실패: {args.target}")
    # square crop to make grid clean
    H, W = target_rgb.shape[:2]
    side = min(H, W)
    target_rgb = target_rgb[(H - side) // 2 : (H + side) // 2, (W - side) // 2 : (W + side) // 2]

    print(f"[mosaic] target {args.target.name} | grid {args.grid} | tiles {len(paths)}")
    cells = _build_layout(
        target_rgb,
        paths,
        tile_full_labs,
        tile_face_labs,
        tile_patch_labs,
        patch_side,
        meta,
        args.grid,
        args.neighbor_penalty,
        args.usage_penalty,
        use_region=not args.no_region,
        color_top_k=args.color_top_k,
        ab_weight=args.ab_weight,
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"mosaic_{args.target.stem}_{ts}_{args.grid}x{args.grid}"
    layout_path = args.out_dir / f"{stem}_layout.json"
    layout_data = {
        "target": rel_to_project(args.target),
        "grid": {"cols": args.grid, "rows": args.grid},
        "params": {
            "blend": args.blend,
            "neighbor_penalty": args.neighbor_penalty,
            "usage_penalty": args.usage_penalty,
            "use_region": not args.no_region,
        },
        "cells": cells,
    }
    layout_path.write_text(json.dumps(layout_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mosaic] layout → {layout_path}")

    # Score render: scale with grid so each cell has ≥16px (preserves SSIM measurement quality)
    render_size = max(args.score_render_size, args.grid * 16)
    # snap to multiple of grid to avoid resampling artifacts
    render_size = (render_size // args.grid) * args.grid
    score_img = _render_score_image(cells, args.grid, args.blend, target_rgb, size=render_size)
    score_path = SCORE_RENDER_DIR / f"{stem}.png"
    Image.fromarray(score_img).save(score_path)
    print(f"[mosaic] score render → {score_path}")
    print(f"[mosaic] 다음: /score-mosaic {layout_path} {args.target}")


if __name__ == "__main__":
    main()
