"""Mosaic builder with CLIP + LAB patch hybrid matching.

For each grid cell:
  1. Compute LAB patch cost (mean per-quadrant ΔE) — same as mosaic.py
  2. Compute CLIP cost (1 - cosine similarity between cell embedding and tile embeddings)
  3. Combined cost = (1 - α) * normalized_lab + α * clip_dist

CLIP captures semantic/perceptual similarity — helps with SSIM and perceptual scores
that pure color matching can't reach.

Outputs: same as mosaic.py (layout JSON + score render PNG).
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

from mosaic import (
    REGION_WEIGHTS,
    SCORE_RENDER_SIZE,
    _classify_cell_region,
    _detect_target_face,
    _load_index,
    _load_meta,
    _render_score_image,
)
from utils import (
    INDEX_CSV,
    META_JSON,
    OUTPUT_DIR,
    SCORE_RENDER_DIR,
    ensure_dirs,
    load_image_rgb,
    rel_to_project,
)

CLIP_NPY = INDEX_CSV.parent / "tile_clip.npy"


def _clip_model():
    import open_clip
    import torch

    device = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    return model, preprocess, device, torch


def _embed_cells(target_rgb: np.ndarray, grid: int, model, preprocess, device, torch, batch=64):
    """Compute CLIP embedding per cell. Each cell is upsampled to 224x224 (CLIP input size)
    via bilinear interpolation since cells are small.
    Returns (grid*grid, 512) L2-normalized.
    """
    H, W = target_rgb.shape[:2]
    cell_h = H // grid
    cell_w = W // grid

    embs = np.zeros((grid * grid, 512), dtype=np.float32)
    batch_imgs = []
    batch_idx = []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        emb = f.cpu().numpy().astype(np.float32)
        for k, idx in enumerate(batch_idx):
            embs[idx] = emb[k]
        batch_imgs.clear()
        batch_idx.clear()

    for r in tqdm(range(grid), desc="cell-clip-rows"):
        for c in range(grid):
            y0 = r * cell_h
            x0 = c * cell_w
            cell = target_rgb[y0 : y0 + cell_h, x0 : x0 + cell_w]
            # upsample to 224x224 for CLIP
            img = Image.fromarray(cell).resize((224, 224), Image.BILINEAR)
            t = preprocess(img)
            batch_imgs.append(t)
            batch_idx.append(r * grid + c)
            if len(batch_imgs) >= batch:
                flush()
    flush()
    return embs


def _build_layout_clip(
    target_rgb: np.ndarray,
    paths: list[str],
    tile_full_labs: np.ndarray,
    tile_face_labs: np.ndarray,
    tile_patch_labs: np.ndarray,
    patch_side: int,
    tile_clip: np.ndarray,
    cell_clip: np.ndarray,
    meta: dict,
    grid: int,
    neighbor_penalty: float,
    usage_penalty: float,
    use_region: bool,
    clip_alpha: float,
) -> list[dict]:
    H, W = target_rgb.shape[:2]
    cell_h = H // grid
    cell_w = W // grid

    face_bbox = _detect_target_face(target_rgb) if use_region else None
    if face_bbox:
        print(f"[clip-mosaic] target face bbox: {face_bbox}")

    bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
    lab_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    layout = [[None] * grid for _ in range(grid)]
    usage = np.zeros(len(paths), dtype=np.float32)

    # Precompute all cell sub-LABs
    cell_sub_labs = np.zeros((grid, grid, patch_side * patch_side, 3), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            y0 = r * cell_h
            x0 = c * cell_w
            cell = lab_full[y0 : y0 + cell_h, x0 : x0 + cell_w]
            for pi in range(patch_side):
                ry0 = pi * cell_h // patch_side
                ry1 = (pi + 1) * cell_h // patch_side
                for pj in range(patch_side):
                    rx0 = pj * cell_w // patch_side
                    rx1 = (pj + 1) * cell_w // patch_side
                    patch = cell[ry0:ry1, rx0:rx1]
                    cell_sub_labs[r, c, pi * patch_side + pj] = patch.reshape(-1, 3).mean(axis=0)

    # Reference scale: typical LAB ΔE range is 0~100. CLIP cos-dist 0~2.
    # Normalize LAB cost by /50 → roughly 0~2 range. Then weighted blend.

    for r in tqdm(range(grid), desc="rows"):
        for c in range(grid):
            sub_labs = cell_sub_labs[r, c]
            diff_q = tile_patch_labs - sub_labs[None, :, :]
            per_q_dist = np.sqrt((diff_q * diff_q).sum(axis=2))
            lab_cost = per_q_dist.mean(axis=1) / 50.0  # normalize

            # CLIP cosine distance
            cell_emb = cell_clip[r * grid + c]
            cos_sim = tile_clip @ cell_emb  # (N,)
            clip_cost = 1.0 - cos_sim

            cost = (1.0 - clip_alpha) * lab_cost + clip_alpha * clip_cost

            region = "unknown"
            if use_region and face_bbox:
                y0 = r * cell_h
                x0 = c * cell_w
                region = _classify_cell_region(x0, y0, cell_w, cell_h, face_bbox, H)

            # face_region: also blend face-skin LAB (in normalized units)
            if region == "face_region":
                face_diff = tile_face_labs - sub_labs.mean(axis=0)
                face_dist = np.sqrt((face_diff * face_diff).sum(axis=1)) / 50.0
                cost = 0.5 * cost + 0.5 * face_dist

            # region weight (in normalized units; original was in ΔE so divide /50)
            if use_region and face_bbox and meta:
                weights_for_region = REGION_WEIGHTS.get(region, {})
                for i, p in enumerate(paths):
                    ct = meta.get(p, {}).get("content_type", "unknown")
                    cost[i] += weights_for_region.get(ct, 10) / 50.0

            cost += usage * (usage_penalty / 50.0)

            for dr, dc in ((-1, 0), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid and 0 <= nc < grid and layout[nr][nc] is not None:
                    cost[layout[nr][nc]] += neighbor_penalty * 50.0 / 50.0

            best = int(cost.argmin())
            layout[r][c] = best
            usage[best] += 1.0

    cells = []
    for r in range(grid):
        for c in range(grid):
            cells.append({"x": c, "y": r, "tile": paths[layout[r][c]]})
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Mosaic builder with CLIP + LAB hybrid matching")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=60)
    parser.add_argument("--blend", type=float, default=0.4)
    parser.add_argument("--neighbor-penalty", type=float, default=2.0)
    parser.add_argument("--usage-penalty", type=float, default=3.0)
    parser.add_argument("--no-region", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--index-csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--meta-json", type=Path, default=META_JSON)
    parser.add_argument("--clip-npy", type=Path, default=CLIP_NPY)
    parser.add_argument(
        "--clip-alpha", type=float, default=0.4,
        help="CLIP weight in cost blend (0=LAB only, 1=CLIP only)"
    )
    args = parser.parse_args()

    ensure_dirs()

    if not args.index_csv.exists():
        raise SystemExit(f"인덱스 없음: {args.index_csv}")
    if not args.clip_npy.exists():
        raise SystemExit(f"CLIP npy 없음: {args.clip_npy}. /clip-index 먼저 실행")
    if not args.target.exists():
        raise SystemExit(f"타겟 없음: {args.target}")

    paths, tile_full_labs, tile_face_labs, tile_patch_labs, patch_side = _load_index(args.index_csv)
    print(f"[clip-mosaic] {len(paths)} tiles | patch {patch_side}x{patch_side} | clip α={args.clip_alpha}")
    meta = _load_meta(args.meta_json)
    tile_clip = np.load(args.clip_npy).astype(np.float32)
    if tile_clip.shape[0] != len(paths):
        raise SystemExit(
            f"CLIP shape {tile_clip.shape} != paths {len(paths)}. /clip-index 다시 실행"
        )

    target_rgb = load_image_rgb(args.target)
    if target_rgb is None:
        raise SystemExit(f"타겟 로드 실패: {args.target}")
    H, W = target_rgb.shape[:2]
    side = min(H, W)
    target_rgb = target_rgb[(H - side) // 2 : (H + side) // 2, (W - side) // 2 : (W + side) // 2]

    print("[clip-mosaic] computing cell CLIP embeddings…")
    model, preprocess, device, torch = _clip_model()
    cell_clip = _embed_cells(target_rgb, args.grid, model, preprocess, device, torch)

    cells = _build_layout_clip(
        target_rgb,
        paths,
        tile_full_labs,
        tile_face_labs,
        tile_patch_labs,
        patch_side,
        tile_clip,
        cell_clip,
        meta,
        args.grid,
        args.neighbor_penalty,
        args.usage_penalty,
        use_region=not args.no_region,
        clip_alpha=args.clip_alpha,
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"mosaic_{args.target.stem}_{ts}_{args.grid}x{args.grid}_clip"
    layout_path = args.out_dir / f"{stem}_layout.json"
    layout_data = {
        "target": rel_to_project(args.target),
        "grid": {"cols": args.grid, "rows": args.grid},
        "params": {
            "blend": args.blend,
            "neighbor_penalty": args.neighbor_penalty,
            "usage_penalty": args.usage_penalty,
            "use_region": not args.no_region,
            "clip_alpha": args.clip_alpha,
        },
        "cells": cells,
    }
    layout_path.write_text(json.dumps(layout_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[clip-mosaic] layout → {layout_path}")

    render_size = max(SCORE_RENDER_SIZE, args.grid * 16)
    render_size = (render_size // args.grid) * args.grid
    score_img = _render_score_image(cells, args.grid, args.blend, target_rgb, size=render_size)
    score_path = SCORE_RENDER_DIR / f"{stem}.png"
    Image.fromarray(score_img).save(score_path)
    print(f"[clip-mosaic] score render → {score_path}")
    print(f"[clip-mosaic] 다음: /score-mosaic {layout_path} {args.target}")


if __name__ == "__main__":
    main()
