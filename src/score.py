"""Rubric scorer for mosaic.

7-axis rubric (weights sum to 100):
  - SSIM (30): structural similarity
  - LPIPS or CLIP cosine (25): perceptual similarity
  - LAB ΔE (15): color fidelity (cell-level)
  - Diversity (10): unique tiles / total cells
  - Adjacency (10): 4-neighbor same-tile ratio (lower better)
  - Sharpness (5): used tiles' Laplacian variance mean
  - Identifiability (5): fraction of tiles at 256x256 disk resolution

Outputs JSON next to layout file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from utils import INDEX_CSV, OUTPUT_DIR, PROJECT_ROOT, SCORE_RENDER_DIR, load_image_rgb

WEIGHTS = {
    "ssim": 30,
    "perceptual": 25,
    "color": 15,
    "diversity": 10,
    "adjacency": 10,
    "sharpness": 5,
    "identifiability": 5,
}

CALIBRATION = {
    # (good = 1.0, bad = 0.0)
    # 모자이크 현실에 맞춰 보정 (SSIM/LPIPS는 모자이크 본질적 한계 반영)
    "ssim": (0.45, 0.10),  # 모자이크는 일반 사진 SSIM(0.7+)과 다른 스케일
    "perceptual_lpips": (0.30, 0.65),  # lower better
    "perceptual_clip": (0.78, 0.35),  # higher better
    "color": (12.0, 45.0),  # ΔE, lower better
    "diversity": (0.40, 0.03),
    "adjacency": (0.03, 0.30),  # lower better
    "sharpness": (200.0, 50.0),
    "identifiability": (0.90, 0.50),
}


def _norm(value: float, good: float, bad: float) -> float:
    """Linear interp, clamp to [0,1]."""
    if good == bad:
        return 1.0
    if good > bad:
        x = (value - bad) / (good - bad)
    else:
        x = (bad - value) / (bad - good)
    return float(max(0.0, min(1.0, x)))


def _grade(total: float) -> str:
    if total >= 85:
        return "A"
    if total >= 70:
        return "B"
    if total >= 55:
        return "C"
    return "D"


def _score_ssim(mosaic_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    h, w = mosaic_rgb.shape[:2]
    target_resized = cv2.resize(target_rgb, (w, h), interpolation=cv2.INTER_AREA)
    g1 = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(target_resized, cv2.COLOR_RGB2GRAY)
    return float(ssim(g1, g2, data_range=255))


def _score_perceptual(
    mosaic_rgb: np.ndarray, target_rgb: np.ndarray
) -> tuple[str, float]:
    """Try LPIPS first; fallback to CLIP cosine."""
    try:
        import lpips
        import torch

        net = lpips.LPIPS(net="alex").eval()
        h, w = mosaic_rgb.shape[:2]
        t = cv2.resize(target_rgb, (w, h), interpolation=cv2.INTER_AREA)
        m_t = torch.tensor(mosaic_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t_t = torch.tensor(t).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad():
            d = float(net(m_t, t_t).item())
        return ("lpips", d)
    except Exception:
        pass

    try:
        import open_clip
        import torch

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device).eval()
        with torch.no_grad():
            m_pil = Image.fromarray(mosaic_rgb)
            t_pil = Image.fromarray(target_rgb)
            m_x = preprocess(m_pil).unsqueeze(0).to(device)
            t_x = preprocess(t_pil).unsqueeze(0).to(device)
            mf = model.encode_image(m_x)
            tf = model.encode_image(t_x)
            mf /= mf.norm(dim=-1, keepdim=True)
            tf /= tf.norm(dim=-1, keepdim=True)
            cos = float((mf @ tf.T).item())
        return ("clip", cos)
    except Exception as e:  # noqa: BLE001
        print(f"[score] perceptual 측정 실패: {e}. 0.5로 처리.")
        return ("none", 0.5)


def _score_color(mosaic_rgb: np.ndarray, target_rgb: np.ndarray, grid: int) -> float:
    h, w = mosaic_rgb.shape[:2]
    cell_h = h // grid
    cell_w = w // grid
    target_resized = cv2.resize(target_rgb, (w, h), interpolation=cv2.INTER_AREA)
    bgr1 = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR)
    bgr2 = cv2.cvtColor(target_resized, cv2.COLOR_RGB2BGR)
    lab1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2LAB).astype(np.float32)
    deltas = []
    for r in range(grid):
        for c in range(grid):
            y0 = r * cell_h
            x0 = c * cell_w
            m = lab1[y0 : y0 + cell_h, x0 : x0 + cell_w].reshape(-1, 3).mean(axis=0)
            t = lab2[y0 : y0 + cell_h, x0 : x0 + cell_w].reshape(-1, 3).mean(axis=0)
            d = float(np.linalg.norm(m - t))
            deltas.append(d)
    return float(np.mean(deltas))


def _score_diversity(cells: list[dict]) -> float:
    return len({c["tile"] for c in cells}) / len(cells)


def _score_adjacency(cells: list[dict], grid: int) -> float:
    grid_arr = [[None] * grid for _ in range(grid)]
    for c in cells:
        grid_arr[c["y"]][c["x"]] = c["tile"]
    same = 0
    total = 0
    for r in range(grid):
        for c in range(grid):
            cur = grid_arr[r][c]
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid and 0 <= nc < grid:
                    total += 1
                    if grid_arr[nr][nc] == cur:
                        same += 1
    return same / max(1, total)


def _score_sharpness_and_id(cells: list[dict], index_csv: Path) -> tuple[float, float]:
    sharps: dict[str, float] = {}
    sizes: dict[str, int] = {}
    with open(index_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sharps[row["path"]] = float(row["sharpness"])
            # disk-side: read from PIL (slow if many tiles, but only for unique)
    used_unique = {c["tile"] for c in cells}
    used_sharp = [sharps.get(t, 0.0) for t in used_unique if t in sharps]
    sharpness_mean = float(np.mean(used_sharp)) if used_sharp else 0.0

    # Identifiability: fraction with min dimension >= 256
    ok = 0
    total = 0
    for t in used_unique:
        full = PROJECT_ROOT / t
        if not full.exists():
            continue
        try:
            with Image.open(full) as img:
                ok += 1 if min(img.size) >= 256 else 0
                total += 1
        except Exception:  # noqa: BLE001
            continue
    identifiability = ok / max(1, total)
    return sharpness_mean, identifiability


def main() -> None:
    parser = argparse.ArgumentParser(description="모자이크 채점")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mosaic-png", type=Path, default=None, help="채점용 렌더 PNG (없으면 자동 추정)")
    parser.add_argument("--index-csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    cells = layout["cells"]
    grid = layout["grid"]["cols"]

    # mosaic png path
    if args.mosaic_png is None:
        stem = args.layout.stem.replace("_layout", "")
        candidate = SCORE_RENDER_DIR / f"{stem}.png"
        args.mosaic_png = candidate
    if not args.mosaic_png.exists():
        raise SystemExit(f"채점용 PNG 없음: {args.mosaic_png}. mosaic.py 다시 실행")

    mosaic_rgb = load_image_rgb(args.mosaic_png)
    target_rgb = load_image_rgb(args.target)
    if mosaic_rgb is None or target_rgb is None:
        raise SystemExit("이미지 로드 실패")
    H, W = target_rgb.shape[:2]
    side = min(H, W)
    target_rgb = target_rgb[(H - side) // 2 : (H + side) // 2, (W - side) // 2 : (W + side) // 2]

    raw_ssim = _score_ssim(mosaic_rgb, target_rgb)
    perc_kind, perc_value = _score_perceptual(mosaic_rgb, target_rgb)
    raw_color = _score_color(mosaic_rgb, target_rgb, grid)
    raw_diversity = _score_diversity(cells)
    raw_adjacency = _score_adjacency(cells, grid)
    raw_sharp, raw_id = _score_sharpness_and_id(cells, args.index_csv)

    n_ssim = _norm(raw_ssim, *CALIBRATION["ssim"])
    perc_cal_key = "perceptual_lpips" if perc_kind == "lpips" else "perceptual_clip"
    n_perc = _norm(perc_value, *CALIBRATION[perc_cal_key]) if perc_kind != "none" else 0.5
    n_color = _norm(raw_color, *CALIBRATION["color"])
    n_div = _norm(raw_diversity, *CALIBRATION["diversity"])
    n_adj = _norm(raw_adjacency, *CALIBRATION["adjacency"])
    n_sharp = _norm(raw_sharp, *CALIBRATION["sharpness"])
    n_id = _norm(raw_id, *CALIBRATION["identifiability"])

    weighted = {
        "ssim": n_ssim * WEIGHTS["ssim"],
        "perceptual": n_perc * WEIGHTS["perceptual"],
        "color": n_color * WEIGHTS["color"],
        "diversity": n_div * WEIGHTS["diversity"],
        "adjacency": n_adj * WEIGHTS["adjacency"],
        "sharpness": n_sharp * WEIGHTS["sharpness"],
        "identifiability": n_id * WEIGHTS["identifiability"],
    }
    total = round(sum(weighted.values()), 2)
    grade = _grade(total)

    result = {
        "layout": str(args.layout),
        "target": str(args.target),
        "grid": grid,
        "raw": {
            "ssim": round(raw_ssim, 4),
            "perceptual": {"kind": perc_kind, "value": round(perc_value, 4)},
            "color_delta_e": round(raw_color, 3),
            "diversity": round(raw_diversity, 4),
            "adjacency": round(raw_adjacency, 4),
            "sharpness": round(raw_sharp, 2),
            "identifiability": round(raw_id, 4),
        },
        "normalized": {
            "ssim": round(n_ssim, 3),
            "perceptual": round(n_perc, 3),
            "color": round(n_color, 3),
            "diversity": round(n_div, 3),
            "adjacency": round(n_adj, 3),
            "sharpness": round(n_sharp, 3),
            "identifiability": round(n_id, 3),
        },
        "weighted": {k: round(v, 2) for k, v in weighted.items()},
        "total": total,
        "grade": grade,
    }

    out_path = args.out or args.layout.with_name(args.layout.stem.replace("_layout", "") + "_score.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[score] {args.layout.name}")
    print(f"  raw ssim={raw_ssim:.3f} | {perc_kind}={perc_value:.3f} | ΔE={raw_color:.1f}")
    print(f"  raw diversity={raw_diversity:.2%} | adjacency={raw_adjacency:.2%} | sharp={raw_sharp:.0f} | id={raw_id:.2%}")
    print(f"\n  weighted: {result['weighted']}")
    print(f"\n  TOTAL = {total} / 100  ({grade})  → {out_path}")


if __name__ == "__main__":
    main()
