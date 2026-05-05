"""Precompute CLIP image embeddings for all tiles.

Saves index/tile_clip.npy (N, 512) L2-normalized + index/tile_clip_paths.json
that lines up with tiles_index.csv 'path' column order.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (
    INDEX_CSV,
    INDEX_DIR,
    PROJECT_ROOT,
    load_image_rgb,
)

CLIP_NPY = INDEX_DIR / "tile_clip.npy"
CLIP_PATHS_JSON = INDEX_DIR / "tile_clip_paths.json"


def _model():
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute tile CLIP embeddings")
    parser.add_argument("--csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--out", type=Path, default=CLIP_NPY)
    parser.add_argument("--paths-out", type=Path, default=CLIP_PATHS_JSON)
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV 없음: {args.csv}")

    paths: list[str] = []
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
    print(f"[clip-index] {len(paths)} tiles")

    from PIL import Image
    model, preprocess, device, torch = _model()
    print(f"[clip-index] device={device}")

    embs = np.zeros((len(paths), 512), dtype=np.float32)
    batch_imgs = []
    batch_idx = []
    BATCH = 64

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

    for i, rel in enumerate(tqdm(paths)):
        p = PROJECT_ROOT / rel
        rgb = load_image_rgb(p)
        if rgb is None:
            continue
        img = Image.fromarray(rgb)
        x = preprocess(img)
        batch_imgs.append(x)
        batch_idx.append(i)
        if len(batch_imgs) >= BATCH:
            flush()
    flush()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, embs)
    args.paths_out.write_text(json.dumps(paths, ensure_ascii=False), encoding="utf-8")
    print(f"[clip-index] saved → {args.out} (shape={embs.shape})")
    print(f"[clip-index] path order → {args.paths_out}")


if __name__ == "__main__":
    main()
