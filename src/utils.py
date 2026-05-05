"""Shared utilities for byunjae-leeds."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TILES_DIR = PROJECT_ROOT / "tiles"
THUMBS_DIR = TILES_DIR / "_thumbs"
REJECTED_DIR = TILES_DIR / "_rejected"
TARGETS_DIR = PROJECT_ROOT / "targets"
INDEX_DIR = PROJECT_ROOT / "index"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCORE_RENDER_DIR = OUTPUT_DIR / "_score_render"

INDEX_CSV = INDEX_DIR / "tiles_index.csv"
META_JSON = INDEX_DIR / "tiles_meta.json"

TILE_SOURCES = ["snl", "youtube", "photoshoot", "misc"]
ORIGINAL_SIZE = 256
THUMB_SIZE = 50

HUE_BINS = {
    "red": (345, 360, 0, 15),
    "orange": (15, 45),
    "yellow": (45, 75),
    "green": (75, 165),
    "blue": (165, 255),
    "purple": (255, 345),
}


def ensure_dirs() -> None:
    for d in (TILES_DIR, THUMBS_DIR, REJECTED_DIR, TARGETS_DIR, INDEX_DIR, OUTPUT_DIR, SCORE_RENDER_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for src in TILE_SOURCES:
        (TILES_DIR / src).mkdir(parents=True, exist_ok=True)


def iter_tile_paths(tiles_dir: Path = TILES_DIR) -> list[Path]:
    """tiles/ 안의 실제 타일 파일들 (언더스코어 시작 서브폴더 모두 제외)."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in tiles_dir.rglob(ext):
            rel = p.relative_to(tiles_dir).parts
            if rel and rel[0].startswith("_"):
                continue
            paths.append(p)
    return sorted(paths)


def rel_to_project(path: Path) -> str:
    """프로젝트 루트 기준 상대 경로(슬래시 통일)."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_image_rgb(path: Path) -> np.ndarray | None:
    """PIL로 RGB ndarray 로드. 실패 시 None."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB → float LAB (skimage 호환 범위가 아닌 OpenCV 기준)."""
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab


def avg_lab(rgb: np.ndarray) -> tuple[float, float, float]:
    lab = rgb_to_lab(rgb)
    return float(lab[..., 0].mean()), float(lab[..., 1].mean()), float(lab[..., 2].mean())


def avg_hsv(rgb: np.ndarray) -> tuple[float, float, float]:
    """HSV — H: 0~360 (uint8 H를 *2), S/V: 0~1."""
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = float(hsv[..., 0].mean()) * 2.0
    s = float(hsv[..., 1].mean()) / 255.0
    v = float(hsv[..., 2].mean()) / 255.0
    return h, s, v


def dominant_hue(rgb: np.ndarray) -> str:
    h, s, _ = avg_hsv(rgb)
    if s < 0.15:
        return "neutral"
    if h >= 345 or h < 15:
        return "red"
    if h < 45:
        return "orange"
    if h < 75:
        return "yellow"
    if h < 165:
        return "green"
    if h < 255:
        return "blue"
    return "purple"


def laplacian_sharpness(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_pad(rgb: np.ndarray, size: int) -> np.ndarray:
    """짧은 변을 size에 맞추고 중앙 크롭."""
    h, w = rgb.shape[:2]
    scale = size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (nh - size) // 2
    x0 = (nw - size) // 2
    return resized[y0 : y0 + size, x0 : x0 + size]


def save_jpg(rgb: np.ndarray, path: Path, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "JPEG", quality=quality)


def source_of(path: Path) -> str:
    """tiles/{source}/file.jpg 에서 source 추출."""
    try:
        rel = path.resolve().relative_to(TILES_DIR)
        first = rel.parts[0]
        return first if first in TILE_SOURCES else "misc"
    except ValueError:
        return "misc"
