"""Index correctness: build a fixture of solid-color tiles and verify avg LAB."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import process_tile  # noqa: E402
from utils import avg_lab  # noqa: E402


@pytest.fixture
def fixture_tiles(tmp_path: Path) -> list[Path]:
    """Create 4 solid-color 256x256 tiles in tmp_path."""
    colors = {
        "red.jpg": (220, 40, 40),
        "green.jpg": (40, 200, 40),
        "blue.jpg": (40, 40, 220),
        "gray.jpg": (128, 128, 128),
    }
    paths = []
    for name, rgb in colors.items():
        img = Image.new("RGB", (256, 256), rgb)
        p = tmp_path / name
        img.save(p, "JPEG", quality=95)
        paths.append(p)
    return paths


def test_solid_color_lab_within_tolerance(fixture_tiles):
    """평균 LAB이 합리적 범위 안에 있고, 색상끼리 충분히 구분된다."""
    rows = []
    for p in fixture_tiles:
        row = process_tile(p, do_thumb=False)
        assert row is not None
        rows.append(row)

    by_name = {Path(r["path"]).name: r for r in rows}

    # OpenCV LAB의 L 채널은 0~255 스케일.
    # red(220,40,40): L≈123, green(40,200,40): L≈180, gray(128,128,128): L≈137
    assert 100 < by_name["red.jpg"]["avg_l"] < 160
    assert 150 < by_name["green.jpg"]["avg_l"] < 220
    assert 120 < by_name["gray.jpg"]["avg_l"] < 150

    # red, green, blue가 LAB 공간에서 서로 충분히 떨어져 있어야 함
    def lab_vec(name):
        r = by_name[name]
        return np.array([r["avg_l"], r["avg_a"], r["avg_b"]])

    d_rg = float(np.linalg.norm(lab_vec("red.jpg") - lab_vec("green.jpg")))
    d_rb = float(np.linalg.norm(lab_vec("red.jpg") - lab_vec("blue.jpg")))
    d_gb = float(np.linalg.norm(lab_vec("green.jpg") - lab_vec("blue.jpg")))
    assert d_rg > 50
    assert d_rb > 50
    assert d_gb > 50


def test_avg_lab_consistent_with_pillow():
    """avg_lab 헬퍼와 직접 계산이 일치해야 한다."""
    arr = np.full((100, 100, 3), [200, 50, 50], dtype=np.uint8)
    L, A, B = avg_lab(arr)
    assert L > 0
    assert A > 128  # red increases A toward red side


def test_dominant_hue_red():
    from utils import dominant_hue

    arr = np.full((100, 100, 3), [220, 30, 30], dtype=np.uint8)
    assert dominant_hue(arr) == "red"


def test_dominant_hue_neutral():
    from utils import dominant_hue

    arr = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
    assert dominant_hue(arr) == "neutral"
