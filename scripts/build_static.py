"""Vercel 배포용 정적 사이트 빌드.

FastAPI 엔드포인트를 정적 JSON으로 사전 렌더링하고
public/ 디렉터리에 모든 자산을 모은다.

실행: uv run python scripts/build_static.py
출력: ./public/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import OUTPUT_DIR, TARGETS_DIR, THUMBS_DIR, TILES_DIR  # noqa: E402

PUBLIC = ROOT / "public"
WEB = ROOT / "web"


def _clean():
    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    PUBLIC.mkdir()


def _copy_web():
    # index.html은 루트, 나머지는 /static/ 경로로 (FastAPI 마운트 호환)
    shutil.copy2(WEB / "index.html", PUBLIC / "index.html")
    static_dir = PUBLIC / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    for name in ("app.js", "style.css", "favicon.png", "apple-touch-icon.png"):
        src = WEB / name
        if src.exists():
            shutil.copy2(src, static_dir / name)
    # favicon은 루트에도 (브라우저 자동 요청)
    if (WEB / "favicon.png").exists():
        shutil.copy2(WEB / "favicon.png", PUBLIC / "favicon.ico")
    print(f"[build] web → public/index.html + public/static/")


def _build_targets_json():
    """/api/targets 와 동일 형식의 정적 JSON 생성."""
    items = []
    for p in sorted(TARGETS_DIR.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if p.name.startswith("_"):
            continue
        layouts = sorted(
            OUTPUT_DIR.glob(f"mosaic_{p.stem}_*_layout.json"),
            key=lambda x: x.stat().st_mtime,
        )
        if not layouts:
            continue
        items.append({
            "stem": p.stem,
            "filename": p.name,
            "thumb_url": f"/targets/{p.name}",
            "layout": layouts[-1].name,
        })

    out_dir = PUBLIC / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "targets.json").write_text(json.dumps(items, ensure_ascii=False))
    print(f"[build] data/targets.json ({len(items)}장)")
    return items


def _build_layouts_json(targets):
    """각 타겟의 latest layout JSON을 stem-key 정적 파일로 복사."""
    out_dir = PUBLIC / "data" / "layouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in targets:
        stem = t["stem"]
        layouts = sorted(
            OUTPUT_DIR.glob(f"mosaic_{stem}_*_layout.json"),
            key=lambda x: x.stat().st_mtime,
        )
        if not layouts:
            continue
        data = json.loads(layouts[-1].read_text(encoding="utf-8"))
        (out_dir / f"{stem}.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"[build] data/layouts/ ({len(targets)}개)")


def _copy_targets():
    out_dir = PUBLIC / "targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in TARGETS_DIR.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and not p.name.startswith("_"):
            shutil.copy2(p, out_dir / p.name)
            n += 1
    print(f"[build] targets/ ({n}장)")


def _copy_tile_assets(layouts_dir: Path):
    """레이아웃에서 실제 사용된 타일만 복사 (전체 풀 아님 — 디스크 절감)."""
    used = set()
    for f in layouts_dir.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        for cell in data.get("cells", []):
            tile = cell.get("tile")
            if tile:
                used.add(tile)
    print(f"[build] 사용 타일: {len(used)}개")

    thumbs_dir = PUBLIC / "tiles_thumb"
    full_dir = PUBLIC / "tiles_full"

    n_thumb, n_full = 0, 0
    for tile_path in sorted(used):
        # tile_path 예: "tiles/photoshoot/bing_xxx.jpg"
        rel = Path(tile_path).relative_to("tiles")
        # thumb
        thumb_src = THUMBS_DIR / rel
        if thumb_src.exists():
            dst = thumbs_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(thumb_src, dst)
            n_thumb += 1
        # full
        full_src = TILES_DIR / rel
        if full_src.exists():
            dst = full_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(full_src, dst)
            n_full += 1

    print(f"[build] tiles_thumb/ ({n_thumb}장), tiles_full/ ({n_full}장)")


def _write_vercel_config():
    config = {
        "cleanUrls": True,
        "trailingSlash": False,
    }
    (PUBLIC / "vercel.json").write_text(json.dumps(config, indent=2))
    print("[build] vercel.json")


def _disk_usage():
    total = sum(p.stat().st_size for p in PUBLIC.rglob("*") if p.is_file())
    print(f"\n[build] public/ 총 {total / (1024*1024):.1f} MB")


def main():
    _clean()
    _copy_web()
    targets = _build_targets_json()
    _build_layouts_json(targets)
    _copy_targets()
    _copy_tile_assets(PUBLIC / "data" / "layouts")
    _write_vercel_config()
    _disk_usage()
    print("\n[build] 완료. 배포: cd public && vercel --prod")


if __name__ == "__main__":
    main()
