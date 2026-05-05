"""FastAPI server for byunjae-leeds web viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from utils import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    TARGETS_DIR,
    THUMBS_DIR,
    TILES_DIR,
)

WEB_DIR = PROJECT_ROOT / "web"
app = FastAPI(title="byunjae-leeds")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(WEB_DIR / "favicon.png", media_type="image/png")


@app.get("/api/targets")
def list_targets():
    items = []
    if TARGETS_DIR.exists():
        for p in sorted(TARGETS_DIR.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                # find latest layout for this target
                layouts = sorted(
                    OUTPUT_DIR.glob(f"mosaic_{p.stem}_*_layout.json"),
                    key=lambda x: x.stat().st_mtime,
                )
                latest_layout = layouts[-1].name if layouts else None
                items.append(
                    {
                        "stem": p.stem,
                        "filename": p.name,
                        "thumb_url": f"/targets/{p.name}",
                        "layout": latest_layout,
                    }
                )
    return items


@app.get("/api/layouts/{stem}")
def get_layout(stem: str):
    layouts = sorted(
        OUTPUT_DIR.glob(f"mosaic_{stem}_*_layout.json"), key=lambda x: x.stat().st_mtime
    )
    if not layouts:
        raise HTTPException(404, f"{stem} layout 없음. mosaic.py 또는 improve.py 실행 필요.")
    data = json.loads(layouts[-1].read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.get("/api/scores/{stem}")
def get_score(stem: str):
    scores = sorted(
        OUTPUT_DIR.glob(f"mosaic_{stem}_*_score.json"), key=lambda x: x.stat().st_mtime
    )
    if not scores:
        return JSONResponse({"score": None})
    data = json.loads(scores[-1].read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.get("/tiles_thumb/{path:path}")
def serve_thumb(path: str):
    thumb = THUMBS_DIR / Path(path).relative_to(Path("tiles")) if path.startswith("tiles/") else THUMBS_DIR / path
    if not thumb.exists():
        # Fallback: full tile
        full = PROJECT_ROOT / path
        if full.exists():
            return FileResponse(full)
        raise HTTPException(404, str(thumb))
    return FileResponse(thumb)


@app.get("/tiles_full/{path:path}")
def serve_full(path: str):
    full = PROJECT_ROOT / path if path.startswith("tiles/") else TILES_DIR / path
    if not full.exists():
        raise HTTPException(404, str(full))
    return FileResponse(full)


# Static targets
app.mount("/targets", StaticFiles(directory=str(TARGETS_DIR)), name="targets")
# Static web assets
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="byunjae-leeds 웹 서버")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    print(f"[server] http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
