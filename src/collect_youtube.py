"""YouTube frame extractor for byunjae-leeds.

Downloads YouTube videos via yt-dlp, then extracts frames at fixed intervals
via OpenCV's VideoCapture (no ffmpeg dependency).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
from PIL import Image

from utils import TILES_DIR, ensure_dirs

# 색·환경 다양한 검색어
DEFAULT_QUERIES = [
    "유병재 SNL 콩트",
    "유병재 인터뷰",
    "유병재 라디오스타",
    "유병재 자취남",
    "유병재 침착맨",
    "유병재 코미디빅리그",
    "유병재 일상 브이로그",
    "유병재 먹방",
    "유병재 야외 촬영",
    "유병재 콘서트 페스티벌",
    "유병재 노래방",
    "유병재 게임",
    "유병재 운동",
    "유병재 여행",
    "유병재 무대 분장",
]

# ROUND 3: 분장/색감 다양화 — 별도 호출용
ROUND3_QUERIES = [
    "유병재 SNL 분장 모음",
    "유병재 SNL 캐릭터 모음",
    "유병재 SNL 여장",
    "유병재 SNL 가발",
    "유병재 영화 소셜포비아",
    "유병재 크라임씬 분장",
    "유병재 캠핑 자연",
    "유병재 골프",
    "유병재 수영",
    "유병재 LED 무대",
    "유병재 컨셉 화보 메이킹",
    "유병재 잡지 화보 비하인드",
]


def _yt_dlp_search(query: str, n: int) -> list[str]:
    """Return list of YouTube video URLs for a search query."""
    try:
        import yt_dlp
    except ImportError:
        print("[yt] yt-dlp 미설치. 'uv add yt-dlp' 후 재시도")
        return []

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "default_search": "ytsearch",
    }
    urls: list[str] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
        except Exception as e:  # noqa: BLE001
            print(f"[yt] search 실패 '{query}': {e}")
            return []
        for entry in info.get("entries", []) or []:
            url = entry.get("url") or entry.get("webpage_url")
            if url:
                if not url.startswith("http"):
                    url = f"https://www.youtube.com/watch?v={url}"
                urls.append(url)
    return urls


def _yt_download(url: str, out_dir: Path) -> Path | None:
    """Download a single video to out_dir, return path."""
    try:
        import yt_dlp
    except ImportError:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "best[height<=480][ext=mp4]/best[height<=480]/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "max_filesize": 60 * 1024 * 1024,  # 60MB cap per video
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as e:  # noqa: BLE001
            print(f"[yt] 다운로드 실패 {url}: {e}")
            return None
    vid = info.get("id")
    if not vid:
        return None
    candidates = list(out_dir.glob(f"{vid}.*"))
    if not candidates:
        return None
    return candidates[0]


def _extract_frames(video_path: Path, target_dir: Path, every_sec: float, max_frames: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every_sec)))

    saved = 0
    frame_idx = 0
    base = target_dir / f"yt_{video_path.stem}"
    target_dir.mkdir(parents=True, exist_ok=True)

    while saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        # discard tiny frames
        if frame.shape[0] < 256 or frame.shape[1] < 256:
            frame_idx += step
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_path = target_dir / f"{base.name}_{saved:03d}.jpg"
        Image.fromarray(rgb).save(out_path, "JPEG", quality=88)
        saved += 1
        frame_idx += step
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube 프레임 추출 수집기")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--videos-per-query", type=int, default=2)
    parser.add_argument("--every-sec", type=float, default=4.0, help="N초 간격 프레임 추출")
    parser.add_argument("--max-frames", type=int, default=40, help="영상당 최대 프레임 수")
    parser.add_argument("--source", default="youtube", help="저장 서브폴더")
    args = parser.parse_args()

    ensure_dirs()
    out_dir = TILES_DIR / args.source

    with tempfile.TemporaryDirectory(prefix="byunjae_yt_") as tmp:
        tmp_dir = Path(tmp)
        total_videos = 0
        total_frames = 0
        for q in args.queries:
            print(f"\n[yt] '{q}' 검색…")
            urls = _yt_dlp_search(q, args.videos_per_query)
            print(f"[yt]   {len(urls)}개 후보 URL")
            for url in urls[: args.videos_per_query]:
                vid_path = _yt_download(url, tmp_dir)
                if not vid_path or not vid_path.exists():
                    continue
                total_videos += 1
                n = _extract_frames(vid_path, out_dir, args.every_sec, args.max_frames)
                total_frames += n
                print(f"[yt]   {vid_path.name} → {n}프레임")
                try:
                    vid_path.unlink()
                except OSError:
                    pass
        print(f"\n[yt] 영상 {total_videos}개에서 총 {total_frames}프레임 추출 → {out_dir}")
        print("[yt] 다음: /classify-tiles")


if __name__ == "__main__":
    main()
