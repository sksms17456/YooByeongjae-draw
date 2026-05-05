"""Identity filter — keep tiles where 유병재 is the dominant subject.

Two phases:
  1. --build-ref: compute reference embedding from clean reference photos
  2. (default) filter tiles/ using reference, moving rejects to _rejected/

Reject reasons:
  - no_face: InsightFace 검출 실패
  - not_byunjae: 모든 검출 얼굴이 reference 와 유사도 < threshold
  - tiny_byunjae: 유병재 얼굴 영역 비율 < min_face_ratio
  - byunjae_not_main: 다른 얼굴이 더 크면서 유병재 얼굴이 주피사체 아님
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (
    INDEX_DIR,
    REJECTED_DIR,
    TILES_DIR,
    ensure_dirs,
    iter_tile_paths,
    load_image_rgb,
)

REF_PATH = INDEX_DIR / "byunjae_ref.npy"
REPORT_PATH = INDEX_DIR / "identity_report.json"
BBOX_PATH = INDEX_DIR / "byunjae_face_bboxes.json"

DEFAULT_SIM_THRESHOLD = 0.35
DEFAULT_MIN_FACE_RATIO = 0.025  # face area / image area
DEFAULT_DOMINANCE_RATIO = 0.85  # 유병재 face / largest face area must be >=

_APP = None


def _get_app():
    global _APP
    if _APP is not None:
        return _APP
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise RuntimeError(
            "insightface 미설치. 'uv add insightface onnxruntime' 후 재시도."
        ) from e

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 → CPU
    _APP = app
    return app


def _embeddings(rgb: np.ndarray):
    """Return list of (bbox_area, normalized_embedding, bbox) tuples."""
    app = _get_app()
    # InsightFace expects BGR
    bgr = rgb[..., ::-1].copy()
    faces = app.get(bgr)
    out = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        area = max(0.0, (x2 - x1) * (y2 - y1))
        emb = f.normed_embedding  # 512-dim L2 normalized
        out.append((float(area), emb.astype(np.float32), (float(x1), float(y1), float(x2), float(y2))))
    return out


def build_reference(ref_paths: list[Path]) -> np.ndarray:
    """Compute mean reference embedding from N clean photos (largest face each)."""
    embs: list[np.ndarray] = []
    for p in ref_paths:
        rgb = load_image_rgb(p)
        if rgb is None:
            print(f"[ref] skip unreadable: {p}")
            continue
        faces = _embeddings(rgb)
        if not faces:
            print(f"[ref] no face in {p.name} — skip")
            continue
        # largest face
        faces.sort(key=lambda t: t[0], reverse=True)
        embs.append(faces[0][1])
        print(f"[ref] {p.name} → embedding 추가 (faces={len(faces)})")
    if not embs:
        raise SystemExit("[ref] reference 임베딩 추출 실패")
    mean = np.mean(np.stack(embs), axis=0)
    mean /= np.linalg.norm(mean) + 1e-8
    return mean.astype(np.float32)


def _move(path: Path, reason: str) -> None:
    sub = REJECTED_DIR / reason
    sub.mkdir(parents=True, exist_ok=True)
    target = sub / path.name
    if target.exists():
        target = sub / f"{path.stem}_{path.parent.name}{path.suffix}"
    shutil.move(str(path), str(target))


def filter_tiles(
    ref: np.ndarray,
    sim_threshold: float,
    min_face_ratio: float,
    dominance_ratio: float,
    dry_run: bool,
) -> tuple[dict, dict]:
    paths = iter_tile_paths()
    report = {
        "total": len(paths),
        "kept": 0,
        "rejected": {"no_face": 0, "not_byunjae": 0, "tiny_byunjae": 0, "byunjae_not_main": 0},
        "samples": {"kept": [], "rejected": {}},
    }
    bboxes: dict[str, dict] = {}  # path → {bbox, sim, ratio}

    for p in tqdm(paths, desc="identity"):
        rgb = load_image_rgb(p)
        if rgb is None:
            continue
        H, W = rgb.shape[:2]
        img_area = float(H * W)

        faces = _embeddings(rgb)
        if not faces:
            report["rejected"]["no_face"] += 1
            report["samples"].setdefault("rejected", {}).setdefault("no_face", []).append(str(p))
            if not dry_run:
                _move(p, "no_face")
            continue

        # similarity for each face
        sims = [(area, ref @ emb, bbox) for area, emb, bbox in faces]
        # is there any 유병재 face?
        byunjae_faces = [(a, s, b) for a, s, b in sims if s >= sim_threshold]
        if not byunjae_faces:
            report["rejected"]["not_byunjae"] += 1
            report["samples"].setdefault("rejected", {}).setdefault("not_byunjae", []).append(str(p))
            if not dry_run:
                _move(p, "not_byunjae")
            continue

        # largest 유병재 face
        byunjae_faces.sort(key=lambda t: t[0], reverse=True)
        bj_area, bj_sim, bj_bbox = byunjae_faces[0]
        bj_ratio = bj_area / img_area

        if bj_ratio < min_face_ratio:
            report["rejected"]["tiny_byunjae"] += 1
            report["samples"].setdefault("rejected", {}).setdefault("tiny_byunjae", []).append(str(p))
            if not dry_run:
                _move(p, "tiny_byunjae")
            continue

        # dominance check: 유병재 face must be ≥ dominance_ratio × largest face
        largest_area = max(area for area, _, _ in sims)
        if bj_area / largest_area < dominance_ratio:
            report["rejected"]["byunjae_not_main"] += 1
            report["samples"].setdefault("rejected", {}).setdefault("byunjae_not_main", []).append(str(p))
            if not dry_run:
                _move(p, "byunjae_not_main")
            continue

        # keep + record bbox for downstream face-region LAB reindex
        report["kept"] += 1
        x1, y1, x2, y2 = bj_bbox
        from utils import rel_to_project as _rel
        rel = _rel(p)
        bboxes[rel] = {
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "sim": round(float(bj_sim), 3),
            "ratio": round(float(bj_ratio), 4),
        }
        if len(report["samples"]["kept"]) < 20:
            report["samples"]["kept"].append({"path": rel, "sim": round(float(bj_sim), 3), "ratio": round(bj_ratio, 4)})

    # cap rejected samples at 20 each
    for k, v in report["samples"].get("rejected", {}).items():
        report["samples"]["rejected"][k] = v[:20]

    return report, bboxes


def main() -> None:
    parser = argparse.ArgumentParser(description="유병재 정체성 필터 (InsightFace)")
    parser.add_argument("--build-ref", nargs="+", type=Path, help="reference 이미지 경로들 (clean frontal 유병재)")
    parser.add_argument("--ref", type=Path, default=REF_PATH, help="reference 임베딩 npy 경로")
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    parser.add_argument("--min-face-ratio", type=float, default=DEFAULT_MIN_FACE_RATIO)
    parser.add_argument("--dominance-ratio", type=float, default=DEFAULT_DOMINANCE_RATIO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    if args.build_ref:
        ref = build_reference(args.build_ref)
        args.ref.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.ref, ref)
        print(f"[ref] saved → {args.ref} (dim={ref.shape[0]})")
        return

    if not args.ref.exists():
        raise SystemExit(f"reference 임베딩 없음: {args.ref}. --build-ref 먼저 실행")
    ref = np.load(args.ref).astype(np.float32)
    if ref.ndim != 1:
        raise SystemExit(f"reference shape 이상: {ref.shape}")

    print(
        f"[id-filter] sim>={args.sim_threshold} | face_ratio>={args.min_face_ratio} | "
        f"dominance>={args.dominance_ratio} | dry_run={args.dry_run}"
    )
    report, bboxes = filter_tiles(
        ref,
        args.sim_threshold,
        args.min_face_ratio,
        args.dominance_ratio,
        args.dry_run,
    )

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    BBOX_PATH.write_text(json.dumps(bboxes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[id-filter] 총 {report['total']}장 → 유지 {report['kept']}장")
    for reason, n in report["rejected"].items():
        print(f"  rejected[{reason}]: {n}장")
    print(f"[id-filter] report → {REPORT_PATH}")
    print(f"[id-filter] face bboxes → {BBOX_PATH}")


if __name__ == "__main__":
    main()
