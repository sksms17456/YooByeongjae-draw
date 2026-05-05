"""Curate ~20 fresh target images via selfie-heuristic search.

Pipeline:
  1. Bing crawl into tiles/_target_candidates/ (separate from tiles/)
  2. InsightFace identity filter (sim>=THRESHOLD, face_ratio>=MIN_FACE_RATIO)
  3. CLIP zero-shot scoring: 'a selfie taken with a phone' vs 'professional studio photo'
  4. Diversity-aware top-K selection per category
  5. Copy winners to targets/ with curated_<cat>_<rank>_<basename>.jpg

Categories:
  selfie     — close-up self portraits
  handsome   — magazine/photoshoot quality face
  funny      — laughing / silly expressions
  recent     — current activity, biased to 2025-2026 keywords
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import (
    PROJECT_ROOT,
    TARGETS_DIR,
    TILES_DIR,
    ensure_dirs,
    laplacian_sharpness,
    load_image_rgb,
)

CANDIDATES_DIR = TILES_DIR / "_target_candidates"

CATEGORY_KEYWORDS = {
    "selfie": [
        "유병재 셀카",
        "유병재 인스타 셀카",
        "유병재 본인 셀카",
        "유병재 셀피",
        "유병재 폰카",
    ],
    "handsome": [
        "유병재 GQ 화보",
        "유병재 잡지 화보",
        "유병재 패션 화보",
        "유병재 ELLE 화보",
    ],
    "funny": [
        "유병재 웃긴 짤",
        "유병재 명장면",
        "유병재 따따부따 짤",
        "유병재 SNL 짤",
    ],
    "recent": [
        "유병재 2026",
        "유병재 2025 활동",
        "유병재 최근",
        "유병재 인스타그램 최근",
        "유병재 자취남 2025",
    ],
    "costume": [
        "유병재 SNL 분장",
        "유병재 분장",
        "유병재 코스튬",
        "유병재 SNL 캐릭터",
        "유병재 가발",
        "유병재 SNL 여장",
    ],
    "selfie2": [
        "유병재 거울 셀카",
        "유병재 인스타 본인 사진",
        "유병재 폰 카메라",
        "유병재 SNS 사진",
        "유병재 트위터 셀카",
        "유병재 일상 사진",
        "Yoo Byung-jae selca",
        "유병재 셀카 모음",
    ],
    "selfie3": [
        "유병재 카페 셀카",
        "유병재 차 안 셀카",
        "유병재 자기 사진",
        "유병재 자찍",
        "유병재 인증샷",
        "유병재 친구와 사진",
        "유병재 야외 셀카",
        "유병재 옷 입은 셀카",
    ],
    "solo": [
        "유병재 단독 사진",
        "유병재 정면 사진",
        "유병재 한장",
        "유병재 단독 화보",
        "유병재 솔로 화보",
        "유병재 인터뷰 단독",
        "유병재 1인 화보",
        "유병재 풀샷",
        "유병재 매거진 화보",
    ],
}

CLIP_PROMPTS = {
    "selfie": (
        "a close-up selfie of one person taken with a phone front camera",
        "a professional studio photograph of a man",
    ),
    "handsome": (
        "a high quality magazine photo of a handsome stylish man",
        "a casual amateur snapshot",
    ),
    "funny": (
        "a man laughing or making a silly funny expression",
        "a serious neutral portrait",
    ),
    "recent": (
        "a casual social media post of one person",
        "an old archival photograph",
    ),
    "costume": (
        "a man wearing costume makeup or theatrical character makeup",
        "a plain casual photo of a man",
    ),
    "selfie2": (
        "a close-up selfie of one person taken with a phone front camera",
        "a professional studio photograph of a man",
    ),
    "selfie3": (
        "a casual snapshot of a man in everyday setting",
        "a professional studio portrait",
    ),
    "solo": (
        "a solo portrait photograph of one man alone",
        "a group photo with multiple people",
    ),
}

SIM_THRESHOLD = 0.40   # stricter than tile filter (0.35)
MIN_FACE_RATIO = 0.06  # stricter than tile filter (0.025) — targets need clean faces
PER_KEYWORD = 25
PER_CATEGORY = 5
SHARPNESS_MIN = 100.0


def _crawl_keyword(keyword: str, target_dir: Path, max_num: int) -> int:
    """Bing only — copy from src/collect.py logic, simpler."""
    from icrawler.builtin import BingImageCrawler

    safe_kw = keyword.replace(" ", "_").replace("/", "_")
    work_dir = target_dir / f"_bing_{safe_kw}"
    work_dir.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(
        downloader_threads=4,
        storage={"root_dir": str(work_dir)},
        log_level=40,
    )
    try:
        crawler.crawl(keyword=keyword, max_num=max_num, min_size=(400, 400))
    except Exception as e:
        print(f"[crawl] {keyword!r} fail: {e}")
        return 0
    moved = 0
    for f in work_dir.iterdir():
        if not f.is_file():
            continue
        dest = target_dir / f"bing_{safe_kw[:30]}_{f.name}"
        if dest.exists():
            continue
        try:
            f.rename(dest)
            moved += 1
        except OSError:
            pass
    try:
        for leftover in work_dir.iterdir():
            leftover.unlink()
        work_dir.rmdir()
    except OSError:
        pass
    return moved


def _phash(p: Path) -> str | None:
    """Square-center-crop before hashing — matches the transform applied when saving
    to targets/, so candidate-vs-target dedup works correctly across runs."""
    try:
        import imagehash
        with Image.open(p) as img:
            img = img.convert("RGB")
            W, H = img.size
            side = min(W, H)
            x0 = (W - side) // 2
            y0 = (H - side) // 2
            img = img.crop((x0, y0, x0 + side, y0 + side))
            return str(imagehash.phash(img))
    except Exception:
        return None


def _existing_phashes() -> set[str]:
    """Hash all current tiles + targets to dedup against."""
    hashes = set()
    for d in (TILES_DIR, TARGETS_DIR):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for p in d.rglob(ext):
                rel = p.relative_to(d).parts
                if rel and rel[0] in {"_thumbs", "_rejected", "_target_candidates"}:
                    continue
                h = _phash(p)
                if h:
                    hashes.add(h)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate ~20 fresh target images")
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    parser.add_argument("--per-keyword", type=int, default=PER_KEYWORD)
    parser.add_argument("--skip-crawl", action="store_true", help="reuse existing _target_candidates/")
    parser.add_argument("--clean-targets", action="store_true", help="remove existing auto_/curated_ targets first")
    parser.add_argument("--categories", nargs="+", default=None, help="제한할 카테고리 목록 (기본: 전체)")
    args = parser.parse_args()

    active_cats = args.categories or list(CATEGORY_KEYWORDS.keys())
    for c in active_cats:
        if c not in CATEGORY_KEYWORDS:
            raise SystemExit(f"unknown category: {c}")

    ensure_dirs()
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean_targets:
        for p in TARGETS_DIR.glob("auto_*.jpg"):
            p.unlink()
        for p in TARGETS_DIR.glob("curated_*.jpg"):
            p.unlink()
        print("[curate] cleaned existing auto_/curated_ targets")

    # -------- Step 1: Crawl --------
    if not args.skip_crawl:
        existing_hashes = _existing_phashes()
        print(f"[curate] existing phashes: {len(existing_hashes)}")

        kw_to_cat = {}
        for cat, kws in CATEGORY_KEYWORDS.items():
            if cat not in active_cats:
                continue
            for kw in kws:
                kw_to_cat[kw] = cat

        total_new = 0
        for kw, cat in kw_to_cat.items():
            cat_dir = CANDIDATES_DIR / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            n = _crawl_keyword(kw, cat_dir, args.per_keyword)
            total_new += n
            print(f"[crawl] [{cat}] {kw!r} → +{n}")

        # dedup new candidates against existing
        removed = 0
        for cat in active_cats:
            cat_dir = CANDIDATES_DIR / cat
            if not cat_dir.exists():
                continue
            for p in list(cat_dir.iterdir()):
                if not p.is_file():
                    continue
                h = _phash(p)
                if h is None:
                    continue
                if h in existing_hashes:
                    p.unlink()
                    removed += 1
                else:
                    existing_hashes.add(h)
        print(f"[curate] crawled {total_new}, dedup-removed {removed}")

    # -------- Step 2: Identity + quality filter per category --------
    from identity_filter import _embeddings, REF_PATH
    if not REF_PATH.exists():
        raise SystemExit(f"reference 없음: {REF_PATH}. /identity-filter --build-ref 먼저")
    ref = np.load(REF_PATH).astype(np.float32)

    # CLIP setup
    import open_clip
    import torch
    device = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(device).eval()
    clip_tok = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def encode_text(prompts):
        toks = clip_tok(prompts).to(device)
        f = clip_model.encode_text(toks)
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def encode_image(rgb):
        img = Image.fromarray(rgb)
        x = clip_preprocess(img).unsqueeze(0).to(device)
        f = clip_model.encode_image(x)
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

    # encode all category text prompts up-front
    cat_text = {}
    for cat, (pos, neg) in CLIP_PROMPTS.items():
        cat_text[cat] = encode_text([pos, neg])  # (2, 512)

    candidates = defaultdict(list)  # cat -> [(score, path, info)]

    for cat in active_cats:
        cat_dir = CANDIDATES_DIR / cat
        if not cat_dir.exists():
            continue
        files = [p for p in cat_dir.iterdir() if p.is_file()]
        print(f"[filter] [{cat}] {len(files)} candidates")
        for p in tqdm(files, desc=f"{cat}"):
            rgb = load_image_rgb(p)
            if rgb is None:
                continue
            H, W = rgb.shape[:2]
            if min(H, W) < 400:
                continue
            sharp = laplacian_sharpness(rgb)
            if sharp < SHARPNESS_MIN:
                continue

            faces = _embeddings(rgb)
            if not faces:
                continue
            # SOLO ONLY: 검출된 얼굴이 정확히 1명이어야 함
            if len(faces) != 1:
                continue
            sims = [(area, ref @ emb, bbox) for area, emb, bbox in faces]
            best = max(sims, key=lambda t: t[1])
            area, sim, bbox = best
            if sim < SIM_THRESHOLD:
                continue
            ratio = area / (H * W)
            if ratio < MIN_FACE_RATIO:
                continue

            # CLIP score: pos similarity - neg similarity (higher = more on-category)
            img_emb = encode_image(rgb)
            cat_score = float(img_emb @ cat_text[cat][0] - img_emb @ cat_text[cat][1])

            # composite ranking score
            composite = 0.5 * float(sim) + 0.3 * cat_score + 0.2 * min(1.0, ratio * 5)
            candidates[cat].append((composite, p, {
                "sim": float(sim),
                "ratio": float(ratio),
                "cat_score": cat_score,
                "sharp": float(sharp),
            }))
        print(f"[filter] [{cat}] survived: {len(candidates[cat])}")

    # -------- Step 3: pick top per category --------
    chosen = []  # (cat, rank, path, info)
    for cat in active_cats:
        ranked = sorted(candidates[cat], key=lambda t: -t[0])
        for rank, (score, p, info) in enumerate(ranked[: args.per_category], 1):
            chosen.append((cat, rank, p, info, score))

    # -------- Step 4: copy to targets/ --------
    for cat, rank, p, info, score in chosen:
        # square crop on face for clean target framing (similar to existing auto_*)
        rgb = load_image_rgb(p)
        H, W = rgb.shape[:2]
        side = min(H, W)
        x0 = (W - side) // 2
        y0 = (H - side) // 2
        cropped = rgb[y0 : y0 + side, x0 : x0 + side]

        out_name = f"curated_{cat}_{rank:02d}_{p.stem[:40]}.jpg"
        out_path = TARGETS_DIR / out_name
        Image.fromarray(cropped).save(out_path, "JPEG", quality=92)
        print(
            f"[chosen] {cat} #{rank}  score={score:.3f}  sim={info['sim']:.2f}  "
            f"ratio={info['ratio']:.2%}  → {out_name}"
        )

    print(f"\n[curate] 완료: {len(chosen)}/{len(active_cats) * args.per_category} 큐레이션됨 → {TARGETS_DIR}")
    print("[curate] 다음: /generate-mosaic 으로 각 타겟 모자이크 생성")


if __name__ == "__main__":
    main()
