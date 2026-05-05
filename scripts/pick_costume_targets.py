"""기존 _target_candidates/costume/에서 중복 없는 단독 코스튬 3장을 추가."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import TARGETS_DIR, TILES_DIR, load_image_rgb, laplacian_sharpness  # noqa: E402

CANDIDATES = TILES_DIR / "_target_candidates" / "costume_v2"
KEYWORDS = [
    "유병재 코스프레",
    "유병재 SNL 분장 단독",
    "유병재 분장 캐릭터",
    "유병재 SNL 가발",
    "유병재 SNL 여장",
    "유병재 SNL 캐릭터 분장",
    "유병재 분장 화보",
    "유병재 코스튬 단독",
]
SIM_THRESHOLD = 0.40
MIN_FACE_RATIO = 0.08  # 얼굴이 이미지의 8% 이상 차지해야
SHARPNESS_MIN = 100.0
N_PICK = 15  # 더 많이 뽑아서 코스프레 후보 추가 탐색


def square_phash(p: Path) -> str | None:
    import imagehash
    try:
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


def square_phash_16(p: Path):
    import imagehash
    try:
        with Image.open(p) as img:
            img = img.convert("RGB")
            W, H = img.size
            side = min(W, H)
            x0 = (W - side) // 2
            y0 = (H - side) // 2
            img = img.crop((x0, y0, x0 + side, y0 + side))
            return imagehash.phash(img, hash_size=16), imagehash.dhash(img, hash_size=16)
    except Exception:
        return None, None


def existing_target_hashes() -> list:
    out = []
    for p in TARGETS_DIR.glob("curated_*.jpg"):
        ph, dh = square_phash_16(p)
        if ph is not None:
            out.append((p.name, ph, dh))
    return out


def crawl_fresh():
    from icrawler.builtin import BingImageCrawler
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    for kw in KEYWORDS:
        safe = kw.replace(" ", "_").replace("/", "_")
        work = CANDIDATES / f"_bing_{safe}"
        work.mkdir(parents=True, exist_ok=True)
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": str(work)},
            log_level=40,
        )
        try:
            crawler.crawl(keyword=kw, max_num=30, min_size=(400, 400))
        except Exception as e:
            print(f"[crawl] {kw!r} fail: {e}")
            continue
        moved = 0
        for f in work.iterdir():
            if not f.is_file():
                continue
            dest = CANDIDATES / f"bing_{safe[:30]}_{f.name}"
            if dest.exists():
                continue
            try:
                f.rename(dest)
                moved += 1
            except OSError:
                pass
        try:
            for leftover in work.iterdir():
                leftover.unlink()
            work.rmdir()
        except OSError:
            pass
        print(f"[crawl] '{kw}' +{moved}")


def main():
    if not CANDIDATES.exists() or not list(CANDIDATES.glob("*.jpg")):
        print("[pick] 신규 크롤 시작...")
        crawl_fresh()
    from identity_filter import _embeddings, REF_PATH
    if not REF_PATH.exists():
        raise SystemExit(f"reference 없음: {REF_PATH}")
    ref = np.load(REF_PATH).astype(np.float32)

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
    tok = open_clip.get_tokenizer("ViT-B-32")

    POS = "a man wearing costume makeup or theatrical character makeup or stage costume"
    NEG = "a plain casual photo of a man with no costume"

    @torch.no_grad()
    def encode_text(prompts):
        toks = tok(prompts).to(device)
        f = model.encode_text(toks)
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def encode_image(rgb):
        img = Image.fromarray(rgb)
        x = preprocess(img).unsqueeze(0).to(device)
        f = model.encode_image(x)
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

    text_emb = encode_text([POS, NEG])

    existing = existing_target_hashes()
    print(f"[pick] 기존 타겟 {len(existing)}장 phash 로드")

    files = sorted([p for p in CANDIDATES.iterdir() if p.is_file()])
    print(f"[pick] 후보 {len(files)}장 평가")

    scored = []
    reasons = {"size": 0, "sharp": 0, "no_face": 0, "multi_face": 0, "low_sim": 0, "small_face": 0, "dup": 0}
    for p in files:
        rgb = load_image_rgb(p)
        if rgb is None:
            reasons["size"] += 1
            continue
        H, W = rgb.shape[:2]
        if min(H, W) < 400:
            reasons["size"] += 1
            continue
        sharp = laplacian_sharpness(rgb)
        if sharp < SHARPNESS_MIN:
            reasons["sharp"] += 1
            continue
        faces = _embeddings(rgb)
        if not faces:
            reasons["no_face"] += 1
            continue
        if len(faces) != 1:  # SOLO ONLY
            reasons["multi_face"] += 1
            continue
        area, emb, bbox = faces[0]
        sim = float(ref @ emb)
        if sim < SIM_THRESHOLD:
            reasons["low_sim"] += 1
            continue
        ratio = area / (H * W)
        if ratio < MIN_FACE_RATIO:
            reasons["small_face"] += 1
            continue

        # phash dedup vs existing targets
        cand_ph, cand_dh = square_phash_16(p)
        is_dup = False
        for name, eph, edh in existing:
            ph_d = cand_ph - eph
            dh_d = cand_dh - edh
            if ph_d + dh_d <= 16:  # 강한 유사
                is_dup = True
                break
        if is_dup:
            reasons["dup"] += 1
            continue

        img_emb = encode_image(rgb)
        clip_score = float(img_emb @ text_emb[0] - img_emb @ text_emb[1])
        composite = 0.5 * sim + 0.4 * clip_score + 0.1 * min(1.0, ratio * 5)
        scored.append((composite, sim, clip_score, ratio, p))
    print(f"[pick] 탈락 사유: {reasons}")

    scored.sort(key=lambda t: -t[0])
    print(f"[pick] 통과 후보 {len(scored)}장 (정렬됨)")
    print()
    for i, (comp, sim, clip_s, ratio, p) in enumerate(scored[:10]):
        marker = "★ 선정" if i < N_PICK else ""
        print(f"  #{i+1} comp={comp:.3f} sim={sim:.2f} clip={clip_s:+.3f} face={ratio:.2%} {p.name} {marker}")

    if len(scored) < N_PICK:
        raise SystemExit(f"선정 가능한 후보 부족: {len(scored)} < {N_PICK}")

    # 기존 costume 타겟 다음 번호부터 부여
    existing_costume = sorted(TARGETS_DIR.glob("curated_costume_*.jpg"))
    next_num = 6  # 기존 _05 다음

    # 기존 타겟 CLIP 임베딩 (같은 장면 검출용)
    existing_clip = []
    for tp in TARGETS_DIR.glob("curated_*.jpg"):
        rgb = load_image_rgb(tp)
        if rgb is None:
            continue
        existing_clip.append((tp.name, encode_image(rgb)))

    print()
    # 모든 통과 후보의 CLIP 임베딩 사전 계산 (greedy farthest-first 위해)
    print("[pick] 후보 CLIP 임베딩 계산 중...")
    cand_data = []  # (comp, p, clip_emb, ph16, dh16)
    for comp, sim, clip_s, ratio, p in scored:
        rgb = load_image_rgb(p)
        emb = encode_image(rgb)
        ph, dh = square_phash_16(p)
        cand_data.append((comp, p, emb, ph, dh))

    # Greedy farthest-first: 첫 번째는 최고점, 이후는 기존 선정과 가장 다른 것
    chosen = []  # [(comp, p, emb, ph, dh)]
    REJECT_CLIP = 0.93  # 기존 타겟과 거의 똑같으면 제외 (정상적으로 다른 콩트는 0.85+ 안 갈 가능성↑)
    REJECT_HASH = 18

    # 기존 타겟과 유사한 후보는 사전 제외
    filtered = []
    for cd in cand_data:
        comp, p, emb, ph, dh = cd
        skip = False
        for name, eemb in existing_clip:
            if float(emb @ eemb) >= REJECT_CLIP:
                skip = True
                break
        if not skip:
            filtered.append(cd)
    print(f"[pick] 기존 타겟과 유사한 후보 제외 후 {len(filtered)}장")

    if not filtered:
        raise SystemExit("필터 후 후보 없음")

    # 1) 최고점부터 시작
    chosen.append(filtered[0])
    print(f"[chosen #1] {filtered[0][1].name} (comp={filtered[0][0]:.3f})")

    # 2~N) farthest-first: 기존 선정과 max CLIP 유사도가 최소인 후보
    while len(chosen) < N_PICK:
        best_cand = None
        best_max_sim = 1.1
        for cd in filtered:
            if cd in chosen:
                continue
            comp, p, emb, ph, dh = cd
            max_sim = max(float(emb @ ce) for _, _, ce, _, _ in chosen)
            # composite: 다양성 우선, 동률시 점수
            if max_sim < best_max_sim - 0.001:
                best_max_sim = max_sim
                best_cand = cd
            elif abs(max_sim - best_max_sim) < 0.001 and best_cand and comp > best_cand[0]:
                best_cand = cd
        if best_cand is None:
            break
        chosen.append(best_cand)
        print(f"[chosen #{len(chosen)}] {best_cand[1].name} (comp={best_cand[0]:.3f}, max_sim_to_chosen={best_max_sim:.3f})")

    # 저장 — 얼굴 기반 정사각 크롭
    print()
    saved = 0
    for comp, p, emb, ph, dh in chosen:
        rgb = load_image_rgb(p)
        H, W = rgb.shape[:2]
        side = min(H, W)

        faces = _embeddings(rgb)
        if faces:
            area, _, bbox = max(faces, key=lambda f: f[0])
            fx1, fy1, fx2, fy2 = bbox
            cx = (fx1 + fx2) / 2
            cy = (fy1 + fy2) / 2
            x0 = int(round(cx - side / 2))
            y0 = int(round(cy - side / 2))
            x0 = max(0, min(x0, W - side))
            y0 = max(0, min(y0, H - side))
        else:
            x0 = (W - side) // 2
            y0 = (H - side) // 2

        cropped = rgb[y0:y0+side, x0:x0+side]

        # 후보 폴더에 임시 저장 (시각 확인용)
        preview_dir = TILES_DIR / "_costume_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"cand_{saved+1:02d}_{p.stem[:40]}.jpg"
        out_path = preview_dir / out_name
        Image.fromarray(cropped).save(out_path, "JPEG", quality=92)
        saved += 1
        print(f"[preview] {out_name} ({W}x{H} → {side}x{side} face-crop)")

    total = len(list(TARGETS_DIR.glob("curated_*.jpg")))
    print(f"\n[pick] 완료. 현재 총 타겟 {total}장")


if __name__ == "__main__":
    main()
