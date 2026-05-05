"""Image crawler for byunjae-leeds.

Uses icrawler (Bing/Google) to download images by keyword. Stores into
tiles/{source}/ with deduplication via perceptual hash.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from utils import OUTPUT_DIR, TILES_DIR, ensure_dirs, iter_tile_paths

# 색·장면·의상·계절 다양성을 노린 광범위 키워드
# (keyword, source_subdir)
KEYWORDS = [
    # 기본
    ("유병재", "misc"),
    ("유병재 코미디언", "misc"),
    ("유병재 방송", "misc"),
    ("유병재 인스타", "misc"),
    ("유병재 인스타그램", "misc"),
    ("Yoo Byung Jae", "misc"),
    ("Yoo Byung-jae comedian", "misc"),
    # SNL · 예능 (다양한 콩트 = 다양한 의상/배경/색)
    ("유병재 SNL", "snl"),
    ("유병재 SNL 코리아", "snl"),
    ("SNL 유병재 콩트", "snl"),
    ("SNL 유병재 분장", "snl"),
    ("SNL 유병재 캐릭터", "snl"),
    ("유병재 무한도전", "snl"),
    ("유병재 라디오스타", "snl"),
    ("유병재 코미디빅리그", "snl"),
    ("유병재 뇌피셜", "snl"),
    # 화보 · 광고 (밝은 색, 컬러풀한 배경)
    ("유병재 화보", "photoshoot"),
    ("유병재 잡지", "photoshoot"),
    ("유병재 광고", "photoshoot"),
    ("유병재 패션화보", "photoshoot"),
    ("유병재 ELLE", "photoshoot"),
    ("유병재 보그", "photoshoot"),
    ("유병재 포스터", "photoshoot"),
    ("유병재 시상식", "photoshoot"),
    ("유병재 레드카펫", "photoshoot"),
    ("유병재 행사", "photoshoot"),
    # 유튜브 · 방송 콘텐츠
    ("유병재 유튜브", "youtube"),
    ("유병재 인터뷰", "youtube"),
    ("유병재 토크쇼", "youtube"),
    ("유병재 라디오", "youtube"),
    ("유병재 자취남", "youtube"),
    ("유병재 침착맨", "youtube"),
    ("유병재 셀프카메라", "youtube"),
    ("유병재 일상", "youtube"),
    ("유병재 브이로그", "youtube"),
    # 영화·드라마 (다른 톤·색)
    ("유병재 크라임씬", "photoshoot"),
    ("유병재 크라임씬 리턴즈", "photoshoot"),
    ("유병재 영화", "photoshoot"),
    # 색·환경 다양성
    ("유병재 야외", "misc"),
    ("유병재 카페", "misc"),
    ("유병재 공항", "misc"),
    ("유병재 패션", "misc"),
    ("유병재 의상", "misc"),
    ("유병재 안경", "misc"),
    ("유병재 정장", "misc"),
    ("유병재 후드", "misc"),
    ("유병재 모자", "misc"),
    ("유병재 셔츠", "misc"),
    # 표정 다양성
    ("유병재 표정", "misc"),
    ("유병재 웃음", "misc"),
    ("유병재 깜짝", "misc"),
    ("유병재 진지", "misc"),
    ("유병재 짤", "misc"),
    ("유병재 짤방", "misc"),
    ("유병재 리액션", "misc"),
    # === ROUND 2: 색 다양성 보강 (vivid red/blue/yellow, 흑백 양극) ===
    # red 톤
    ("유병재 빨간옷", "misc"),
    ("유병재 빨간색", "misc"),
    ("유병재 레드카펫 시상식", "photoshoot"),
    ("유병재 무대 빨간조명", "snl"),
    # blue 톤
    ("유병재 파란옷", "misc"),
    ("유병재 청색", "misc"),
    ("유병재 데님", "misc"),
    ("유병재 청바지", "misc"),
    # yellow / orange / green
    ("유병재 노란옷", "misc"),
    ("유병재 후드티", "misc"),
    ("유병재 초록", "misc"),
    # 흰 / 검 (밝기 양극)
    ("유병재 흰옷", "misc"),
    ("유병재 화이트 스튜디오", "photoshoot"),
    ("유병재 검은옷", "misc"),
    ("유병재 검정 정장", "photoshoot"),
    # 환경 색감
    ("유병재 콘서트 무대", "snl"),
    ("유병재 야간 촬영", "youtube"),
    ("유병재 노을", "youtube"),
    ("유병재 음식 먹방", "youtube"),
    ("유병재 카페 인테리어", "youtube"),
    ("유병재 자연 풍경", "youtube"),
    ("유병재 네온", "snl"),
    # === ROUND 3: 분장/코스튬/조명 다양화 (정체성 필터 후 색 풀 확보) ===
    # SNL 캐릭터 분장 (얼굴색/조명 다양)
    ("유병재 SNL 분장 캐릭터", "snl"),
    ("유병재 SNL 코스프레", "snl"),
    ("유병재 SNL 여장", "snl"),
    ("유병재 SNL 가발", "snl"),
    ("유병재 SNL 메이크업", "snl"),
    ("유병재 SNL 한복", "snl"),
    ("유병재 SNL 교복", "snl"),
    ("유병재 SNL 의사", "snl"),
    ("유병재 SNL 군인", "snl"),
    ("유병재 분장 콩트", "snl"),
    # 영화·드라마 (필터·톤 다양)
    ("유병재 소셜포비아", "photoshoot"),
    ("유병재 영화 출연", "photoshoot"),
    ("유병재 드라마 출연", "photoshoot"),
    ("유병재 단편영화", "photoshoot"),
    # 초록 톤 (가장 부족)
    ("유병재 초록색 옷", "misc"),
    ("유병재 그린스크린", "snl"),
    ("유병재 잔디밭", "youtube"),
    ("유병재 캠핑", "youtube"),
    ("유병재 골프", "youtube"),
    ("유병재 등산", "youtube"),
    # 파랑 톤 보강
    ("유병재 바다", "youtube"),
    ("유병재 수영장", "youtube"),
    ("유병재 하늘", "youtube"),
    ("유병재 푸른 조명", "snl"),
    # 매우 밝은 (L>180) 보강
    ("유병재 화이트 화보", "photoshoot"),
    ("유병재 밝은 스튜디오", "photoshoot"),
    ("유병재 흰색 배경", "photoshoot"),
    # 강한 색 조명·LED
    ("유병재 보라색 조명", "snl"),
    ("유병재 핑크 조명", "snl"),
    ("유병재 LED 무대", "snl"),
    ("유병재 클럽 무대", "snl"),
    # 컨셉 화보 (잡지)
    ("유병재 GQ", "photoshoot"),
    ("유병재 에스콰이어", "photoshoot"),
    ("유병재 Dazed", "photoshoot"),
    ("유병재 컨셉 화보", "photoshoot"),
    # === ROUND 4: 미사용 한글 + 영문 키워드 (목표 4000 원본) ===
    # 구체적 SNL 콩트
    ("유병재 따따부따", "snl"),
    ("유병재 클럽MK", "snl"),
    ("유병재 위켄드 업데이트", "snl"),
    ("유병재 SNL 시즌1", "snl"),
    ("유병재 SNL 시즌9", "snl"),
    ("유병재 SNL 다시보기", "snl"),
    # 광고/CF
    ("유병재 CF", "photoshoot"),
    ("유병재 광고 출연", "photoshoot"),
    ("유병재 모델", "photoshoot"),
    ("유병재 브랜드", "photoshoot"),
    # 라디오/팟캐스트
    ("유병재의 가짜사나이", "youtube"),
    ("유병재 두시탈출", "youtube"),
    ("유병재 야밤의 음악편지", "youtube"),
    ("유병재 컬투쇼", "youtube"),
    ("유병재 팟캐스트", "youtube"),
    # 토크쇼/예능
    ("유병재 라스", "youtube"),
    ("유병재 놀면뭐하니", "youtube"),
    ("유병재 아는형님", "youtube"),
    ("유병재 라디오스타 출연", "youtube"),
    ("유병재 맛있는녀석들", "youtube"),
    ("유병재 동상이몽", "youtube"),
    # 행사/대담/팬미팅
    ("유병재 강연", "misc"),
    ("유병재 북콘서트", "misc"),
    ("유병재 저자사인회", "misc"),
    ("유병재 토크 콘서트", "misc"),
    ("유병재 팬미팅", "misc"),
    # 셀카/사진
    ("유병재 셀카", "misc"),
    ("유병재 사진집", "photoshoot"),
    ("유병재 사진전", "photoshoot"),
    ("유병재 폴라로이드", "misc"),
    # 영문 키워드
    ("Yoo Byung-jae interview", "youtube"),
    ("Yoo Byung Jae stand up", "youtube"),
    ("Korean comedian Yoo Byung-jae", "misc"),
    ("Yoo Byungjae Netflix", "photoshoot"),
    ("Yoo Byungjae comedy special", "youtube"),
    # 출연 작품 추가
    ("유병재 영화 단편", "photoshoot"),
    ("유병재 웹드라마", "photoshoot"),
    ("유병재 시트콤", "photoshoot"),
    ("유병재 단막극", "photoshoot"),
    # === ROUND 5: vivid 색감 부족분 보강 + 다양성 ===
    # 빨강 (vivid)
    ("유병재 빨간 LED 무대", "snl"),
    ("유병재 빨간 조명 SNL", "snl"),
    ("유병재 빨간 정장", "photoshoot"),
    ("유병재 빨간 셔츠", "misc"),
    ("유병재 빨간 모자", "misc"),
    ("유병재 빨간 점퍼", "misc"),
    ("유병재 크리스마스 빨간", "misc"),
    ("유병재 빨간 자켓", "misc"),
    # 노랑 (vivid)
    ("유병재 노란 무대조명", "snl"),
    ("유병재 노란 점퍼", "misc"),
    ("유병재 노란 셔츠", "misc"),
    ("유병재 노란 자켓", "misc"),
    ("유병재 황금 조명", "snl"),
    ("유병재 노란 배경", "photoshoot"),
    # 파랑 (vivid)
    ("유병재 파란 무대조명", "snl"),
    ("유병재 파란 정장", "photoshoot"),
    ("유병재 파란 셔츠", "misc"),
    ("유병재 파란 LED 무대", "snl"),
    ("유병재 파란 자켓", "misc"),
    ("유병재 청색 무대", "snl"),
    ("유병재 청자켓", "misc"),
    # 초록 (vivid)
    ("유병재 초록 무대조명", "snl"),
    ("유병재 그린 LED", "snl"),
    ("유병재 초록 셔츠", "misc"),
    ("유병재 초록 자켓", "misc"),
    ("유병재 잔디 배경", "youtube"),
    # 흰색 (vivid)
    ("유병재 흰색 정장", "photoshoot"),
    ("유병재 흰 셔츠", "misc"),
    ("유병재 흰 코트", "misc"),
    ("유병재 흰 배경 화보 풀샷", "photoshoot"),
    ("유병재 화이트 톤 스튜디오", "photoshoot"),
    ("유병재 흰 후드", "misc"),
    # 핑크 / 보라 / 주황
    ("유병재 핑크 셔츠", "misc"),
    ("유병재 보라 무대조명", "snl"),
    ("유병재 보라 자켓", "misc"),
    ("유병재 주황 옷", "misc"),
    ("유병재 주황 무대조명", "snl"),
    ("유병재 핑크 무대조명", "snl"),
    # 다양 환경 (다양성 ↑)
    ("유병재 댄스 무대", "youtube"),
    ("유병재 헬스장", "youtube"),
    ("유병재 캠핑장", "youtube"),
    ("유병재 호텔", "youtube"),
    ("유병재 비행기", "youtube"),
    ("유병재 식당", "youtube"),
    ("유병재 빙수", "youtube"),
    ("유병재 운동", "youtube"),
    ("유병재 산책", "youtube"),
    ("유병재 길거리", "youtube"),
]

ENGINES = ("bing", "duckduckgo")


def _perceptual_hash(path: Path) -> str | None:
    try:
        import imagehash

        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def _crawl_keyword(engine: str, keyword: str, target_dir: Path, max_num: int) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
    except ImportError:
        print("[collect] icrawler 미설치")
        return 0

    if engine == "bing":
        cls = BingImageCrawler
    elif engine == "google":
        cls = GoogleImageCrawler
    elif engine == "duckduckgo":
        try:
            from icrawler.builtin import DuckDuckGoImageCrawler  # type: ignore

            cls = DuckDuckGoImageCrawler
        except ImportError:
            return 0
    else:
        return 0
    # Use a subdir per (engine, keyword hash) to avoid filename collision overwriting
    safe_kw = keyword.replace(" ", "_").replace("/", "_")
    work_dir = target_dir / f"_{engine}_{safe_kw}"
    work_dir.mkdir(parents=True, exist_ok=True)

    crawler = cls(
        downloader_threads=4,
        storage={"root_dir": str(work_dir)},
        log_level=40,
    )
    before = len(list(work_dir.iterdir()))
    try:
        crawler.crawl(
            keyword=keyword,
            max_num=max_num,
            min_size=(256, 256),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[collect] {engine} '{keyword}' 실패: {e}")
        return 0
    new_files = []
    for f in work_dir.iterdir():
        if f.is_file():
            new_files.append(f)
    # Move into target_dir with unique names
    moved = 0
    for f in new_files:
        target_name = f"{engine}_{safe_kw[:30]}_{f.name}"
        dest = target_dir / target_name
        if dest.exists():
            continue
        try:
            f.rename(dest)
            moved += 1
        except OSError:
            pass
    # Cleanup workdir
    try:
        for leftover in work_dir.iterdir():
            leftover.unlink()
        work_dir.rmdir()
    except OSError:
        pass
    return moved


def _drop_duplicates() -> int:
    """전체 tiles/ 스캔 → perceptual hash 같으면 후순위 파일 삭제."""
    seen: dict[str, Path] = {}
    removed = 0
    for p in iter_tile_paths():
        h = _perceptual_hash(p)
        if not h:
            continue
        if h in seen:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        else:
            seen[h] = p
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="유병재 이미지 수집기")
    parser.add_argument("--target", type=int, default=500, help="목표 총 장수")
    parser.add_argument("--per-keyword", type=int, default=0, help="키워드별 장수 (0=목표/키워드*1.5)")
    parser.add_argument("--engines", nargs="+", default=list(ENGINES))
    parser.add_argument("--from-index", type=int, default=0, help="KEYWORDS 슬라이스 시작 인덱스 (이미 수집한 키워드 건너뛰기)")
    args = parser.parse_args()

    ensure_dirs()

    keywords = KEYWORDS[args.from_index :]
    per = args.per_keyword or max(20, int(args.target / max(1, len(keywords)) * 1.5) + 5)
    total_attempts = len(keywords) * len(args.engines) * per
    print(
        f"[collect] {len(keywords)}키워드 × {len(args.engines)}엔진 × {per}장 = 시도 최대 {total_attempts}장 (from-index={args.from_index})"
    )

    log_path = OUTPUT_DIR / "_collect_log.txt"
    log_lines = []
    by_source: Counter[str] = Counter()
    new_total = 0

    for engine in args.engines:
        for kw, source in keywords:
            target_dir = TILES_DIR / source
            try:
                n = _crawl_keyword(engine, kw, target_dir, per)
            except Exception as e:  # noqa: BLE001
                log_lines.append(f"FAIL\t{engine}\t{kw}\t{e}")
                continue
            new_total += n
            by_source[source] += n
            print(f"[collect] [{engine}] '{kw}' → {source}/ +{n}장")

    print("\n[collect] perceptual hash로 중복 제거 중…")
    pre = sum(1 for _ in iter_tile_paths())
    removed = _drop_duplicates()
    post = sum(1 for _ in iter_tile_paths())
    print(f"[collect] 중복 {removed}장 제거 (총 {pre}→{post}장)")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\n[collect] 신규 {new_total}장 수집, 소스별 분포:")
    for src, n in by_source.most_common():
        print(f"  - {src}: {n}장 (수집)")
    print(f"[collect] 현재 총 {post}장 (중복 제거 후). 다음: /classify-tiles")


if __name__ == "__main__":
    main()
