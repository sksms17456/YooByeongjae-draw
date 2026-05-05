# byunjae-leeds — 유병재 포토 모자이크

웹에서 보는 인터랙티브 포토 모자이크. 상단의 타겟 후보를 클릭하면 타일들이 FLIP 애니메이션으로 재배치되어 타겟을 구성한다. 줌인하면 개별 타일(유병재의 다른 사진들)이 보인다.

## 환경

- Python 3.14 (`uv` 기반 가상환경)
- 모든 스크립트는 `argparse` + `--help` 지원
- 정적 사이트 배포: `scripts/build_static.py` → `public/` → Vercel

## 디렉터리 구조

```
src/         Python 파이프라인 (collect/curate/classify/preprocess/mosaic/score/improve/server)
web/         vanilla JS 프론트엔드 (index.html, app.js, style.css)
tiles/       타일 이미지 (256×256 원본). 소스별 서브폴더(snl/youtube/photoshoot/misc)
tiles/_thumbs/         50×50 썸네일 (초기 로드용)
tiles/_rejected/       정체성 필터 탈락분 (no_face/not_byunjae/tiny_byunjae/byunjae_not_main)
tiles/_target_candidates/  타겟 큐레이션 후보 (build에서 자동 제외)
targets/     타겟 후보 (사용자 큐레이션, 커밋). 23장
targets/_rejected_targets/  중복/제거된 타겟 (참고용 격리)
index/       tiles_index.csv (수치형) + tiles_meta.json (카테고리형). 재생성 가능 → gitignore
output/      *_layout.json (1차 산출), *_score.json, *_improve_log.json
output/_score_render/    채점용 1024×1024 PNG (사용자 비노출)
public/      Vercel 배포용 정적 빌드 (build_static.py로 생성, ~93MB)
scripts/     보조 스크립트 (build_static, regenerate_all_targets, pick_costume_targets 등)
```

`src/utils.py iter_tile_paths()`는 `_`로 시작하는 서브폴더 모두 제외 — 신규 underscore 폴더 추가해도 안전.

## 인덱스 스키마

### `index/tiles_index.csv` (수치형, 빠른 매칭)

```
path, avg_l, avg_a, avg_b,
face_l, face_a, face_b,                    # 얼굴 영역 평균 LAB (없으면 -1)
q0_l..q0_b, q1_l..q1_b, q2_l..q2_b, q3_l..q3_b,  # 2x2 patch LAB (4분할)
dominant_hue, brightness, saturation, sharpness,
has_face, face_x, face_y, face_w, face_h
```

`reindex_face_lab.py` 실행 시 `avg_l/a/b`가 얼굴 영역 평균색으로 교체됨.

### `index/tiles_meta.json` (카테고리형 + 신뢰도)

```json
{
  "tiles/snl/0042.jpg": {
    "content_type": "face",
    "content_conf": 0.87,
    "expression": "smile",
    "expression_conf": 0.71,
    "source": "snl",
    "aesthetic": 6.2
  }
}
```

CLIP 분류 신뢰도 < 0.4 → `content_type = unknown` (매칭 시 모든 영역 동일 가중치).

## 핵심 파라미터 (디폴트, v8)

- 격자: **80×80** (6,400 셀) — 90×90은 1024÷90 비정수로 SSIM 손해, 100×100은 풀 부족
- 타일 디스크 해상도: **256×256** 원본 + **50×50** 썸네일
- 블렌딩: **0.75** (`output = target*0.75 + tile*0.25`) — 채점 천장 도달
- 매칭: **LAB ΔE chroma-weighted (ab×3) + 2x2 patch + top-K 80**
- 페널티: usage 5.0 (타일 재사용 분산), neighbor 2.0
- **REGION_WEIGHTS 비활성**, face_lab blend 비활성 (순수 색 매칭)
- 채점용 렌더 PNG: 1024×1024
- 현재 23장 평균: **93.40 (A)** — 목표 70(B)·권장 85(A) 모두 초과 달성
- 개선 루프 최대 반복: 5회

격자 vs 점수 (solo_01 기준, 풀 3,520):

| 격자 | Best 점수 | 비고 |
|---|---|---|
| 60×60 | 86.35 | 채점 sweet spot 아님 |
| **80×80** ⭐ | **92.14** | 현재 디폴트 |
| 90×90 | 91.41 | U자형 local minimum |
| 100×100 | 91.79 | 풀 3,520 부족, diversity 손실 |

## 산출물 정책

- **`*_layout.json`이 1차 산출물** — 웹 뷰어가 읽음
- 채점용 1024 PNG는 `output/_score_render/`에 사용자 비노출
- **인쇄/다운로드용 고해상도 PNG는 생성하지 않음** (웹 전용)

## 명명 규약

- 출력: `mosaic_{target_basename}_{YYYYMMDD-HHMMSS}_{grid}x{grid}_layout.json`
- 부산물: 동일 basename으로 `*_score.json`, `*_improve_log.json`

## Vercel 배포

```bash
uv run python scripts/build_static.py    # public/ 생성 (~93MB)
cd public && vercel --prod                # CLI 배포
# 또는 GitHub push → vercel.com에서 import (Root Directory: public)
```

빌드 시 사용된 타일만 `public/tiles_thumb/`, `public/tiles_full/`에 복사 — 전체 풀 X.

웹 프론트엔드(`web/app.js`)는 정적/동적 양쪽 호환:
- 정적 우선: `/data/targets.json`, `/data/layouts/{stem}.json`
- 폴백: FastAPI `/api/targets`, `/api/layouts/{stem}` (로컬 개발)

## 모자이크 생성 전 가드

- `index/tiles_index.csv` 존재 + 행 수 ≥ 100 확인 (없으면 에러)
- `index/tiles_meta.json` 존재 확인 (분류 누락 시 워닝, content_type 가중치 비활성)

## 절대 커밋 금지

`tiles/`, `output/`, `.venv/`, `index/`, `targets/_*/`, `.vercel/`

## collect.py 주의사항

`_drop_duplicates()`의 phash 중복 제거가 **양방향**이라 재실행 시 기존 타일도 같이 삭제될 수 있음. 풀 손실 시 `rescue_color_tiles.py`로 `_rejected/`에서 sim 0.30~0.32 재구제 가능.

## 사용자 커뮤니케이션

사용자는 시맨틱 웹/PKG 비전공자. 알고리즘·메트릭 설명 시 비전공자 친화 + 구체적 예시 첨부.
