# Vercel 배포

## 빠른 배포

```bash
# 1. 정적 빌드 (public/ 생성, ~80MB)
uv run python scripts/build_static.py

# 2. 로컬 미리보기 (선택)
cd public && uv run python -m http.server 9999
# → http://localhost:9999

# 3. Vercel 배포
cd public
vercel --prod
```

`vercel` CLI 첫 실행 시 로그인이 필요합니다. 새 프로젝트를 생성할지 물어보면 Yes.

## 배포되는 자산

- `index.html`, `app.js`, `style.css`, `favicon.png` — 웹 프론트엔드
- `data/targets.json` — 타겟 목록
- `data/layouts/{stem}.json` — 각 타겟의 모자이크 레이아웃
- `targets/*.jpg` — 23장 타겟 원본
- `tiles_thumb/{src}/*.jpg` — 50×50 썸네일 (~17MB)
- `tiles_full/{src}/*.jpg` — 256×256 원본 (~60MB, 줌 인 시 사용)

## 재배포

레이아웃을 다시 생성하거나 타겟을 추가하면 빌드 → 배포 순으로:

```bash
uv run python scripts/build_static.py
cd public && vercel --prod
```

## 비용

Vercel Hobby 무료 플랜 범위 내:
- 100GB 대역폭/월
- 정적 배포 무제한

## 동적 모드 vs 정적 모드

`web/app.js`는 양쪽 호환:
- 정적(`/data/targets.json` 우선) — Vercel용
- 폴백 FastAPI(`/api/targets`) — 로컬 개발 시
