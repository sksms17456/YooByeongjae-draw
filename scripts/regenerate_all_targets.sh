#!/bin/bash
# 23장 전 타겟 80x80 모자이크 재생성 + 채점
set -e
cd "$(dirname "$0")/.."

scores_csv="output/_scores_v8_80x80.csv"
echo "target,grid,blend,total,grade,ssim,perceptual,color,diversity" > "$scores_csv"

count=0
total_score=0
for target in targets/curated_*.jpg; do
  name=$(basename "$target" .jpg)
  count=$((count+1))
  echo "[$count/23] $name"

  uv run python src/mosaic.py --target "$target" 2>&1 | tail -1 > /dev/null

  layout=$(ls -t output/mosaic_${name}_*_80x80_layout.json 2>/dev/null | head -1)
  if [ -z "$layout" ]; then
    echo "  ! layout 없음"
    continue
  fi
  uv run python src/score.py --layout "$layout" --target "$target" 2>&1 | grep -E "TOTAL" | tail -1
done

echo ""
echo "완료. layout/score는 output/ 에 저장됨"
