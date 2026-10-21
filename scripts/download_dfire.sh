#!/usr/bin/env bash
# D-Fire 테스트 분할(4,306장)만 다운로드.
# 원본: https://github.com/gaiasd/DFireDataset (컬렉션 CC0, 일부 이미지는 원 출처 약관을 따름 → 재배포하지 않음)
# 미러: HF badsaarow/d-fire (parquet: image, label=YOLO 텍스트, filename)
set -euo pipefail
DST="$(cd "$(dirname "$0")/.." && pwd)/data/dfire"
mkdir -p "$DST"
for i in 0 1 2; do
  f="test-0000${i}-of-00003.parquet"
  [ -s "$DST/$f" ] || curl -sL --retry 3 -o "$DST/$f" "https://huggingface.co/datasets/badsaarow/d-fire/resolve/main/data/$f"
done
ls -la "$DST"
