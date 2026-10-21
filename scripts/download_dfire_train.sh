#!/usr/bin/env bash
# D-Fire 학습 분할 (LoRA 학습용, D-25). 미러: HF badsaarow/d-fire
set -euo pipefail
DST="$(cd "$(dirname "$0")/.." && pwd)/data/dfire"
mkdir -p "$DST"
for i in 0 1 2 3 4 5 6 7 8; do
  f="train-0000${i}-of-00009.parquet"
  [ -s "$DST/$f" ] || curl -sL --retry 3 -o "$DST/$f" "https://huggingface.co/datasets/badsaarow/d-fire/resolve/main/data/$f"
done
ls -la "$DST"
