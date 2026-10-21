#!/usr/bin/env bash
# 2단계(FP16 기준선, 256M) + 3단계(bitsandbytes) 전체 실행. 결과: results/<config>/
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}  # 예: PY=/path/to/venv/bin/python bash scripts/run_all.sh
for args in "--config fp16" "--config fp16 --full-frame" "--config fp16_256m" "--config bnb8" "--config bnb4" "--config bnb8_all"; do
  echo "=== $args $(date +%T)"
  $PY scripts/run_eval.py $args 2>&1 | grep -v -i -E "warn|pad_token_id|Loading weights"
done
echo "=== done $(date +%T)"
