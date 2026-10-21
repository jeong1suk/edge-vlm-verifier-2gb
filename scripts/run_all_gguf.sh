#!/usr/bin/env bash
# 3단계(GGUF) 전체 실행. 결과: results/<config>/
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}  # 예: PY=/path/to/venv/bin/python bash scripts/run_all_gguf.sh
for args in "--config gguf_f16" "--config gguf_q8_0" "--config gguf_q4_k_m" "--config gguf_q4_0" "--config gguf_q8_0_mmq8" "--config gguf_f16 --no-mmap" "--config gguf_q4_0 --no-mmap"; do
  tag=$(echo "$args" | sed -E 's/--config //; s/ --no-mmap/_nommap/')
  if [ -s "results/$tag/summary.json" ]; then echo "=== skip $tag (done)"; continue; fi
  echo "=== $args $(date +%T)"
  $PY scripts/run_gguf.py $args 2>&1 | grep -v "^  "
done
echo "=== done $(date +%T)"
