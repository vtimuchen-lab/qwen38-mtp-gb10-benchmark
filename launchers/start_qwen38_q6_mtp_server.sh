#!/usr/bin/env bash
set -euo pipefail

mtp_port="${1:-18081}"

exec /home/admin/llama.cpp-main-20260810/build/bin/llama-server \
  --model /usr/share/ollama/.ollama/models/blobs/sha256-739202186fd9389bb58497c58b56c8a0d4253d99d20131e6a0427e363e678fc8 \
  --alias qwen3.8-MTP:27b-q6 \
  --host 127.0.0.1 \
  --port "${mtp_port}" \
  --no-webui \
  --offline \
  --ctx-size 65536 \
  --parallel 1 \
  --gpu-layers all \
  --fit off \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  --reasoning-format deepseek \
  --spec-type draft-mtp \
  --spec-draft-n-max 10 \
  --spec-draft-ngl all
