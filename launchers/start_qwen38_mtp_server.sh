#!/usr/bin/env bash
set -euo pipefail

mtp_port="${1:-18080}"

exec /home/admin/llama.cpp-main-20260810/build/bin/llama-server \
  --model /usr/share/ollama/.ollama/models/blobs/sha256-bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372 \
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
