#!/usr/bin/env bash
#
# TPU VM 内で vLLM TPU の Docker サーバを起動するラッパー。
# 公式 Gemma 4 レシピ準拠 (https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)。
#
# 使い方 (TPU VM 上):
#   HF_TOKEN=hf_xxx bash ~/run-vllm-tpu-docker.sh
#
# 環境変数で上書き可:
#   IMAGE   (default: vllm/vllm-tpu:gemma4)
#   MODEL   (default: google/gemma-4-31B-it)
#   TP      (default: 4 = v6e-4)
#   MAX_LEN (default: 16384)
#   PORT    (default: 8000)

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN 必須}"

IMAGE="${IMAGE:-vllm/vllm-tpu:gemma4}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
TP="${TP:-4}"
MAX_LEN="${MAX_LEN:-16384}"
PORT="${PORT:-8000}"
NAME="${NAME:-gemma4-tpu}"

# 既存コンテナを片付け
sudo docker rm -f "$NAME" 2>/dev/null || true

sudo docker run -itd --name "$NAME" \
  --privileged --network host --shm-size 16G \
  -v /dev/shm:/dev/shm \
  -e HF_TOKEN="$HF_TOKEN" \
  "$IMAGE" \
    vllm serve "$MODEL" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_LEN" \
      --disable_chunked_mm_input \
      --host 0.0.0.0 --port "$PORT"

echo
echo "Container '$NAME' started. Tail logs with:"
echo "  sudo docker logs -f $NAME"
