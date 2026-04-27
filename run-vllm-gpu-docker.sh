#!/usr/bin/env bash
#
# GPU VM 内で vLLM の Docker サーバを起動するラッパー。
# 公式 vLLM image (vllm/vllm-openai) を使う。Gemma 4 公式レシピは GPU 側で
# pip install を案内しているが、Docker のほうが TPU 側と運用を揃えられる。
#
# 使い方 (GPU VM 上):
#   HF_TOKEN=hf_xxx bash ~/run-vllm-gpu-docker.sh
#
# 環境変数で上書き可:
#   IMAGE   (default: vllm/vllm-openai:latest)
#   MODEL   (default: google/gemma-4-31B-it)
#   TP      (default: 2 = 80GB×2 GPU 構成)
#   MAX_LEN (default: 16384)
#   PORT    (default: 8000)
#   GPU_MEM_UTIL (default: 0.90)
#   KV_DTYPE (default: fp8_e5m2)
#     - A100 (Ampere) は fp8_e4m3 (= fp8e4nv) 非対応、fp8_e5m2 のみ。
#     - H100 (Hopper) は両対応。
#     - TPU v6e は自動で fp8_e5m2 を選ぶため、横断で fp8_e5m2 が比較に公平。

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN 必須}"

IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-16384}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
KV_DTYPE="${KV_DTYPE:-fp8_e5m2}"
NAME="${NAME:-gemma4-gpu}"

# 事前チェック
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker が見つかりません" >&2
  exit 1
fi
if ! sudo docker info 2>/dev/null | grep -qi 'runtimes.*nvidia'; then
  echo "WARN: NVIDIA Container Toolkit (nvidia-container-toolkit) が docker に登録されていない可能性があります"
  echo "      DLVM の common-cu129-ubuntu-2404-nvidia-580 では事前導入されているはず"
fi

# HF キャッシュをホスト側に永続化（再起動時の再 DL を回避）
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# 既存コンテナを片付け
sudo docker rm -f "$NAME" 2>/dev/null || true

sudo docker run -itd --name "$NAME" \
  --gpus all --ipc=host --shm-size=16g \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -p "${PORT}:${PORT}" \
  "$IMAGE" \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --host 0.0.0.0 --port "$PORT"

echo
echo "Container '$NAME' started. Tail logs with:"
echo "  sudo docker logs -f $NAME"
