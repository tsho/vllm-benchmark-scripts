#!/usr/bin/env bash
#
# GPU VM (A100 80GB × 1) 内で vLLM の Docker サーバを起動する。単チップ (TP=1) 用。
#
# 使い方 (GPU VM 上):
#   HF_TOKEN=hf_xxx bash ~/run-vllm-gpu-docker.sh
#
# 環境変数で上書き可:
#   IMAGE        (default: vllm/vllm-openai:v0.25.0)
#   MODEL        (default: google/gemma-4-12B-it) # HF ID は要確認
#   MAX_LEN      (default: 8192)  TPU 側と揃える (公平性)
#   MAX_NUM_SEQS (default: 16)    基本系列は TPU 側の採用値と同値に揃える。
#                                 「A100 の余裕を開放した系列」を取るときだけ 128 等に上げ、
#                                 別系列として記録する。
#   GPU_MEM_UTIL (default: 0.90)
#   PORT         (default: 8000)
#
# KV cache dtype は指定しない (auto)。起動ログに出る実際の dtype を記録し、
# TPU 側と異なる場合は記事に明記する。

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN 必須}"

IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.0}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
TP=1
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
PORT="${PORT:-8000}"
NAME="${NAME:-gemma4-12b-gpu}"

# 既存コンテナを片付け
sudo docker rm -f "$NAME" 2>/dev/null || true

sudo docker run -itd --name "$NAME" \
  --gpus all --network host --shm-size 16G \
  -v /dev/shm:/dev/shm \
  -e HF_TOKEN="$HF_TOKEN" \
  "$IMAGE" \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --host 0.0.0.0 --port "$PORT"

cat <<EOF

Container '$NAME' started (TP=$TP, max-model-len=$MAX_LEN, max-num-seqs=$MAX_NUM_SEQS, gpu-mem-util=$GPU_MEM_UTIL).
ログ確認:
  sudo docker logs -f $NAME
KV cache の実測値を記録 (記事用):
  sudo docker logs $NAME 2>&1 | grep -iE "kv cache|memory"
EOF
