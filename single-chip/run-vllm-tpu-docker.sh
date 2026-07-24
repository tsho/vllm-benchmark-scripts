#!/usr/bin/env bash
#
# TPU VM (v6e-1) 内で vLLM TPU の Docker サーバを起動する。単チップ (TP=1) 用。
#
# 使い方 (TPU VM 上):
#   HF_TOKEN=hf_xxx bash ~/run-vllm-tpu-docker.sh
#
# 環境変数で上書き可:
#   IMAGE        (default: vllm/vllm-tpu:v0.25.0)  # TPU/GPU で vLLM バージョンを v0.25.0 に統一
#   MODEL        (default: Qwen/Qwen3-8B)
#                Gemma 4 12B は v6e-1 で KV cache が 13,056 tok まで枯渇し実験不成立、
#                gemma-4-E4B-it は tpu-inference の mesh 初期化 (1,4) で単チップ起動不可。
#                Qwen3-8B は JAX ネイティブ実装があり KV ~98k tok を確保できる (2026-07-24 実測)。
#   MAX_LEN      (default: 8192)  long プロファイル 4096+1024 が収まる最小の2べき
#   MAX_NUM_SEQS (default: 32)    Qwen3-8B + KV ~98k tok なら 32 で short/medium は KV に余裕。
#                                 long (5,120 tok/req) は KV 律速 ~19 並列で頭打ち (それ自体が観測値)。
#                                 採用値は記事に載せるので必ず記録すること。
#   MAX_BATCHED_TOKENS (default: 4096)
#                                 マルチモーダル対応モデル + --disable_chunked_mm_input の組合せは
#                                 max_tokens_per_mm_item (Gemma 4 系は 2496) 以上が必須。
#                                 デフォルトの 2048 だと起動時に ValueError で落ちる。
#                                 GPU 側と同値に揃えること (公平性)。
#   PORT         (default: 8000)
#
# KV cache dtype は指定しない (auto)。起動ログに出る実際の dtype を記録する:
#   sudo docker logs qwen3-8b-tpu 2>&1 | grep -i "kv cache"
# (v6e では fp8_e5m2 が自動選択される)

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN 必須}"

IMAGE="${IMAGE:-vllm/vllm-tpu:v0.25.0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TP=1
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
PORT="${PORT:-8000}"
NAME="${NAME:-qwen3-8b-tpu}"

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
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --disable_chunked_mm_input \
      --host 0.0.0.0 --port "$PORT"

cat <<EOF

Container '$NAME' started (TP=$TP, max-model-len=$MAX_LEN, max-num-seqs=$MAX_NUM_SEQS).
ログ確認:
  sudo docker logs -f $NAME
KV cache の実測値を記録 (記事用):
  sudo docker logs $NAME 2>&1 | grep -iE "kv cache|memory"
EOF
