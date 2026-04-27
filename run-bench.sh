#!/usr/bin/env bash
#
# vllm serve をバックグラウンド起動し、vllm bench serve でスイープを実行する雛形。
#
# 使い方 (GPU VM):  TP=2 bash run-bench.sh
# 使い方 (TPU v6e-4): TP=4 bash run-bench.sh
#
# 結果は ./bench-results/<timestamp>/ に保存。最後に GCS にアップロードしたい場合は
# GCS_BUCKET 環境変数を指定:
#   TP=2 GCS_BUCKET=my-bucket bash run-bench.sh

set -euo pipefail

# venv の自動アクティベート (setup-vllm-{gpu,tpu}.sh が作る ~/.venvs/vllm を期待)
VENV="${VENV:-$HOME/.venvs/vllm}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: 'vllm' コマンドが見つかりません。" >&2
  echo "  setup-vllm-{gpu,tpu}.sh で venv ($VENV) を作成済みか、" >&2
  echo "  または 'source $VENV/bin/activate' してから再実行してください。" >&2
  exit 1
fi

MODEL="${MODEL:-google/gemma-4-31B-it}"
TP="${TP:?tensor-parallel-size 必須 (GPU=2, TPU v6e-4=4)}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
KV_DTYPE="${KV_DTYPE:-fp8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

OUT_DIR="${OUT_DIR:-./bench-results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

# サーバ起動 (バックグラウンド)
SERVE_ARGS=(
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype "$KV_DTYPE"
  --host "$HOST" --port "$PORT"
)
echo "starting: vllm serve $MODEL ${SERVE_ARGS[*]}"
vllm serve "$MODEL" "${SERVE_ARGS[@]}" > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# レディ待ち (最大 10 分)
echo "waiting for server to be ready..."
for i in $(seq 1 120); do
  if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
    echo "server ready (after ${i}x5s)"
    break
  fi
  sleep 5
  if [[ $i -eq 120 ]]; then
    echo "ERROR: server not ready in 10min. See $OUT_DIR/server.log" >&2
    exit 1
  fi
done

# ウォームアップ (結果は捨てる)
echo "=== warmup ==="
vllm bench serve \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 50 --request-rate 4 \
  > "$OUT_DIR/warmup.log" 2>&1 || true

# スイープ
PROFILES=("1024:256:short" "4096:512:medium" "8000:1000:long")
RATES=(1 2 4 8 16 32 inf)

for prof in "${PROFILES[@]}"; do
  IFS=: read -r in_len out_len name <<< "$prof"
  for rate in "${RATES[@]}"; do
    echo "=== profile=$name rate=$rate ==="
    bench_args=(
      --model "$MODEL" --host "$HOST" --port "$PORT"
      --dataset-name random
      --random-input-len "$in_len"
      --random-output-len "$out_len"
      --num-prompts 1000
      --request-rate "$rate"
      --save-result
      --result-dir "$OUT_DIR"
      --result-filename "result-${name}-rate${rate}.json"
    )
    vllm bench serve "${bench_args[@]}" \
      > "$OUT_DIR/bench-${name}-rate${rate}.log" 2>&1 \
      || echo "WARN: ${name} rate=${rate} failed (see log)"
  done
done

echo "Results: $OUT_DIR"

# GCS アップロード (任意)
if [[ -n "${GCS_BUCKET:-}" ]]; then
  echo "uploading to gs://$GCS_BUCKET/bench-results/"
  gsutil -m cp -r "$OUT_DIR" "gs://$GCS_BUCKET/bench-results/"
fi
