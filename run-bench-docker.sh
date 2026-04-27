#!/usr/bin/env bash
#
# Docker で起動済みの vLLM サーバ (gemma4-gpu / gemma4-tpu) に対して
# vllm bench serve をスイープ実行するクライアントスクリプト。
#
# 使い方:
#   CONTAINER=gemma4-tpu MODEL=google/gemma-4-31B-it bash run-bench-docker.sh
#
# 環境変数:
#   CONTAINER  サーバが動いているコンテナ名 (必須: gemma4-gpu / gemma4-tpu)
#   MODEL      ベンチで指定するモデル ID
#   HOST/PORT  サーバの host/port (default: 127.0.0.1:8000)
#   GCS_BUCKET 結果アップロード先 (任意)

set -euo pipefail

CONTAINER="${CONTAINER:?CONTAINER 必須 (例: gemma4-tpu)}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if ! sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "ERROR: container '$CONTAINER' が起動していません" >&2
  exit 1
fi

# サーバ ready 確認
if ! curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: http://${HOST}:${PORT}/v1/models に到達できません" >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-$HOME/bench-results/$(date +%Y%m%d-%H%M%S)-${CONTAINER}}"
mkdir -p "$OUT_DIR"

echo "Output: $OUT_DIR"

# image 情報も保存
sudo docker inspect "$CONTAINER" \
  --format '{{.Config.Image}} {{index .Config.Labels "org.opencontainers.image.revision"}}' \
  > "$OUT_DIR/container-image.txt" || true
sudo docker image inspect "$(sudo docker inspect $CONTAINER --format '{{.Config.Image}}')" \
  --format '{{json .RepoDigests}}' > "$OUT_DIR/image-digest.txt" || true

# bench を docker exec で実行するヘルパ
bench() {
  local in_len="$1" out_len="$2" rate="$3" name="$4"
  local result_file="result-${name}-rate${rate}.json"
  local log_file="bench-${name}-rate${rate}.log"

  sudo docker exec "$CONTAINER" vllm bench serve \
    --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --dataset-name random \
    --random-input-len "$in_len" \
    --random-output-len "$out_len" \
    --num-prompts 1000 \
    --request-rate "$rate" \
    --save-result \
    --result-dir /tmp \
    --result-filename "$result_file" \
    > "$OUT_DIR/$log_file" 2>&1 || echo "WARN: ${name} rate=${rate} failed (see log)"

  # 結果をコンテナからホストへコピー
  sudo docker cp "$CONTAINER:/tmp/$result_file" "$OUT_DIR/$result_file" 2>/dev/null || true
}

# warmup
echo "=== warmup ==="
sudo docker exec "$CONTAINER" vllm bench serve \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 50 --request-rate 4 \
  > "$OUT_DIR/warmup.log" 2>&1 || true

# sweep
PROFILES=("1024:256:short" "4096:512:medium" "8000:1000:long")
RATES=(1 2 4 8 16 32 inf)

for prof in "${PROFILES[@]}"; do
  IFS=: read -r in_len out_len name <<< "$prof"
  for rate in "${RATES[@]}"; do
    echo "=== profile=$name rate=$rate ==="
    bench "$in_len" "$out_len" "$rate" "$name"
  done
done

echo
echo "Results: $OUT_DIR"

if [[ -n "${GCS_BUCKET:-}" ]]; then
  echo "uploading to gs://$GCS_BUCKET/bench-results/"
  gsutil -m cp -r "$OUT_DIR" "gs://$GCS_BUCKET/bench-results/"
fi
