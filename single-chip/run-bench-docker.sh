#!/usr/bin/env bash
#
# Docker で起動済みの vLLM サーバ (gemma4-12b-tpu / gemma4-12b-gpu) に対して
# 単チップ実験のスイープ (3 プロファイル × 7 レート = 21 ケース) を実行する。
#
# プロファイルは PLAN.md 準拠 (31B 版とは異なるので注意):
#   short  : in 128  / out 128   (チャット・対話)
#   medium : in 1024 / out 512   (RAG・要約)
#   long   : in 4096 / out 1024  (長文生成・コード)
#
# 使い方:
#   CONTAINER=gemma4-12b-tpu bash run-bench-docker.sh
#
# 環境変数:
#   CONTAINER   サーバが動いているコンテナ名 (必須)
#   MODEL       (default: google/gemma-4-12B-it)
#   HOST/PORT   (default: 127.0.0.1:8000)
#   NUM_PROMPTS (default: 500)  単チップは吸収能力が低いので 31B 版の 1000 から半減。
#                               両プラットフォームで同値にすること。
#   GCS_BUCKET  結果アップロード先 (任意)

set -euo pipefail

CONTAINER="${CONTAINER:?CONTAINER 必須 (例: gemma4-12b-tpu)}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"

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

# 再現性のためのメタデータ保存
sudo docker inspect "$CONTAINER" \
  --format '{{.Config.Image}} {{index .Config.Labels "org.opencontainers.image.revision"}}' \
  > "$OUT_DIR/container-image.txt" || true
sudo docker image inspect "$(sudo docker inspect $CONTAINER --format '{{.Config.Image}}')" \
  --format '{{json .RepoDigests}}' > "$OUT_DIR/image-digest.txt" || true
# サーバ起動引数 (max-num-seqs 等の採用値) も保存 — 記事に載せる
sudo docker inspect "$CONTAINER" --format '{{json .Config.Cmd}}' \
  > "$OUT_DIR/server-args.json" || true
sudo docker logs "$CONTAINER" 2>&1 | grep -iE "kv cache|dtype|memory" | head -30 \
  > "$OUT_DIR/server-memory-lines.log" || true

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
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$rate" \
    --save-result \
    --result-dir /tmp \
    --result-filename "$result_file" \
    > "$OUT_DIR/$log_file" 2>&1 || echo "WARN: ${name} rate=${rate} failed (see log)"

  # 結果をコンテナからホストへコピー
  sudo docker cp "$CONTAINER:/tmp/$result_file" "$OUT_DIR/$result_file" 2>/dev/null || true
}

# warmup (結果は捨てる)
echo "=== warmup ==="
sudo docker exec "$CONTAINER" vllm bench serve \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 128 \
  --num-prompts 50 --request-rate 4 \
  > "$OUT_DIR/warmup.log" 2>&1 || true

# sweep (PLAN.md のプロファイル)
PROFILES=("128:128:short" "1024:512:medium" "4096:1024:long")
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
echo "集計: bash ../summarize-bench.sh $OUT_DIR (jq 必要)"

if [[ -n "${GCS_BUCKET:-}" ]]; then
  echo "uploading to gs://$GCS_BUCKET/bench-results/"
  gsutil -m cp -r "$OUT_DIR" "gs://$GCS_BUCKET/bench-results/"
fi
