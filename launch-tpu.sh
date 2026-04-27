#!/usr/bin/env bash
#
# Gemma 4 31B 用 TPU VM を作成する雛形。
# v6e (Trillium) と v7x (Ironwood / TPU7x) を切替可能。
#
# 使い方:
#   PROJECT=p ZONE=us-east5-b TYPE=v6e ./launch-tpu.sh
#   PROJECT=p ZONE=<v7x-zone> TYPE=v7x ./launch-tpu.sh
#
# 注意: ACCEL と RUNTIME はリージョン/時期で変わる。事前に確認:
#   bash check-quota.sh tpu
# v7x (Ironwood) は full-host, 4-chip VM 単位 (公式 docs より)。
# v7x の accelerator-type / runtime 名は時期で変わるため、check-quota.sh の
# 出力を見て上書き: ACCEL=... RUNTIME=... ZONE=... TYPE=v7x ./launch-tpu.sh

set -euo pipefail

PROJECT="${PROJECT:?PROJECT 必須}"
ZONE="${ZONE:?ZONE 必須}"
TYPE="${TYPE:-v6e}"

case "$TYPE" in
  v6e)
    DEFAULT_ACCEL="v6e-4"
    DEFAULT_RUNTIME="v2-alpha-tpuv6e"
    ;;
  v7x)
    # v7x の正式 accelerator-type / runtime は check-quota.sh で確認のうえ上書きすること
    DEFAULT_ACCEL="v7x-4"
    DEFAULT_RUNTIME="v2-alpha-tpuv7x"
    ;;
  *)
    echo "TYPE は v6e または v7x" >&2
    exit 2
    ;;
esac

NAME="${NAME:-vllm-bench-tpu-${TYPE}-4}"
ACCEL="${ACCEL:-$DEFAULT_ACCEL}"
RUNTIME="${RUNTIME:-$DEFAULT_RUNTIME}"

gcloud compute tpus tpu-vm create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --accelerator-type="$ACCEL" \
  --version="$RUNTIME"

cat <<EOF

Created: $NAME (zone=$ZONE, accelerator=$ACCEL, runtime=$RUNTIME)

次のステップ:
  gcloud compute tpus tpu-vm scp setup-vllm-tpu.sh run-bench.sh $NAME:~/ --zone=$ZONE --project=$PROJECT
  gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --project=$PROJECT
  # TPU VM 内で:
  #   HF_TOKEN=hf_xxx bash setup-vllm-tpu.sh
  #   TP=4 bash run-bench.sh   # v6e-4 / v7x-4 ともに 4 チップ
EOF
