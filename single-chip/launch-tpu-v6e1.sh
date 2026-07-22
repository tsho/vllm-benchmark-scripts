#!/usr/bin/env bash
#
# 単チップベンチ用 TPU VM (v6e-1) を作成する。
#
# 使い方:
#   PROJECT=p ZONE=us-east5-b bash launch-tpu-v6e1.sh
#
# 注意: RUNTIME はリージョン/時期で変わる。事前に確認:
#   bash ../check-quota.sh tpu

set -euo pipefail

PROJECT="${PROJECT:?PROJECT 必須}"
ZONE="${ZONE:?ZONE 必須}"

NAME="${NAME:-vllm-bench-tpu-v6e-1}"
ACCEL="${ACCEL:-v6e-1}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"

gcloud compute tpus tpu-vm create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --accelerator-type="$ACCEL" \
  --version="$RUNTIME"

cat <<EOF

Created: $NAME (zone=$ZONE, accelerator=$ACCEL, runtime=$RUNTIME)

次のステップ:
  gcloud compute tpus tpu-vm scp run-vllm-tpu-docker.sh run-bench-docker.sh $NAME:~/ --zone=$ZONE --project=$PROJECT
  gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --project=$PROJECT
  # TPU VM 内で:
  #   HF_TOKEN=hf_xxx bash run-vllm-tpu-docker.sh
  #   CONTAINER=gemma4-12b-tpu bash run-bench-docker.sh

後片付け (必須):
  gcloud compute tpus tpu-vm delete $NAME --zone=$ZONE --project=$PROJECT
EOF
