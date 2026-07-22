#!/usr/bin/env bash
#
# 単チップベンチ用 GPU VM (A100 80GB × 1 = a2-ultragpu-1g) を作成する。
#
# 使い方:
#   PROJECT=p ZONE=us-central1-a bash launch-gpu-a100x1.sh
#
# 前提: ../check-quota.sh で在庫・クォータが確保されていること。

set -euo pipefail

PROJECT="${PROJECT:?PROJECT 必須}"
ZONE="${ZONE:?ZONE 必須 (例: us-central1-a)}"
NAME="${NAME:-vllm-bench-gpu-a100x1}"
DISK_SIZE="${DISK_SIZE:-300GB}"

MACHINE="${MACHINE:-a2-ultragpu-1g}"
ACCEL="${ACCEL:-type=nvidia-a100-80gb,count=1}"

# Deep Learning VM イメージ (31B 版と同じ family)。利用可能な family の確認:
#   gcloud compute images list --project=deeplearning-platform-release --filter="family~common-cu" --format="value(family)" | sort -u
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="$ACCEL" \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE --restart-on-failure \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform

cat <<EOF

Created: $NAME (zone=$ZONE, machine=$MACHINE)

次のステップ:
  gcloud compute scp run-vllm-gpu-docker.sh run-bench-docker.sh $NAME:~/ --zone=$ZONE --project=$PROJECT
  gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT
  # VM 内で:
  #   HF_TOKEN=hf_xxx bash run-vllm-gpu-docker.sh
  #   CONTAINER=gemma4-12b-gpu bash run-bench-docker.sh

後片付け (必須):
  gcloud compute instances delete $NAME --zone=$ZONE --project=$PROJECT
EOF
