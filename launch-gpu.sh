#!/usr/bin/env bash
#
# Gemma 4 31B 用 GPU VM を作成する雛形。
#
# 使い方:
#   PROJECT=p ZONE=us-central1-a TYPE=a100 ./launch-gpu.sh
#   PROJECT=p ZONE=us-central1-a TYPE=h100 ./launch-gpu.sh
#
# 前提: check-quota.sh で在庫・クォータが確保されていること。

set -euo pipefail

PROJECT="${PROJECT:?PROJECT 必須}"
ZONE="${ZONE:?ZONE 必須 (例: us-central1-a)}"
TYPE="${TYPE:?TYPE=a100 または TYPE=h100}"
NAME="${NAME:-vllm-bench-gpu-$TYPE}"
DISK_SIZE="${DISK_SIZE:-500GB}"

case "$TYPE" in
  a100)
    MACHINE=a2-ultragpu-2g
    ACCEL="type=nvidia-a100-80gb,count=2"
    ;;
  h100)
    MACHINE=a3-highgpu-2g
    ACCEL="type=nvidia-h100-80gb,count=2"
    ;;
  *) echo "TYPE must be a100 or h100" >&2; exit 2 ;;
esac

# Deep Learning VM の CUDA 12.9 + NVIDIA driver 580 + Ubuntu 24.04 LTS イメージ。
# 利用可能な family を確認するには:
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
  gcloud compute scp setup-vllm-gpu.sh run-bench.sh $NAME:~/ --zone=$ZONE --project=$PROJECT
  gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT
  # VM 内で:
  #   HF_TOKEN=hf_xxx bash setup-vllm-gpu.sh
  #   TP=2 bash run-bench.sh
EOF
