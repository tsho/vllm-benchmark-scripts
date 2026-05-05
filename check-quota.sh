#!/usr/bin/env bash
#
# Gemma 4 31B ベンチマーク用の GCP 在庫・クォータ確認コマンド集
#
# 対象ハードウェア:
#   - GPU: a2-ultragpu-2g (A100 80GB×2)
#   - GPU: a3-highgpu-2g  (H100 80GB×2)
#   - TPU: v6e-4 (Trillium × 4 chips)
#
# 使い方:
#   PROJECT=your-project-id ./check-quota.sh           # 全部実行
#   PROJECT=your-project-id ./check-quota.sh gpu       # GPU だけ
#   PROJECT=your-project-id ./check-quota.sh tpu       # TPU だけ
#
# 注意: リージョン候補は 2026 年時点で A100/H100/v6e の提供がある代表的なものを挙げているだけで、
#       全リージョンで在庫・クォータがあることは保証しない。実行結果で必ず確認すること。

set -euo pipefail

PROJECT="${PROJECT:?PROJECT 環境変数が必要です (例: PROJECT=my-gcp-project)}"
TARGET="${1:-all}"

# 候補リージョン（2026 年時点・要 verify）
A100_REGIONS=(us-central1 us-east4 europe-west4 asia-southeast1)
H100_REGIONS=(us-central1 us-east5 europe-west4 asia-northeast1)
# TPU は v6e (Trillium) と v7x (Ironwood / 第7世代) を両方クエリ
TPU_REGIONS=(us-east5 us-east1 us-central1 europe-west4 asia-northeast1)
# accelerator-type の grep パターン
TPU_PATTERN='v6e|v7x|tpu7x|ironwood'

# timeout コマンド検出 (macOS には標準で入っていない → coreutils の gtimeout を使う)
# bash 3.2 (macOS) では空配列展開が unbound variable を起こすため no-op (env) で代替
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD=(timeout 30)
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD=(gtimeout 30)
else
  echo "WARN: timeout/gtimeout が見つかりません (macOS なら 'brew install coreutils')。タイムアウトなしで実行します。" >&2
  TIMEOUT_CMD=(env)
fi

hr() { printf '\n===== %s =====\n' "$1"; }

# ---------------------------------------------------------------------------
# GPU: クォータと在庫
# ---------------------------------------------------------------------------
check_gpu() {
  hr "A100 80GB クォータ (リージョン別)"
  for r in "${A100_REGIONS[@]}"; do
    echo "--- $r ---"
    gcloud compute regions describe "$r" --project="$PROJECT" \
      --format="value(quotas)" \
      | tr ';' '\n' \
      | grep -i 'NVIDIA_A100_80GB' || echo "(該当クォータなし)"
  done

  hr "H100 80GB クォータ (リージョン別)"
  for r in "${H100_REGIONS[@]}"; do
    echo "--- $r ---"
    gcloud compute regions describe "$r" --project="$PROJECT" \
      --format="value(quotas)" \
      | tr ';' '\n' \
      | grep -i 'NVIDIA_H100_80GB' || echo "(該当クォータなし)"
  done

  hr "a2-ultragpu-2g (A100 80GB×2) を提供するゾーン"
  gcloud compute machine-types list --project="$PROJECT" \
    --filter="name=a2-ultragpu-2g" \
    --format="table(name,zone,guestCpus,memoryMb)"

  hr "a3-highgpu-2g (H100 80GB×2) を提供するゾーン"
  gcloud compute machine-types list --project="$PROJECT" \
    --filter="name=a3-highgpu-2g" \
    --format="table(name,zone,guestCpus,memoryMb)"

  hr "アクセラレータタイプ (A100 80GB / H100 80GB) のゾーン"
  gcloud compute accelerator-types list --project="$PROJECT" \
    --filter="name=nvidia-a100-80gb OR name=nvidia-h100-80gb" \
    --format="table(name,zone)"
}

# ---------------------------------------------------------------------------
# TPU: クォータと在庫
# ---------------------------------------------------------------------------
check_tpu() {
  hr "Cloud TPU API 有効化チェック"
  if gcloud services list --project="$PROJECT" --enabled \
      --filter="config.name=tpu.googleapis.com" --format="value(config.name)" \
      | grep -q tpu.googleapis.com; then
    echo "OK: tpu.googleapis.com 有効"
  else
    echo "NG: tpu.googleapis.com 無効。以下で有効化してから再実行:"
    echo "  gcloud services enable tpu.googleapis.com --project=$PROJECT"
    return 1
  fi

  hr "TPU アクセラレータタイプ - ゾーン別 (v6e / v7x / Ironwood)"
  echo "(該当が無いゾーンは '-' と表示)"
  found_any=0
  first_sample_zone=""
  for r in "${TPU_REGIONS[@]}"; do
    for z in "${r}-a" "${r}-b" "${r}-c"; do
      printf "  %-25s " "$z"
      raw=$("${TIMEOUT_CMD[@]}" gcloud compute tpus accelerator-types list \
        --project="$PROJECT" --zone="$z" 2>&1 || true)
      out=$(echo "$raw" | grep -iE "$TPU_PATTERN" || true)
      if [[ -n "$out" ]]; then
        echo
        echo "$out" | sed 's/^/      /'
        found_any=1
      else
        echo "-"
        [[ -z "$first_sample_zone" ]] && first_sample_zone="$z"
      fi
    done
  done

  if [[ $found_any -eq 0 && -n "$first_sample_zone" ]]; then
    hr "診断: 該当アクセラレータが見つからなかったため $first_sample_zone の生レスポンス"
    "${TIMEOUT_CMD[@]}" gcloud compute tpus accelerator-types list \
      --project="$PROJECT" --zone="$first_sample_zone" 2>&1 | head -40 || true
    echo
    echo "→ 上記から実名を確認のうえ、TPU_PATTERN/TPU_REGIONS を調整して再実行。"
  fi

  hr "TPU クォータ (v6e / v7x / Ironwood をフィルタ)"
  "${TIMEOUT_CMD[@]}" gcloud beta quotas info list \
    --service=tpu.googleapis.com \
    --project="$PROJECT" \
    --filter="quotaId~v6e OR quotaId~v7x OR quotaId~ironwood OR metricDisplayName~v6e OR metricDisplayName~v7x OR metricDisplayName~Ironwood" \
    --format="table(quotaId,metricDisplayName,quotaInfo.dimensions,quotaInfo.containerType)" 2>&1 \
    || echo "(取得失敗。Console で確認: https://console.cloud.google.com/iam-admin/quotas?service=tpu.googleapis.com)"

  hr "TPU runtime バージョン (v6e / v7x / Ironwood 候補をフィルタ)"
  for r in "${TPU_REGIONS[@]}"; do
    z="${r}-a"
    echo "--- $z ---"
    raw=$("${TIMEOUT_CMD[@]}" gcloud compute tpus tpu-vm versions list \
      --project="$PROJECT" --zone="$z" 2>&1 || true)
    filtered=$(echo "$raw" | grep -iE 'v6e|tpuv6|v7x|tpu7x|ironwood|v2-alpha' || true)
    if [[ -n "$filtered" ]]; then
      echo "$filtered"
    else
      echo "(該当 runtime がこの zone には出ていません)"
    fi
  done
  echo
  echo "→ vLLM + TPU7x で使う runtime 名は要確認 (公式 TPU 7x docs では JAX のみ言及):"
  echo "    https://docs.cloud.google.com/tpu/docs/tpu7x"
  echo "    https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html"
  echo "    https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html"
}

# ---------------------------------------------------------------------------
# 価格表へのリンク（手動確認）
# ---------------------------------------------------------------------------
print_pricing_links() {
  hr "オンデマンド価格表（実験実施時点で要確認）"
  cat <<EOF
GPU pricing: https://cloud.google.com/compute/gpus-pricing
TPU pricing: https://cloud.google.com/tpu/pricing
Compute   : https://cloud.google.com/compute/all-pricing
EOF
}

case "$TARGET" in
  gpu)   check_gpu ;;
  tpu)   check_tpu ;;
  all)   check_gpu; check_tpu; print_pricing_links ;;
  *)     echo "usage: PROJECT=<id> $0 [gpu|tpu|all]" >&2; exit 2 ;;
esac
