#!/usr/bin/env bash
#
# GPU VM 内で実行するセットアップ (Docker 経路)。
#
# 想定 image: DLVM common-cu129-ubuntu-2404-nvidia-580
#   - NVIDIA Driver / NVIDIA Container Toolkit は事前同梱
#   - **Docker は同梱されていない**: 初回は別途 install が必要
#   - python3 標準に pip が入っておらず、Ubuntu 24.04 の PEP 668 で
#     `pip install --user` も制限されているため、HF 事前 DL は venv で実施
#
# 使い方 (VM 内):
#   HF_TOKEN=hf_xxx bash setup-vllm-gpu.sh
#
# 環境変数で上書き可:
#   IMAGE       (default: vllm/vllm-openai:latest)
#   MODEL       (default: google/gemma-4-31B-it)
#   HF_CACHE    (default: $HOME/.cache/huggingface)
#   SKIP_HF_DOWNLOAD=1 で事前 DL をスキップ (Docker 起動時に DL される)

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN 必須}"

IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SKIP_HF_DOWNLOAD="${SKIP_HF_DOWNLOAD:-0}"

# ---------------------------------------------------------------------------
# 1) Docker
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "INFO: Docker が見つかりません。apt install で導入します..."

  # apt のロック解放を待つ (新規 VM 起動直後は unattended-upgrades が走っている)
  while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || \
        sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
    echo "  waiting for apt lock... ($(date +%H:%M:%S))"
    sleep 15
  done

  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker

  # NVIDIA runtime 統合 (NVIDIA Container Toolkit は DLVM 同梱前提)
  if command -v nvidia-ctk >/dev/null 2>&1; then
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  else
    echo "WARN: nvidia-ctk が無いため NVIDIA runtime 統合は手動で行ってください" >&2
  fi
fi

# ---------------------------------------------------------------------------
# 2) GPU と NVIDIA runtime 確認
# ---------------------------------------------------------------------------
echo "=== nvidia-smi ==="
nvidia-smi || { echo "ERROR: NVIDIA driver がロードされていません" >&2; exit 1; }

echo "=== docker info (NVIDIA runtime 確認) ==="
sudo docker info 2>/dev/null | grep -iE 'runtimes|nvidia' || \
  echo "WARN: NVIDIA runtime が見えません。docker --gpus all が動かない可能性"

echo "=== docker --gpus all 動作確認 ==="
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 && \
  echo "OK: GPU が docker から見えています" || \
  echo "WARN: docker --gpus all で GPU が見えませんでした"

# ---------------------------------------------------------------------------
# 3) HF キャッシュ + 事前 DL (任意)
# ---------------------------------------------------------------------------
mkdir -p "$HF_CACHE"

if [[ "$SKIP_HF_DOWNLOAD" == "1" ]]; then
  echo "INFO: SKIP_HF_DOWNLOAD=1 のため HF 事前 DL をスキップします (Docker 起動時に DL されます)"
else
  echo "=== HuggingFace 事前 DL ==="

  # hf CLI を venv で入れる (Ubuntu 24.04 の PEP 668 対応)
  # huggingface_hub 1.12 以降は huggingface-cli が廃止され `hf` コマンドに移行
  HF_VENV="$HOME/.venvs/hf"
  if [[ ! -x "$HF_VENV/bin/hf" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
      echo "ERROR: python3 が見つかりません" >&2
      exit 1
    fi

    # Ubuntu 24.04 では python3-venv が同梱だが念のため
    if ! python3 -c 'import ensurepip' 2>/dev/null; then
      while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 10; done
      PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
      sudo apt-get install -y "python${PYVER}-venv" || sudo apt-get install -y python3-venv
    fi

    python3 -m venv "$HF_VENV"
    "$HF_VENV/bin/pip" install -U pip
    "$HF_VENV/bin/pip" install -U "huggingface_hub[cli]"
  fi

  HF_TOKEN="$HF_TOKEN" "$HF_VENV/bin/hf" auth login --token "$HF_TOKEN" --add-to-git-credential
  HF_HOME="$HF_CACHE" "$HF_VENV/bin/hf" download "$MODEL"
fi

# ---------------------------------------------------------------------------
# 4) Docker image を事前 pull
# ---------------------------------------------------------------------------
echo "=== docker pull $IMAGE ==="
sudo docker pull "$IMAGE"

echo
echo "Setup complete. 次のステップ:"
echo "  HF_TOKEN=\$HF_TOKEN bash ~/run-vllm-gpu-docker.sh"
