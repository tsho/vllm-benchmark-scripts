#!/usr/bin/env bash
#
# TPU VM (v6e) 内で実行する vLLM TPU セットアップ。
#
# 公式手順は以下を必ず参照すること（API・パッケージ名は変動が早い）:
#   https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html
#   https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
#
# 使い方 (TPU VM 内):
#   HF_TOKEN=hf_xxx bash setup-vllm-tpu.sh

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN 必須}"

# 0) Python venv 用パッケージ (TPU runtime v2-alpha-tpuv6e は Python 3.10、
#    python3.10-venv が標準で入っていないため apt install)
if ! python3 -c 'import ensurepip' 2>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y python3.10-venv
fi

# 1) Python venv (TPU VM の system python = 3.10 を利用)
VENV="${VENV:-$HOME/.venvs/vllm}"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip

# 2) vLLM TPU 版インストール
#    実際のコマンドは公式 TPU インストール手順に追従すること。
#    例（変動あり）:
#      pip install vllm-tpu --pre --extra-index-url https://wheels.vllm.ai/nightly/tpu
#    または torch_xla / libtpu の固定バージョン指定が必要な場合がある。
#    ※下記はプレースホルダ。実行前に最新ドキュメントで差し替える。
pip install --pre vllm \
  --extra-index-url https://wheels.vllm.ai/nightly/tpu \
  || { echo "ERROR: TPU 用 wheel の URL を確認してください" >&2; exit 1; }

# 3) HuggingFace ログイン + モデル事前 DL
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
huggingface-cli download google/gemma-4-31B-it

# 4) freeze
pip freeze > "$HOME/pip-freeze-$(date +%Y%m%d-%H%M%S).txt"

echo "Done. Activate venv with: source $VENV/bin/activate"
