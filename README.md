# vLLM Gemma 4 31B Benchmark Scripts

`vllm-tpu-gpu-benchmark-2026-april` 記事の検証で使うスクリプト雛形。
**全て公式手順とリージョン在庫の確認が必要なので、コピペ実行ではなく中身を読んで使うこと。**

## ファイル

| ファイル | 役割 |
|---|---|
| `check-quota.sh` | A100/H100/TPU v6e のクォータ・在庫・ゾーンを調べる |
| `launch-gpu.sh` | `a2-ultragpu-2g` (A100×2) または `a3-highgpu-2g` (H100×2) を作成 |
| `launch-tpu.sh` | TPU VM `v6e-4` (または `v7x-4`) を作成 |
| `setup-vllm-gpu.sh` | GPU VM 内で実行: Docker + NVIDIA runtime 確認、HF DL、`vllm/vllm-openai` 事前 pull |
| `setup-vllm-tpu.sh` | TPU VM 内で実行: vLLM TPU 版 venv セットアップ（補助）。**実 vLLM サーバは Docker で起動する** |
| `run-vllm-gpu-docker.sh` | GPU VM 内で `vllm/vllm-openai` を Docker で起動 |
| `run-vllm-tpu-docker.sh` | TPU VM 内で `vllm/vllm-tpu:gemma4` を Docker で起動 |
| `run-bench.sh` | サーバへの負荷をかけてスイープ実行（Docker サーバ前提でも動く） |

## 設計方針: 全て Docker 経由

GPU 側・TPU 側ともに **vLLM を pip ではなく Docker (公式 image)** で動かす。
- GPU: `vllm/vllm-openai:latest`
- TPU: `vllm/vllm-tpu:gemma4` (公式 Gemma 4 レシピで指定)

理由:
- TPU 側で pip 経由だと PyPI の CUDA 版 vllm が引かれてしまい、torch_xla を含む TPU 対応版が入らないため `Failed to infer device type` で死ぬ。
- Docker なら依存パッケージ（torch_xla / libtpu / CUDA Toolkit）が image に固定されているので環境差を気にしなくて良い。
- GPU 側も Docker に揃えると記事・運用手順が綺麗になる。

## 流れ

```bash
# 0. プロジェクト ID と HF トークンを準備
export PROJECT=your-gcp-project
read -s -p "HF_TOKEN: " HF_TOKEN; export HF_TOKEN  # チャットに貼らない

# 1. クォータ・在庫確認
PROJECT=$PROJECT bash check-quota.sh all

# 2a. GPU 側 (例: H100 を us-central1-a で)
PROJECT=$PROJECT ZONE=us-central1-a TYPE=h100 bash launch-gpu.sh
# 作成された VM に scp:
gcloud compute scp setup-vllm-gpu.sh run-vllm-gpu-docker.sh run-bench.sh \
  vllm-bench-gpu-h100:~/ --zone=us-central1-a --project=$PROJECT
# ssh して中で:
#   HF_TOKEN=hf_xxx bash setup-vllm-gpu.sh
#   HF_TOKEN=hf_xxx bash run-vllm-gpu-docker.sh   # Docker でサーバ起動
#   sudo docker logs -f gemma4-gpu                 # ready 待ち
#   TP=2 bash run-bench.sh                          # ベンチ

# 2b. TPU 側 (例: v6e-4 を us-east5-b で)
PROJECT=$PROJECT ZONE=us-east5-b bash launch-tpu.sh
gcloud compute tpus tpu-vm scp setup-vllm-tpu.sh run-vllm-tpu-docker.sh run-bench.sh \
  vllm-bench-tpu-v6e-4:~/ --zone=us-east5-b --project=$PROJECT
# ssh して中で:
#   HF_TOKEN=hf_xxx bash setup-vllm-tpu.sh        # python3.10-venv 等の前準備（任意）
#   HF_TOKEN=hf_xxx bash run-vllm-tpu-docker.sh    # Docker でサーバ起動
#   sudo docker logs -f gemma4-tpu                 # ready 待ち
#   TP=4 bash run-bench.sh                          # ベンチ

# 3. 結果回収
gsutil -m cp -r "gs://your-bucket/bench-results/*" ./local-results/

# 4. 後片付け（インスタンスは課金が高いので必ず破棄）
gcloud compute instances delete <gpu-vm-name> --zone=<zone> --project=$PROJECT
gcloud compute tpus tpu-vm delete <tpu-vm-name> --zone=<zone> --project=$PROJECT
```

## 各スクリプトでスイープするパラメータ

`run-bench.sh` のデフォルト:

- 入出力プロファイル: `(1024,256)`, `(4096,512)`, `(8000,1000)`
- リクエストレート: `1, 2, 4, 8, 16, 32, inf`
- リクエスト数: `--num-prompts 1000`
- KV キャッシュ: `--kv-cache-dtype fp8`（メモリ ~50% 削減、公式レシピ）

## 確認が必要な前提（変動が早い箇所）

- **DLVM のイメージファミリ**: `common-cu129-debian-12` 想定。利用可能な最新を `gcloud compute images list --project deeplearning-platform-release` で確認
- **TPU runtime バージョン**: `v2-alpha-tpuv6e` を仮置き。実環境では以下で確認:
  - `gcloud compute tpus tpu-vm versions list --zone=$ZONE` で zone 内の候補一覧を取得
  - vLLM TPU 公式手順に書かれている推奨 runtime を最終確認:
    [tpu-installation](https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html) /
    [Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- **vLLM TPU 用 wheel の URL**: 公式 TPU インストール手順に追従
- **`vllm bench serve` の `--save-result` 系フラグ**: 使用 vLLM バージョンの `vllm bench serve --help` で確認

## v6e ゾーン（このアカウントで実測）

`check-quota.sh tpu` の結果から、v6e-4 を含む全トポロジが取れることが確認できたゾーン:

- `us-east5-a` / `us-east5-b` / `us-east5-c`
- `europe-west4-a`
- `asia-northeast1-b`

`launch-tpu.sh` の `ZONE` にはこれらから選んでください（クォータがあるゾーンを優先）。
