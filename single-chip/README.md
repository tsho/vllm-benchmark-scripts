# Single-Chip Benchmark: TPU v6e-1 vs A100 80GB×1 (Gemma 4 12B)

`v6e-1 vs A100` 記事用の新スクリプト群。親ディレクトリの 31B 用 (v6e-4 / A100×2) とは独立。
実験計画: implicit-none-blog リポジトリの `experiments/tpu-v6e1-vs-a100/PLAN.md`。

**コピペ実行ではなく中身を読んで使うこと。** 特にモデル ID・image タグ・ゾーン在庫は実行前に要確認。

## 構成

| 区分 | インスタンス | アクセラレータ | HBM | TP |
|---|---|---|---|---|
| TPU | v6e-1 (ct6e-standard-1t) | Trillium × 1 | 32 GB | 1 |
| GPU | a2-ultragpu-1g | A100 80GB × 1 | 80 GB | 1 |

モデル: **Gemma 4 12B Instruct (bf16)** — 重み ~24GB。
v6e-1 では KV cache に残り ~8GB しかないため `MAX_LEN` / `MAX_NUM_SEQS` を絞る。
**このメモリ制約の実測値そのものが記事のコンテンツ**なので、採用した値は必ず記録する。

## ファイル

| ファイル | 役割 |
|---|---|
| `launch-tpu-v6e1.sh` | TPU VM `v6e-1` を作成 |
| `launch-gpu-a100x1.sh` | `a2-ultragpu-1g` (A100 80GB×1) を作成 |
| `run-vllm-tpu-docker.sh` | TPU VM 内: vLLM TPU Docker サーバ起動 (TP=1) |
| `run-vllm-gpu-docker.sh` | GPU VM 内: vLLM GPU Docker サーバ起動 (TP=1) |
| `run-bench-docker.sh` | 3 プロファイル × 7 レート = 21 ケースのスイープ実行 |

クォータ確認は親ディレクトリの `check-quota.sh`、集計は `summarize-bench.sh` をそのまま使う
(結果ファイルの命名規則は 31B 版と互換)。

## 実行前の確認事項

- [ ] Gemma 4 12B の HF モデル ID (`google/gemma-4-12B-it` を仮置き。要確認)
- [ ] vLLM TPU image のタグ (31B は `vllm/vllm-tpu:gemma4` だった。12B で同じか要確認)
- [ ] v6e-1 / a2-ultragpu-1g のゾーン在庫: `bash ../check-quota.sh`
- [ ] 実行時点の on-demand 時間単価を記録 (コスト計算の分母。記事に載せる)

## 流れ

```bash
# 0. クォータ・在庫確認
bash ../check-quota.sh

# 1. インスタンス作成 (ローカルから)
PROJECT=<p> ZONE=<tpu-zone> bash launch-tpu-v6e1.sh
PROJECT=<p> ZONE=<gpu-zone> bash launch-gpu-a100x1.sh

# 2. スクリプト転送 + SSH (launch スクリプトの末尾に案内が出る)

# 3. 各 VM 内でサーバ起動
HF_TOKEN=hf_xxx bash run-vllm-tpu-docker.sh   # TPU VM
HF_TOKEN=hf_xxx bash run-vllm-gpu-docker.sh   # GPU VM

# 4. 各 VM 内でベンチ実行 (21 ケース、~2h)
CONTAINER=gemma4-12b-tpu bash run-bench-docker.sh   # TPU VM
CONTAINER=gemma4-12b-gpu bash run-bench-docker.sh   # GPU VM

# 5. 集計 (結果をローカルに回収してから)
bash ../summarize-bench.sh <results-dir>

# 6. 後片付け (借りっぱなし事故防止。同一セッションで必ずやる)
gcloud compute tpus tpu-vm delete <name> --zone=<zone> --project=<p>
gcloud compute instances delete <name> --zone=<zone> --project=<p>
```

## 公平性の設計 (記事に明記する)

- `MAX_LEN` は両側 8192 に揃える (long プロファイル 4096+1024=5120 が収まる最小の2べき)
- `MAX_NUM_SEQS` は TPU 側の実用上限に合わせて **両側同値にした系列** を基本とし、
  「A100 の余裕 (KV ~56GB) を開放した系列」は追加実験として別に取る
- KV cache dtype はデフォルト auto。両側で実際に選ばれた dtype をサーバログから記録する
- vLLM バージョン (image digest) を両側で記録する (`run-bench-docker.sh` が自動保存)
