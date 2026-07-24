# Single-Chip Benchmark: TPU v6e-1 vs A100 80GB×1 (Qwen3-8B)

`v6e-1 vs A100` 記事用の新スクリプト群。親ディレクトリの 31B 用 (v6e-4 / A100×2) とは独立。
実験計画: implicit-none-blog リポジトリの `experiments/tpu-v6e1-vs-a100/PLAN.md`。

**コピペ実行ではなく中身を読んで使うこと。** 特にモデル ID・image タグ・ゾーン在庫は実行前に要確認。

## 構成

| 区分 | インスタンス | アクセラレータ | HBM | TP |
|---|---|---|---|---|
| TPU | v6e-1 (ct6e-standard-1t) | Trillium × 1 | 32 GB | 1 |
| GPU | a2-ultragpu-1g | A100 80GB × 1 | 80 GB | 1 |

モデル: **Qwen3-8B (bf16)** — 重み 15.26 GiB。v6e-1 で KV cache 98,048 tok (fp8_e5m2 auto) を確保 (2026-07-24 実測)。

### モデル選定の経緯 (記事の H4 素材。2026-07-24)

1. **Gemma 4 12B-it**: 重み 22.28 GiB で KV cache が **13,056 tok** まで枯渇。
   medium (1,536 tok/req) で ~8 並列、long (5,120 tok/req) で ~2 並列しか入らず実験不成立。
   さらに `Gemma4UnifiedForConditionalGeneration` は tpu-inference の JAX ネイティブ実装に
   未登録で PyTorch (torchax) フォールバック、XLA コンパイル ~21 分。
2. **gemma-4-E4B-it** (8.0B): JAX ネイティブ実装はあるが mesh 初期化が形状 (1,4) を要求し
   単チップで `ValueError: cannot reshape array of size 1 into shape (1,4)` で起動不可。
   加えて `--disable_chunked_mm_input` 使用時は `--max-num-batched-tokens >= 2496` が必須
   (デフォルト 2048 だと起動時 ValueError)。
3. **Qwen3-8B**: JAX ネイティブ実装 (`qwen3.py`) で起動、コンパイル数分、KV 98k tok。採用。

**このメモリ制約・実装成熟度の実測値そのものが記事のコンテンツ**なので、採用した値は必ず記録する。

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

- [x] モデル ID: `Qwen/Qwen3-8B` (gated なし。12B/E4B からの変更経緯は上記)
- [x] vLLM image: TPU/GPU とも `v0.25.0` に統一 (確認済み 2026-07-22。両リポジトリにタグ実在)
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
CONTAINER=qwen3-8b-tpu bash run-bench-docker.sh   # TPU VM
CONTAINER=qwen3-8b-gpu bash run-bench-docker.sh   # GPU VM

# 5. 集計 (結果をローカルに回収してから)
bash ../summarize-bench.sh <results-dir>

# 6. 後片付け (借りっぱなし事故防止。同一セッションで必ずやる)
gcloud compute tpus tpu-vm delete <name> --zone=<zone> --project=<p>
gcloud compute instances delete <name> --zone=<zone> --project=<p>
```

## 公平性の設計 (記事に明記する)

- `MAX_LEN` は両側 8192 に揃える (long プロファイル 4096+1024=5120 が収まる最小の2べき)
- `MAX_NUM_SEQS` は **両側 32** を基本系列とする (TPU 側 KV 98k tok で short/medium は余裕、
  long は KV 律速 ~19 並列)。「A100 の余裕を開放した系列」は追加実験として別に取る
- `MAX_BATCHED_TOKENS` は両側 4096 (MM バジェット検査対応で導入。モデル変更後も同値を維持)
- KV cache dtype はデフォルト auto。両側で実際に選ばれた dtype をサーバログから記録する
- vLLM バージョン (image digest) を両側で記録する (`run-bench-docker.sh` が自動保存)
