#!/usr/bin/env bash
#
# vllm bench serve の結果 JSON をサマリ表に整形する。
#
# 使い方:
#   bash summarize-bench.sh /path/to/bench-results/<timestamp>-<container>/
#
# jq が必要 (brew install jq).

set -euo pipefail

DIR="${1:-.}"
cd "$DIR"

if [[ ! -f result-short-rate1.json ]]; then
  echo "ERROR: result-short-rate1.json が見つかりません ($DIR)" >&2
  exit 1
fi

{
  printf 'profile\trate\treq/s\tout_tok/s\ttotal_tok/s\tTTFT_p50_ms\tTTFT_p99_ms\tTPOT_p50_ms\tTPOT_p99_ms\n'
  for prof in short medium long; do
    for rate in 1 2 4 8 16 32 inf; do
      f="result-${prof}-rate${rate}.json"
      [[ -f "$f" ]] || continue
      jq -r --arg p "$prof" --arg r "$rate" \
        '[$p,$r,
          (.request_throughput|tostring),
          (.output_throughput|tostring),
          (.total_token_throughput|tostring),
          (.median_ttft_ms|tostring),
          (.p99_ttft_ms|tostring),
          (.median_tpot_ms|tostring),
          (.p99_tpot_ms|tostring)
         ]|@tsv' "$f"
    done
  done
} | column -t -s $'\t'
