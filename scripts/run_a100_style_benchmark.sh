#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Carbon-500M}"
VARIANT="${VARIANT:-baseline}"

MODEL_ROOT="${MODEL_ROOT:-$HOME/models/carbon}"
COMPILED_ROOT="${COMPILED_ROOT:-$HOME/compiled/carbon}"
TP_DEGREE="${TP_DEGREE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-256}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-170}"

case "$VARIANT" in
  baseline) suffix="bf16"; result_suffix="a100_style" ;;
  fusedqkv) suffix="bf16-fusedqkv"; result_suffix="fusedqkv_a100_style" ;;
  topkkernel) suffix="bf16-topkkernel"; result_suffix="topkkernel_a100_style" ;;
  topkglobal1) suffix="bf16-topkglobal1"; result_suffix="topkglobal1_a100_style" ;;
  kvfp8) suffix="bf16-kvfp8"; result_suffix="kvfp8_a100_style" ;;
  nki_cte_mlp) suffix="bf16-nki-cte-mlp"; result_suffix="nki_cte_mlp_a100_style" ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

slug="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
compiled="$COMPILED_ROOT/${MODEL}-tp${TP_DEGREE}-bs${BATCH_SIZE}-s${SEQ_LEN}-c${MAX_CONTEXT_LENGTH}-n${MAX_NEW_TOKENS}-${suffix}"
output="$ROOT_DIR/results/${slug}_tp${TP_DEGREE}_bs${BATCH_SIZE}_s${SEQ_LEN}_c${MAX_CONTEXT_LENGTH}_n${MAX_NEW_TOKENS}_${result_suffix}.json"

python "$ROOT_DIR/experiments/carbon_a100_style_benchmark.py" \
  --model-name "$MODEL" \
  --model-path "$MODEL_ROOT/$MODEL" \
  --compiled-model-path "$compiled" \
  --output "$output"
