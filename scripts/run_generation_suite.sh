#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Carbon-500M}"

MODEL_ROOT="${MODEL_ROOT:-$HOME/models/carbon}"
COMPILED_ROOT="${COMPILED_ROOT:-$HOME/compiled/carbon}"
TP_DEGREE="${TP_DEGREE:-4}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-256}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-180}"

slug="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
compiled="$COMPILED_ROOT/${MODEL}-tp${TP_DEGREE}-bs1-s${SEQ_LEN}-c${MAX_CONTEXT_LENGTH}-n${MAX_NEW_TOKENS}-bf16"
output="$ROOT_DIR/results/${slug}_tp${TP_DEGREE}_bs1_s${SEQ_LEN}_c${MAX_CONTEXT_LENGTH}_n${MAX_NEW_TOKENS}_generation_suite.json"

python "$ROOT_DIR/experiments/carbon_generation_benchmark.py" \
  --model-name "$MODEL" \
  --model-path "$MODEL_ROOT/$MODEL" \
  --compiled-model-path "$compiled" \
  --output "$output"
