#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-${MODEL_ROOT:-$HOME/models/carbon}}"
MODELS="${MODELS:-Carbon-500M Carbon-3B Carbon-8B}"

mkdir -p "$DEST"

for model in $MODELS; do
  echo "Downloading HuggingFaceBio/$model to $DEST/$model"
  huggingface-cli download "HuggingFaceBio/$model" \
    --local-dir "$DEST/$model"
done
