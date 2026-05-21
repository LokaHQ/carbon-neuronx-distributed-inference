#!/usr/bin/env bash
set -euo pipefail

for model in Carbon-500M Carbon-3B Carbon-8B; do
  scripts/compile_carbon.sh "$model"
  scripts/run_a100_style_benchmark.sh "$model"
done
