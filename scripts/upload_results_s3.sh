#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${S3_PREFIX:-}" ]]; then
  echo "S3_PREFIX is required, for example: s3://<bucket>/<prefix>" >&2
  exit 2
fi

aws s3 sync results/ "$S3_PREFIX/results/carbon/" --exclude "*" --include "carbon_*.json"
aws s3 sync configs/ "$S3_PREFIX/results/carbon/configs/" --exclude "*" --include "compiled__carbon-*__neuron_config.json"
aws s3 sync manifests/ "$S3_PREFIX/results/carbon/manifests/" --exclude "*" --include "carbon_*"
aws s3 cp docs/index.html "$S3_PREFIX/reports/carbon-trainium2-nxd-blogpost-2026-05-20.html"
