#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-carbon-sdk29}"
NXDI_REPO="${NXDI_REPO:-https://github.com/aws-neuron/neuronx-distributed-inference.git}"
NXDI_REF="${NXDI_REF:-main}"
NXDI_DIR="${NXDI_DIR:-$ROOT_DIR/third_party/neuronx-distributed-inference}"

echo "Creating virtual environment at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
python -m pip install -r "$ROOT_DIR/requirements-neuron-sdk29.txt"

mkdir -p "$(dirname "$NXDI_DIR")"
if [[ ! -d "$NXDI_DIR/.git" ]]; then
  git clone "$NXDI_REPO" "$NXDI_DIR"
fi

git -C "$NXDI_DIR" fetch --all --tags
git -C "$NXDI_DIR" checkout "$NXDI_REF"

PATCH="$ROOT_DIR/patches/nxdi-carbon-support.patch"
if git -C "$NXDI_DIR" apply --check "$PATCH"; then
  git -C "$NXDI_DIR" apply "$PATCH"
  echo "Applied Carbon NxDI patch."
else
  echo "Patch did not apply cleanly. It may already be present; continuing."
fi

python -m pip install -e "$NXDI_DIR"

echo
echo "Bootstrap complete."
echo "Activate with: source $VENV_DIR/bin/activate"
