#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Carbon-500M}"
VARIANT="${VARIANT:-baseline}"

MODEL_ROOT="${MODEL_ROOT:-$HOME/models/carbon}"
COMPILED_ROOT="${COMPILED_ROOT:-$HOME/compiled/carbon}"
TP_DEGREE="${TP_DEGREE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-256}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-170}"
PAD_TOKEN_ID="${PAD_TOKEN_ID:-151643}"
LOGICAL_NC_CONFIG="${LOGICAL_NC_CONFIG:-2}"

suffix="bf16"
extra_args=()

case "$VARIANT" in
  baseline)
    suffix="bf16"
    ;;
  fusedqkv)
    suffix="bf16-fusedqkv"
    extra_args+=(--fused-qkv --qkv-kernel-enabled)
    ;;
  topkkernel)
    suffix="bf16-topkkernel"
    extra_args+=(--top-k-kernel-enabled)
    ;;
  topkglobal1)
    suffix="bf16-topkglobal1"
    extra_args+=(--top-k-kernel-enabled --global-topk 1)
    ;;
  kvfp8)
    suffix="bf16-kvfp8"
    extra_args+=(--kv-cache-quant --k-quant-method per_tensor_symmetric --v-quant-method per_tensor_symmetric --kv-quant-dtype float8_e4m3fn --kv-direct-cast)
    ;;
  nki_cte_mlp)
    suffix="bf16-nki-cte-mlp"
    extra_args+=(--attn-block-cte-nki-kernel-enabled --mlp-kernel-enabled)
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    echo "Supported variants: baseline, fusedqkv, topkkernel, topkglobal1, kvfp8, nki_cte_mlp" >&2
    exit 2
    ;;
esac

MODEL_PATH="$MODEL_ROOT/$MODEL"
COMPILED_MODEL_PATH="$COMPILED_ROOT/${MODEL}-tp${TP_DEGREE}-bs${BATCH_SIZE}-s${SEQ_LEN}-c${MAX_CONTEXT_LENGTH}-n${MAX_NEW_TOKENS}-${suffix}"

mkdir -p "$COMPILED_ROOT"

echo "Compiling $MODEL ($VARIANT)"
echo "Model path: $MODEL_PATH"
echo "Compiled path: $COMPILED_MODEL_PATH"

inference_demo \
  --model-type llama \
  --task-type causal-lm \
  run \
  --model-path "$MODEL_PATH" \
  --compiled-model-path "$COMPILED_MODEL_PATH" \
  --torch-dtype bfloat16 \
  --batch-size "$BATCH_SIZE" \
  --tp-degree "$TP_DEGREE" \
  --max-context-length "$MAX_CONTEXT_LENGTH" \
  --seq-len "$SEQ_LEN" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --enable-bucketing \
  --context-encoding-buckets "$MAX_CONTEXT_LENGTH" \
  --token-generation-buckets "$SEQ_LEN" \
  --on-device-sampling \
  --top-k 1 \
  --top-p 1.0 \
  --temperature 1.0 \
  --pad-token-id "$PAD_TOKEN_ID" \
  --trust-remote-code \
  --logical-nc-config "$LOGICAL_NC_CONFIG" \
  --compile-only \
  --prompt "<dna>ACGTTGCAACGTTGCAACGTTGCAACGTTGCA" \
  "${extra_args[@]}"
