# Carbon on AWS Trainium2 with NxD Inference

This repository contains Loka's Carbon-specific NxD Inference benchmark package for AWS Trainium2.

It is intentionally focused: it does not vendor the full NxD Inference repository. Instead, it
contains the small Carbon compatibility patch, benchmark scripts, measured results, and
the HTML blog artifact that can be published with GitHub Pages.

## Why Carbon

[Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon) is Hugging Face Bio's open DNA
foundation model family. It ships with open weights, training code, a data pipeline, and three model
sizes:

- `HuggingFaceBio/Carbon-500M`
- `HuggingFaceBio/Carbon-3B`
- `HuggingFaceBio/Carbon-8B`

Architecturally, Carbon is close to a Llama causal model. The domain-specific part is the tokenizer:
Carbon uses a DNA-native 6-mer tokenizer while preserving single-base resolution through its
training objective. That combination makes it a strong fit for testing whether a fresh open biology
model can run on AWS Trainium2 through an existing Llama-compatible NxD Inference path.

## Key Result

All three Carbon model sizes compiled and ran on a single `trn2.3xlarge` with TP=4.

The A100-style benchmark shape used 16 DNA prompts, 1020 bp prompt length, and approximately 1020 bp
generated per prompt.

| Model | p50 throughput | Cost point on `trn2.3xlarge` |
| --- | ---: | ---: |
| Carbon-500M | 18.83 kbp/s | $33 / generated Gbp |
| Carbon-3B | 11.00 kbp/s | $56 / generated Gbp |
| Carbon-8B | 8.21 kbp/s | $76 / generated Gbp |

The best measured setup for this workload is plain BF16 NxD Inference. KV-cache FP8, fused QKV,
top-k kernel tuning, the safe NKI CTE/MLP subset, and vLLM on Neuron were tested but did not beat
the baseline for this short-context batch-16 shape.

## The Carbon Tokenizer Fix

This is the main practical compatibility lesson.

Carbon requires Hugging Face custom tokenizer code and DNA-mode prompts:

- pass `--trust-remote-code` to `inference_demo`
- use `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` in Python
- prefix DNA prompts with `<dna>`
- use Carbon pad token id `151643`

Example:

```python
import os

from transformers import AutoTokenizer

model_root = os.path.expanduser("~/models/carbon")
tokenizer = AutoTokenizer.from_pretrained(
    f"{model_root}/Carbon-500M",
    trust_remote_code=True,
    padding_side="right",
)
tokenizer.pad_token_id = 151643
inputs = tokenizer(["<dna>ACGTTGCAACGTTGCA"], return_tensors="pt", padding=True)
```

Without that tokenizer path, the model may still run, but the benchmark no longer measures the
intended compact Carbon DNA representation.

## Quick Start On `trn2.3xlarge`

Use an AWS Neuron DLAMI or Neuron DLC with Neuron SDK 2.29/PyTorch 2.9 support.

```bash
git clone <repository-url>
cd carbon-neuronx-distributed-inference

NXDI_REF=main ./scripts/bootstrap_nxdi.sh
source .venv-carbon-sdk29/bin/activate

huggingface-cli login
./scripts/download_carbon_models.sh "$HOME/models/carbon"
```

Compile and run the three baseline A100-style benchmarks:

```bash
./scripts/run_all_baselines.sh
```

Or run one model explicitly:

```bash
./scripts/compile_carbon.sh Carbon-500M
./scripts/run_a100_style_benchmark.sh Carbon-500M
```

Results are written to `results/`.

## Useful Variants

Compile and benchmark fused QKV:

```bash
VARIANT=fusedqkv ./scripts/compile_carbon.sh Carbon-3B
VARIANT=fusedqkv ./scripts/run_a100_style_benchmark.sh Carbon-3B
```

Compile and benchmark KV-cache FP8:

```bash
VARIANT=kvfp8 ./scripts/compile_carbon.sh Carbon-3B
VARIANT=kvfp8 ./scripts/run_a100_style_benchmark.sh Carbon-3B
```

Compile and benchmark the safe NKI CTE/MLP subset:

```bash
VARIANT=nki_cte_mlp ./scripts/compile_carbon.sh Carbon-500M
VARIANT=nki_cte_mlp ./scripts/run_a100_style_benchmark.sh Carbon-500M
```

See [docs/index.html](docs/index.html) for the publishable benchmark writeup.

## GitHub Pages Blog

The blog is checked in as:

```text
docs/index.html
```

The intended Pages source is the `docs/` directory on the `main` branch. The repository includes
`docs/.nojekyll` so GitHub Pages serves the static HTML and assets directly.

## Technical Post

The publishable technical article now lives in `docs/index.html`. It focuses on the setup commands,
tokenizer fix, benchmark shape, and measured results.

## What This Repo Includes

- `patches/nxdi-carbon-support.patch` - Carbon-related NxD Inference patch.
- `scripts/` - setup, model download, compile, benchmark, and S3 upload helpers.
- `experiments/` - Python benchmark/report scripts.
- `results/` - curated JSON benchmark results from the Trainium2 run.
- `configs/` - saved Neuron config JSON files for compiled artifacts.
- `manifests/` - plain-text run manifests and result summaries.
- `docs/index.html` - the GitHub Pages blog post.

## Repository Layout

```text
.
├── configs/                  # Saved Neuron config JSONs
├── docs/                     # GitHub Pages static site
├── experiments/              # Python benchmark/report scripts
├── manifests/                # Plain-text experiment manifests
├── patches/                  # Carbon patch for NxD Inference
├── results/                  # Curated benchmark JSON outputs
└── scripts/                  # Bootstrap, compile, benchmark, upload helpers
```

## Known Caveats

- This is a direct 6-mer benchmark, not a full Factorized Nucleotide Supervision base-pair-level
  benchmark.
- Carbon-500M and Carbon-8B emit a TP=4 GQA fallback warning in NxD Inference.
- Optimization variants were tested, but the current recommended benchmark baseline is plain BF16
  NxD Inference.

## License

The NxD Inference patch follows the upstream repository license terms. The benchmark scripts and
documentation in this repository are provided under this repository's license.
