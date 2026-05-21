#!/usr/bin/env python3
"""Run a DNA-aware Carbon generation benchmark on compiled NxDI artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter


DNA_RE = re.compile(r"[ACGT]+")
TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class PromptCase:
    name: str
    description: str
    sequence: str


def repeat_to_length(seed: str, length: int) -> str:
    return (seed * ((length // len(seed)) + 1))[:length]


def prompt_suite(length: int) -> list[PromptCase]:
    balanced = repeat_to_length("ACGTTGCA", length)
    cpg = repeat_to_length("CGCCGCGGCGCGCCGG", length)
    at_rich = repeat_to_length("AATAAATTTATAAATTA", length)
    gc_rich = repeat_to_length("GCGGCCGCCGCGGCGC", length)
    orf = "ATG" + repeat_to_length("GCTGACGAGTTCGCCAAAGGCTACTAC", length - 6) + "TAA"
    tata = repeat_to_length("TTGACATATAAAGGCTACGATCGTTA", length)
    repeat = repeat_to_length("CAG", length)
    return [
        PromptCase("balanced", "Balanced synthetic DNA", balanced),
        PromptCase("cpg_island", "CpG-rich regulatory-style sequence", cpg),
        PromptCase("at_rich", "AT-rich low-complexity sequence", at_rich),
        PromptCase("gc_rich", "GC-rich high-complexity sequence", gc_rich),
        PromptCase("orf_like", "ATG/open-reading-frame style sequence", orf[:length]),
        PromptCase("tata_box", "Promoter-style TATA motif sequence", tata),
        PromptCase("triplet_repeat", "CAG triplet-repeat stress case", repeat),
    ]


def dna_only(text: str) -> str:
    return "".join(DNA_RE.findall(text.upper()))


def gc_content(seq: str) -> float | None:
    if not seq:
        return None
    return (seq.count("G") + seq.count("C")) / len(seq)


def shannon_entropy(values: Iterable[str]) -> float | None:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return None
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def kmers(seq: str, k: int) -> list[str]:
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def longest_homopolymer(seq: str) -> int:
    best = 0
    current = 0
    prev = None
    for base in seq:
        current = current + 1 if base == prev else 1
        best = max(best, current)
        prev = base
    return best


def motif_counts(seq: str) -> dict[str, int]:
    motifs = ["TATA", "CG", "ATG", "TAA", "TAG", "TGA", "CAG"]
    return {motif: seq.count(motif) for motif in motifs}


def sequence_metrics(text: str) -> dict[str, float | int | None]:
    seq = dna_only(text)
    sixmers = kmers(seq, 6)
    dimers = kmers(seq, 2)
    return {
        "bp": len(seq),
        "valid_dna_fraction": len(seq) / max(len(text), 1),
        "gc_content": gc_content(seq),
        "longest_homopolymer": longest_homopolymer(seq),
        "unique_6mer_ratio": (len(set(sixmers)) / len(sixmers)) if sixmers else None,
        "one_mer_entropy": shannon_entropy(seq),
        "two_mer_entropy": shannon_entropy(dimers),
        "special_tag_count": len(TAG_RE.findall(text)),
        "non_dna_characters": max(len(text) - len(seq), 0),
        **{f"motif_{motif}": count for motif, count in motif_counts(seq).items()},
    }


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "p50": None, "min": None, "max": None}
    return {
        "avg": statistics.fmean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-bp", type=int, default=1080)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--runs-per-prompt", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--pad-token-id", type=int, default=151643)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def load_generation_config(args: argparse.Namespace) -> GenerationConfig:
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config.update(
        do_sample=False,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        pad_token_id=args.pad_token_id,
        eos_token_id=[],
        max_new_tokens=args.max_new_tokens,
    )
    return generation_config


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token_id = args.pad_token_id

    model = NeuronLlamaForCausalLM(args.compiled_model_path)
    model.load(args.compiled_model_path)
    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = load_generation_config(args)
    sampling_params = prepare_sampling_params(
        batch_size=model.neuron_config.batch_size,
        top_k=[args.top_k],
        top_p=[args.top_p],
        temperature=[args.temperature],
    )

    cases = prompt_suite(args.prompt_bp)
    results: list[dict[str, object]] = []

    for case in cases:
        prompt_text = f"<dna>{case.sequence}"
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
        input_tokens = int(inputs.input_ids.shape[-1])
        target_max_length = min(
            int(model.neuron_config.max_length),
            input_tokens + args.max_new_tokens,
        )

        for run_idx in range(args.warmup_runs + args.runs_per_prompt):
            model.reset()
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = generation_model.generate(
                    inputs.input_ids,
                    generation_config=generation_config,
                    attention_mask=inputs.attention_mask,
                    max_length=target_max_length,
                    sampling_params=sampling_params,
                )
            elapsed = time.perf_counter() - start

            decoded = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            generated_ids = output_ids[0, input_tokens:]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            generated_tokens = int(generated_ids.numel())
            generated_bp = len(dna_only(generated_text))

            row = {
                "model": args.model_name,
                "case": asdict(case),
                "prompt_text_preview": prompt_text[:160],
                "run_index": run_idx,
                "is_warmup": run_idx < args.warmup_runs,
                "prompt_tokens": input_tokens,
                "target_max_length": target_max_length,
                "generated_tokens": generated_tokens,
                "generated_bp": generated_bp,
                "elapsed_s": elapsed,
                "tokens_per_s": generated_tokens / elapsed if elapsed > 0 else None,
                "bp_per_s": generated_bp / elapsed if elapsed > 0 else None,
                "full_text_preview": decoded[:800],
                "generated_text_preview": generated_text[:800],
                "prompt_metrics": sequence_metrics(case.sequence),
                "generated_metrics": sequence_metrics(generated_text),
            }
            results.append(row)

            print(
                f"{args.model_name} {case.name} run={run_idx} "
                f"tokens={generated_tokens} bp={generated_bp} elapsed={elapsed:.3f}s"
            )

    measured = [row for row in results if not row["is_warmup"]]
    report = {
        "schema_version": 1,
        "model": args.model_name,
        "model_path": args.model_path,
        "compiled_model_path": args.compiled_model_path,
        "shape": {
            "batch_size": int(model.neuron_config.batch_size),
            "tp_degree": int(model.neuron_config.tp_degree),
            "max_context_length": int(model.neuron_config.max_context_length),
            "max_length": int(model.neuron_config.max_length),
            "max_new_tokens": int(model.neuron_config.max_new_tokens),
            "prompt_bp": args.prompt_bp,
            "requested_max_new_tokens": args.max_new_tokens,
        },
        "generation": {
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "forced_no_eos": True,
        },
        "summary": {
            "measured_runs": len(measured),
            "elapsed_s": summarize([float(row["elapsed_s"]) for row in measured]),
            "tokens_per_s": summarize([float(row["tokens_per_s"]) for row in measured if row["tokens_per_s"] is not None]),
            "bp_per_s": summarize([float(row["bp_per_s"]) for row in measured if row["bp_per_s"] is not None]),
            "valid_dna_fraction": summarize([
                float(row["generated_metrics"]["valid_dna_fraction"])  # type: ignore[index]
                for row in measured
            ]),
            "gc_content": summarize([
                float(row["generated_metrics"]["gc_content"])  # type: ignore[index]
                for row in measured
                if row["generated_metrics"]["gc_content"] is not None  # type: ignore[index]
            ]),
        },
        "runs": results,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
