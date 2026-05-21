#!/usr/bin/env python3
"""Run an A100-style batch throughput benchmark for Carbon on NxDI."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter


DNA_RE = re.compile(r"[ACGT]+")


@dataclass(frozen=True)
class PromptCase:
    name: str
    sequence: str


def repeat_to_length(seed: str, length: int) -> str:
    return (seed * ((length // len(seed)) + 1))[:length]


def batch_prompt_suite(length: int, batch_size: int) -> list[PromptCase]:
    seeds = [
        ("balanced_1", "ACGTTGCA"),
        ("balanced_2", "GATTACAG"),
        ("cpg_island_1", "CGCCGCGGCGCGCCGG"),
        ("cpg_island_2", "GCGCGACGCCGGCGCG"),
        ("at_rich_1", "AATAAATTTATAAATTA"),
        ("at_rich_2", "TATTTAAATAAATTAT"),
        ("gc_rich_1", "GCGGCCGCCGCGGCGC"),
        ("gc_rich_2", "CCGCGGCGGCCGCGGC"),
        ("orf_like_1", "ATGGCTGACGAGTTCGCCAAAGGCTACTACTAA"),
        ("orf_like_2", "ATGAAACCCGTTGACGCTTACGGTGACTGA"),
        ("tata_box_1", "TTGACATATAAAGGCTACGATCGTTA"),
        ("tata_box_2", "GCTATAAACTGACTATAGGCTTACGA"),
        ("triplet_repeat_1", "CAG"),
        ("triplet_repeat_2", "GAA"),
        ("mixed_regulatory_1", "ACGTATATAACGCGGTTAGCGCATTA"),
        ("mixed_regulatory_2", "CGTTAACCGGTATAAAGCGTACGATA"),
    ]
    cases = [
        PromptCase(name=f"{name}_{idx // len(seeds)}", sequence=repeat_to_length(seed, length))
        for idx, (name, seed) in enumerate((seeds * ((batch_size // len(seeds)) + 1))[:batch_size])
    ]
    return cases


def dna_only(text: str) -> str:
    return "".join(DNA_RE.findall(text.upper()))


def summarize(values: Iterable[float]) -> dict[str, float | None]:
    vals = list(values)
    if not vals:
        return {"avg": None, "p50": None, "min": None, "max": None}
    return {
        "avg": statistics.fmean(vals),
        "p50": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-bp", type=int, default=1020)
    parser.add_argument("--target-output-bp", type=int, default=1020)
    parser.add_argument("--max-new-tokens", type=int, default=170)
    parser.add_argument("--runs", type=int, default=3)
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
    batch_size = int(model.neuron_config.batch_size)
    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = load_generation_config(args)
    sampling_params = prepare_sampling_params(
        batch_size=batch_size,
        top_k=[args.top_k],
        top_p=[args.top_p],
        temperature=[args.temperature],
    )

    cases = batch_prompt_suite(args.prompt_bp, batch_size)
    prompts = [f"<dna>{case.sequence}" for case in cases]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_tokens = int(inputs.input_ids.shape[-1])
    target_max_length = min(
        int(model.neuron_config.max_length),
        input_tokens + args.max_new_tokens,
    )

    results: list[dict[str, object]] = []
    for run_idx in range(args.warmup_runs + args.runs):
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

        per_sequence = []
        total_generated_tokens = 0
        total_generated_bp = 0
        for idx, case in enumerate(cases):
            generated_ids = output_ids[idx, input_tokens:]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            generated_tokens = int(generated_ids.numel())
            generated_bp = len(dna_only(generated_text))
            total_generated_tokens += generated_tokens
            total_generated_bp += generated_bp
            per_sequence.append(
                {
                    "index": idx,
                    "case": asdict(case),
                    "generated_tokens": generated_tokens,
                    "generated_bp": generated_bp,
                    "valid_dna_fraction": generated_bp / max(len(generated_text), 1),
                    "generated_text_preview": generated_text[:240],
                }
            )

        row = {
            "run_index": run_idx,
            "is_warmup": run_idx < args.warmup_runs,
            "elapsed_s": elapsed,
            "batch_size": batch_size,
            "prompt_tokens": input_tokens,
            "target_max_length": target_max_length,
            "generated_tokens_total": total_generated_tokens,
            "generated_bp_total": total_generated_bp,
            "tokens_per_s": total_generated_tokens / elapsed if elapsed > 0 else None,
            "bp_per_s": total_generated_bp / elapsed if elapsed > 0 else None,
            "kbp_per_s": (total_generated_bp / 1000.0) / elapsed if elapsed > 0 else None,
            "per_sequence": per_sequence,
        }
        results.append(row)
        print(
            f"{args.model_name} run={run_idx} warmup={row['is_warmup']} "
            f"batch={batch_size} bp={total_generated_bp} elapsed={elapsed:.3f}s "
            f"kbp/s={row['kbp_per_s']:.3f}"
        )

    measured = [row for row in results if not row["is_warmup"]]
    report = {
        "schema_version": 1,
        "benchmark": "carbon_a100_style_n16_1kbp",
        "model": args.model_name,
        "model_path": args.model_path,
        "compiled_model_path": args.compiled_model_path,
        "shape": {
            "batch_size": batch_size,
            "tp_degree": int(model.neuron_config.tp_degree),
            "max_context_length": int(model.neuron_config.max_context_length),
            "max_length": int(model.neuron_config.max_length),
            "max_new_tokens": int(model.neuron_config.max_new_tokens),
            "prompt_bp": args.prompt_bp,
            "target_output_bp": args.target_output_bp,
            "requested_max_new_tokens": args.max_new_tokens,
        },
        "generation": {
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "forced_no_eos": True,
            "direct_6mer": True,
            "fns_marginal": False,
        },
        "summary": {
            "measured_runs": len(measured),
            "elapsed_s": summarize(float(row["elapsed_s"]) for row in measured),
            "tokens_per_s": summarize(float(row["tokens_per_s"]) for row in measured if row["tokens_per_s"] is not None),
            "bp_per_s": summarize(float(row["bp_per_s"]) for row in measured if row["bp_per_s"] is not None),
            "kbp_per_s": summarize(float(row["kbp_per_s"]) for row in measured if row["kbp_per_s"] is not None),
        },
        "runs": results,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
