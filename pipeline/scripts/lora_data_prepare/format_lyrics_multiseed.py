#!/usr/bin/env python3
"""Retry format_sample across multiple seeds/temperatures for one lyrics file,
validate each attempt's word coverage against the raw transcript, and print
all attempts so the best can be chosen (nothing is written to disk here).

Usage:
    uv run python3 format_lyrics_multiseed.py <raw_lyrics_txt_path>
"""
import re
import sys
from pathlib import Path

import torch

from acestep.llm_inference import LLMHandler
from acestep.inference import format_sample

sys.path.insert(0, str(Path(__file__).parent))
from validate_lyrics_coverage import normalize_words, coverage_report


ATTEMPTS = [
    {"seed": 42, "temperature": 0.85, "repetition_penalty": 1.0},
    {"seed": 123, "temperature": 0.7, "repetition_penalty": 1.0},
    {"seed": 7, "temperature": 0.5, "repetition_penalty": 1.05},
    {"seed": 999, "temperature": 0.85, "repetition_penalty": 1.0},
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except ImportError:
        pass


def main():
    if len(sys.argv) != 2:
        print("Usage: format_lyrics_multiseed.py <raw_lyrics_txt_path>")
        sys.exit(1)

    raw_path = Path(sys.argv[1])
    raw_lyrics = raw_path.read_text(encoding="utf-8")
    raw_words = normalize_words(raw_lyrics)

    llm_handler = LLMHandler()
    status, ok = llm_handler.initialize(
        checkpoint_dir="./checkpoints",
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="vllm",
        device="auto",
    )
    print(f"[INFO] LLM init: {status} (ok={ok})")
    if not ok:
        sys.exit(1)

    best = None
    for i, attempt in enumerate(ATTEMPTS, 1):
        print(f"\n{'='*70}\n[Attempt {i}/{len(ATTEMPTS)}] seed={attempt['seed']} temp={attempt['temperature']} rep_penalty={attempt['repetition_penalty']}\n{'='*70}")
        set_seed(attempt["seed"])

        result = format_sample(
            llm_handler=llm_handler,
            caption="",
            lyrics=raw_lyrics,
            user_metadata=None,
            temperature=attempt["temperature"],
            repetition_penalty=attempt["repetition_penalty"],
            use_constrained_decoding=True,
        )

        if not result.success:
            print(f"[FAIL] generation error: {result.error}")
            continue

        formatted_words = normalize_words(result.lyrics or "")
        report = coverage_report(raw_words, formatted_words)
        cov, inv = report["coverage_ratio"], report["invented_ratio"]
        passed = cov >= 0.7 and inv <= 0.35

        print(f"coverage={cov:.0%} invented={inv:.0%}  {'PASS' if passed else 'FAIL'}")
        print(f"bpm={result.bpm} key={result.keyscale} lang={result.language}")
        print(f"lyrics:\n{result.lyrics}")

        if passed:
            # Rank by lowest invented ratio first, tie-broken by highest coverage --
            # a higher-coverage attempt with more fabricated content is not "better".
            key = (inv, -cov)
            if best is None or key < (best[1]["invented_ratio"], -best[1]["coverage_ratio"]):
                best = (attempt, report, result)

    print(f"\n{'='*70}")
    if best:
        attempt, report, result = best
        print(f"[BEST] seed={attempt['seed']} temp={attempt['temperature']} coverage={report['coverage_ratio']:.0%} invented={report['invented_ratio']:.0%}")
        print(f"BEST_LYRICS_START\n{result.lyrics}\nBEST_LYRICS_END")
        print(f"BEST_META bpm={result.bpm} key={result.keyscale} timesig={result.timesignature} lang={result.language}")
        print(f"BEST_CAPTION {result.caption}")
    else:
        print("[NONE] No attempt passed the coverage gate")


if __name__ == "__main__":
    main()
