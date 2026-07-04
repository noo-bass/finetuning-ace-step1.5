#!/usr/bin/env python3
"""Batch-run the acestep-5Hz-lm 'format' mode over all lyrics.txt files in a
folder: backs up the raw whisper transcript, overwrites the .lyrics.txt with
structure-tagged lyrics, and writes a sidecar {name}.json with caption/bpm/
keyscale/timesignature/language for the LoRA dataset scanner.

Usage:
    uv run python3 format_lyrics_batch.py <dataset_dir>
"""
import json
import sys
from pathlib import Path

from acestep.llm_inference import LLMHandler
from acestep.inference import format_sample


def main():
    if len(sys.argv) != 2:
        print("Usage: format_lyrics_batch.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    lyrics_files = sorted(dataset_dir.glob("*.lyrics.txt"))
    if not lyrics_files:
        print(f"[FAIL] No .lyrics.txt files found in {dataset_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(lyrics_files)} lyrics files to format")

    llm_handler = LLMHandler()
    status, ok = llm_handler.initialize(
        checkpoint_dir="./checkpoints",
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="vllm",  # auto-selects MLX/PyTorch fallback on Apple Silicon
        device="auto",
    )
    print(f"[INFO] LLM init: {status} (ok={ok})")
    if not ok:
        sys.exit(1)

    results = []
    for i, lyrics_path in enumerate(lyrics_files, 1):
        stem = lyrics_path.name[: -len(".lyrics.txt")]
        raw_backup_path = lyrics_path.with_name(f"{stem}.raw.lyrics.txt")
        json_path = dataset_dir / f"{stem}.json"

        raw_lyrics = lyrics_path.read_text(encoding="utf-8")
        print(f"\n[{i}/{len(lyrics_files)}] {stem}")

        result = format_sample(
            llm_handler=llm_handler,
            caption="",
            lyrics=raw_lyrics,
            user_metadata=None,
            temperature=0.85,
            use_constrained_decoding=True,
        )

        if not result.success:
            print(f"  [FAIL] {result.error}")
            results.append({"file": stem, "success": False, "error": result.error})
            continue

        # Preserve the raw whisper transcript before overwriting
        if not raw_backup_path.exists():
            raw_backup_path.write_text(raw_lyrics, encoding="utf-8")

        lyrics_path.write_text(result.lyrics or raw_lyrics, encoding="utf-8")

        metadata = {
            "caption": result.caption or "",
            "bpm": result.bpm,
            "keyscale": result.keyscale or "",
            "timesignature": result.timesignature or "",
            "language": result.language or "unknown",
        }
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  [OK] bpm={result.bpm} key={result.keyscale} lang={result.language}")
        results.append({"file": stem, "success": True, **metadata})

    print("\n" + "=" * 60)
    ok_count = sum(1 for r in results if r["success"])
    print(f"Done: {ok_count}/{len(results)} formatted successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
