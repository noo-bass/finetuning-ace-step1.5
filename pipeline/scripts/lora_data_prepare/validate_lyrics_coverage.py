#!/usr/bin/env python3
"""Check that LM-formatted lyrics actually preserve the raw whisper transcript's
words, rather than the LM having discarded/hallucinated replacement content.

Since format_sample's 'understand' generation phase regenerates lyrics
freeform (falling back to the raw input only when generation is empty), it
can silently drop or fabricate content. This is a coverage check, not a
diff -- it tolerates re-tagging/line-wrapping but flags missing or invented
vocabulary.

Usage:
    uv run python3 validate_lyrics_coverage.py <dataset_dir> [--min-coverage 0.7]
"""
import argparse
import re
import sys
from pathlib import Path


def normalize_words(text: str) -> list[str]:
    """Lowercase, strip structure tags/brackets, and tokenize into words."""
    text = re.sub(r"\[.*?\]", " ", text)  # drop [Verse 1], [Chorus], etc.
    text = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
    return [w for w in text.split() if w]


def coverage_report(raw_words: list[str], formatted_words: list[str]) -> dict:
    raw_set = set(raw_words)
    formatted_set = set(formatted_words)

    missing = raw_set - formatted_set
    invented = formatted_set - raw_set

    covered = raw_set & formatted_set
    coverage_ratio = len(covered) / len(raw_set) if raw_set else 1.0
    invented_ratio = len(invented) / len(formatted_set) if formatted_set else 0.0

    return {
        "raw_unique_words": len(raw_set),
        "formatted_unique_words": len(formatted_set),
        "coverage_ratio": coverage_ratio,
        "invented_ratio": invented_ratio,
        "missing_words": sorted(missing),
        "invented_words": sorted(invented),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--min-coverage", type=float, default=0.7,
                         help="Flag tracks below this fraction of raw vocabulary retained")
    parser.add_argument("--max-invented", type=float, default=0.35,
                         help="Flag tracks above this fraction of formatted vocabulary being novel")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    raw_files = sorted(dataset_dir.glob("*.raw.lyrics.txt"))

    if not raw_files:
        print(f"[FAIL] No .raw.lyrics.txt files found in {dataset_dir}")
        sys.exit(1)

    flagged = []
    print(f"{'Track':<55} {'Coverage':>9} {'Invented':>9}  Status")
    print("-" * 90)

    for raw_path in raw_files:
        stem = raw_path.name[: -len(".raw.lyrics.txt")]
        formatted_path = dataset_dir / f"{stem}.lyrics.txt"

        if not formatted_path.exists():
            print(f"{stem:<55} {'--':>9} {'--':>9}  [MISSING] no formatted lyrics file")
            flagged.append((stem, "missing formatted file"))
            continue

        raw_words = normalize_words(raw_path.read_text(encoding="utf-8"))
        formatted_words = normalize_words(formatted_path.read_text(encoding="utf-8"))

        report = coverage_report(raw_words, formatted_words)
        cov = report["coverage_ratio"]
        inv = report["invented_ratio"]

        bad_coverage = cov < args.min_coverage
        bad_invented = inv > args.max_invented
        status = "OK"
        if bad_coverage and bad_invented:
            status = "[FAIL] lyrics dropped AND replaced with invented content"
        elif bad_coverage:
            status = "[FAIL] real lyrics dropped"
        elif bad_invented:
            status = "[WARN] high invented-word ratio"

        print(f"{stem:<55} {cov:>8.0%} {inv:>8.0%}  {status}")

        if bad_coverage or bad_invented:
            flagged.append((stem, status, report))

    print("-" * 90)
    if flagged:
        print(f"\n{len(flagged)}/{len(raw_files)} tracks flagged for review:\n")
        for item in flagged:
            stem = item[0]
            print(f"  - {stem}")
            if len(item) > 2:
                report = item[2]
                if report["missing_words"]:
                    sample_missing = report["missing_words"][:15]
                    print(f"      missing (sample): {', '.join(sample_missing)}")
                if report["invented_words"]:
                    sample_invented = report["invented_words"][:15]
                    print(f"      invented (sample): {', '.join(sample_invented)}")
    else:
        print(f"\nAll {len(raw_files)} tracks passed coverage check.")

    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
