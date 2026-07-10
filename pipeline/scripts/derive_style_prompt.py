#!/usr/bin/env python3
"""Derive a dataset-level style prompt from per-track MOSS schemas.

Consumes moss_schemas.json (written by moss_caption_service.py --clinical),
counts attribute support across tracks, keeps attributes clearing a support
threshold, and renders a clinical style prompt. Prints the full evidence
table so every phrase in the prompt is auditable ("appears in N/12 tracks")
and disagreements (bimodal datasets) are visible rather than averaged away.

Usage:
    python3 derive_style_prompt.py <moss_schemas.json> [--trigger "mkg style"]
        [--threshold 0.55]
"""
import argparse
import json
import math
import sys
from collections import Counter


def normalize(value: str) -> str:
    return " ".join(str(value).lower().strip().split())


def collect(schemas: dict) -> dict:
    """field -> Counter of normalized attribute -> track support count."""
    fields = {"genres": Counter(), "instrumentation": Counter(),
              "vocal_character": Counter(), "production": Counter(),
              "aesthetic": Counter()}
    for schema in schemas.values():
        # set() per track: an attribute counts once per track, not per mention
        for field, key in (("genres", "genres"), ("instrumentation", "instrumentation"),
                            ("production", "production"), ("aesthetic", "aesthetic")):
            for v in set(normalize(x) for x in schema.get(key) or []):
                fields[field][v] += 1
        vocals = schema.get("vocals") or {}
        if vocals.get("present"):
            for v in set(normalize(x) for x in vocals.get("character") or []):
                fields["vocal_character"][v] += 1
    return fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schemas_json")
    ap.add_argument("--trigger", default="mkg style")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="min fraction of tracks an attribute must appear in")
    args = ap.parse_args()

    schemas = json.load(open(args.schemas_json))
    n = len(schemas)
    min_support = math.ceil(args.threshold * n)
    fields = collect(schemas)

    print(f"# Evidence table ({n} tracks, threshold >= {min_support})\n")
    kept = {}
    for field, counter in fields.items():
        kept[field] = [a for a, c in counter.most_common() if c >= min_support]
        print(f"## {field}")
        for attr, count in counter.most_common():
            marker = "KEEP" if count >= min_support else "    "
            print(f"  {marker}  {count:2d}/{n}  {attr}")
        print()

    # Disagreement check: a field where nothing clears threshold but several
    # attributes sit just under it suggests a bimodal dataset.
    for field, counter in fields.items():
        near = [a for a, c in counter.items() if min_support > c >= max(2, min_support - 2)]
        if not kept[field] and near:
            print(f"[WARN] no consensus in {field!r}; near-misses: {near} -- "
                  f"dataset may be stylistically bimodal, consider splitting\n")

    parts = []
    if kept["genres"]:
        parts.append(", ".join(kept["genres"]))
    if kept["instrumentation"]:
        parts.append("built on " + ", ".join(kept["instrumentation"]))
    if kept["vocal_character"]:
        parts.append("vocals: " + ", ".join(kept["vocal_character"]))
    if kept["production"]:
        parts.append("production: " + ", ".join(kept["production"]))
    if kept["aesthetic"]:
        parts.append(", ".join(kept["aesthetic"]))
    if not parts:
        print("[ERROR] nothing cleared the support threshold; lower --threshold")
        sys.exit(1)

    style_prompt = f"{args.trigger}: " + "; ".join(parts) + "."
    print("# Style prompt\n")
    print(style_prompt)
    print(f"\n({len(style_prompt.split())} words)")


if __name__ == "__main__":
    main()
