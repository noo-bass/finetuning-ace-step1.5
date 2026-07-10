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
import re
import json
import math
import sys
from collections import Counter


def normalize(value: str) -> str:
    return " ".join(str(value).lower().strip().split())


STOPWORDS = {
    "the", "a", "an", "of", "with", "for", "and", "or", "on", "in", "into",
    "that", "to", "at", "over", "through", "while", "yet", "without", "via",
    "across", "all", "its", "their", "his", "her", "throughout", "rather",
    "than", "meets", "blending", "creating", "preserving", "emphasizing",
    "evoking", "adding", "conveying", "prioritizing", "reminiscent",
}


def term_support(fields: dict, n_tracks_by_field: dict) -> dict:
    """Per-field track-support counts at the TERM level (unigrams+bigrams,
    stopword-filtered). MOSS paraphrases the same attribute differently per
    track ('gentle tape saturation' vs 'tape saturation warmth'), so exact-
    phrase counting drastically undercounts consensus; term counting keeps
    the audit trail ('saturation: 10/12') without hand-merging synonyms."""
    per_field = {}
    for field, per_track_phrases in fields.items():
        counter = Counter()
        for phrases in per_track_phrases:
            terms = set()
            for phrase in phrases:
                words = [w for w in re.split(r"[^a-z0-9&'-]+", phrase)
                         if w and w not in STOPWORDS]
                terms.update(words)
                terms.update(f"{a} {b}" for a, b in zip(words, words[1:]))
            counter.update(terms)
        per_field[field] = counter
    return per_field


def pick_terms(counter: Counter, min_support: int, max_terms: int) -> list:
    """Highest-support terms, preferring bigrams over the unigrams they
    contain (keep 'tape saturation', drop bare 'tape'/'saturation')."""
    kept = []
    eligible = [(t, c) for t, c in counter.most_common() if c >= min_support]
    bigrams = [(t, c) for t, c in eligible if " " in t]
    unigrams = [(t, c) for t, c in eligible if " " not in t]
    covered = set()
    for term, _ in bigrams:
        if len(kept) >= max_terms:
            break
        kept.append(term)
        covered.update(term.split())
    for term, _ in unigrams:
        if len(kept) >= max_terms:
            break
        if term not in covered:
            kept.append(term)
    return kept


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

    # per-field list of per-track phrase lists (term counting needs
    # track-level grouping, not a flat counter)
    per_track = {"genres": [], "instrumentation": [], "vocal_character": [],
                 "production": [], "aesthetic": []}
    for schema in schemas.values():
        per_track["genres"].append([normalize(x) for x in schema.get("genres") or []])
        per_track["instrumentation"].append([normalize(x) for x in schema.get("instrumentation") or []])
        per_track["production"].append([normalize(x) for x in schema.get("production") or []])
        per_track["aesthetic"].append([normalize(x) for x in schema.get("aesthetic") or []])
        vocals = schema.get("vocals") or {}
        per_track["vocal_character"].append(
            [normalize(x) for x in (vocals.get("character") or [])] if vocals.get("present") else [])

    terms = term_support(per_track, {})
    print(f"# Term-level evidence ({n} tracks, threshold >= {min_support}; "
          f"genres use >= 3)\n")
    for field, counter in terms.items():
        print(f"## {field}")
        for term, count in counter.most_common(15):
            marker = "KEEP" if count >= (3 if field == "genres" else min_support) else "    "
            print(f"  {marker}  {count:2d}/{n}  {term}")
        print()

    kept = {
        # genres are inherently multi-modal across a discography: lower bar
        "genres": pick_terms(terms["genres"], 3, 4),
        "instrumentation": pick_terms(terms["instrumentation"], min_support, 6),
        "vocal_character": pick_terms(terms["vocal_character"], min_support, 4),
        "production": pick_terms(terms["production"], min_support, 5),
        "aesthetic": pick_terms(terms["aesthetic"], min_support, 3),
    }

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
