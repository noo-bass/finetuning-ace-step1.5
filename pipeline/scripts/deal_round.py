#!/usr/bin/env python3
"""Deal a curation round: blind a batch of clips and emit a rating page.

Takes a manifest of generated clips (with their true configs), shuffles and
renames them to opaque ids (c01, c02, ...) so the rater can't be biased by
config names, copies the audio into the round dir, and renders rate.html
from the template. The true id->config mapping goes to mapping.json, which
the page never loads -- unblinding happens only after ratings are exported.

Re-deals (consistency check, H1): pass --redeal previous_round_dir and up to
--redeal-n clips that were already rated get dealt again under fresh blind
ids; agreement between the two ratings measures rater consistency.

Usage:
    python3 deal_round.py --manifest round0_manifest.json --round-dir rounds/round0
    # manifest: {"clips": [{"id": "base_alesis_s42", "path": ".../x.flac",
    #                       "config": {...anything, for the log...}}, ...]}
"""
import argparse
import json
import random
import shutil
from pathlib import Path

TEMPLATE = Path(__file__).parent / "rate_template.html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--redeal", default=None,
                    help="previous round dir (mapping.json + ratings.json) for consistency re-deals")
    ap.add_argument("--redeal-n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234,
                    help="shuffle seed, recorded in mapping.json for reproducibility")
    ap.add_argument("--excerpt-seconds", type=float, default=0,
                    help="deal a short excerpt for rating (0 = full clip); "
                         "mapping.json keeps the FULL clip path -- rate short, train long")
    ap.add_argument("--excerpt-start", type=float, default=12,
                    help="excerpt start offset in seconds (skip intros)")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    clips = list(manifest["clips"])

    if args.redeal:
        prev = Path(args.redeal)
        prev_map = json.loads((prev / "mapping.json").read_text())
        ratings_file = prev / "ratings.json"
        if ratings_file.exists():
            rng = random.Random(args.seed)
            rated_blind_ids = list(json.loads(ratings_file.read_text())["ratings"].keys())
            for blind_id in rng.sample(rated_blind_ids, min(args.redeal_n, len(rated_blind_ids))):
                entry = prev_map["clips"][blind_id]
                clips.append({**entry, "redeal_of": f"{prev.name}/{blind_id}"})
        else:
            print(f"[deal] WARNING: no ratings.json in {prev}, skipping re-deals")

    rng = random.Random(args.seed)
    rng.shuffle(clips)

    round_dir = Path(args.round_dir)
    (round_dir / "audio").mkdir(parents=True, exist_ok=True)

    mapping = {"seed": args.seed, "excerpt_seconds": args.excerpt_seconds, "clips": {}}
    page_clips = []
    for i, clip in enumerate(clips, 1):
        blind = f"c{i:02d}"
        src = Path(clip["path"])
        dst = round_dir / "audio" / f"{blind}{src.suffix}"
        if args.excerpt_seconds > 0:
            # rate short, train long: the dealt audio is an excerpt; the
            # mapping keeps the full-clip path for the training mix
            import subprocess
            subprocess.run(
                ["ffmpeg", "-v", "quiet", "-y", "-ss", str(args.excerpt_start),
                 "-i", str(src), "-t", str(args.excerpt_seconds), "-c:a", "flac", str(dst)],
                check=True)
        else:
            shutil.copy2(src, dst)
        mapping["clips"][blind] = clip
        page_clips.append({"id": blind, "file": f"audio/{blind}{src.suffix}"})

    (round_dir / "mapping.json").write_text(json.dumps(mapping, indent=2))
    html = TEMPLATE.read_text().replace("__CLIPS_JSON__", json.dumps(page_clips))
    html = html.replace("__ROUND_NAME__", round_dir.name)
    (round_dir / "rate.html").write_text(html)

    n_redeal = sum(1 for c in clips if "redeal_of" in c)
    print(f"[deal] {len(clips)} clips dealt ({n_redeal} consistency re-deals) -> {round_dir}/rate.html")
    print(f"[deal] mapping.json kept blind from the page; unblind after export")


if __name__ == "__main__":
    main()
