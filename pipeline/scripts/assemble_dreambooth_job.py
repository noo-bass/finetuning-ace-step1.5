#!/usr/bin/env python3
"""Assemble a DreamBooth-style training dataset for an existing clinical job.

Merges: real tracks (caption prefixed with the style prompt, which carries
the trigger phrase) + base-model regularization clips (clinical caption
UNprefixed, same lyrics/bpm/keyscale as their source track). The trained
delta between "with prefix" and "without" is where the artist's sound
concentrates; the reg clips anchor the model's untriggered behavior to its
own prior (prior preservation).

The reg clips live in per-config dirs fetched by generate_sweep_samples.sh
fetch, named reg_s<seed>_t<trackindex>/<uuid>.flac.

Usage:
    python3 assemble_dreambooth_job.py \
        --clinical-dataset mkgee_clinical_dataset.json \
        --style-prompt "mkg style: ..." \
        --reg-dir generations/mkgee-clinical/generations \
        --staging-dir /tmp/mkgee-db-staging

Then upload (only reg audio + merged dataset.json -- real audio is already
on the volume in the clinical job):
    modal volume put acestep-lora-jobs <staging>/audio/<each reg flac> <job>/audio/...
    modal volume put --force acestep-lora-jobs <staging>/dataset.json <job>/dataset.json
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical-dataset", required=True)
    ap.add_argument("--style-prompt", required=True,
                    help="full style prompt incl. trigger, e.g. 'mkg style: ...'")
    ap.add_argument("--reg-dir", required=True,
                    help="dir containing reg_s<seed>_t<idx>/ subdirs with flacs")
    ap.add_argument("--staging-dir", required=True)
    ap.add_argument("--max-caption-words", type=int, default=160)
    args = ap.parse_args()

    ds = json.loads(Path(args.clinical_dataset).read_text())
    samples = ds["samples"]
    staging = Path(args.staging_dir)
    (staging / "audio").mkdir(parents=True, exist_ok=True)

    new_samples = []
    for s in samples:
        styled = f"{args.style_prompt} {s['caption']}"
        wc = len(styled.split())
        if wc > args.max_caption_words:
            print(f"[WARN] styled caption {wc} words (> {args.max_caption_words}) "
                  f"for {s['filename']} -- ACE truncates at 256 tokens incl. "
                  f"template; consider a shorter style prompt")
        new_samples.append({**s, "caption": styled})

    reg_dirs = sorted(Path(args.reg_dir).glob("reg_s*_t*"))
    if not reg_dirs:
        sys.exit(f"no reg_s*_t* dirs under {args.reg_dir}")
    n_reg = 0
    for d in reg_dirs:
        m = re.match(r"reg_s(\d+)_t(\d+)$", d.name)
        if not m:
            continue
        seed, idx = int(m.group(1)), int(m.group(2))
        if idx >= len(samples):
            sys.exit(f"{d.name}: track index {idx} out of range")
        flacs = sorted(d.glob("*.flac"))
        if not flacs:
            print(f"[WARN] {d.name}: no flac, skipping")
            continue
        src = samples[idx]
        reg_name = f"audio/{d.name}.flac"
        shutil.copy2(flacs[0], staging / reg_name)
        new_samples.append({
            **src,
            "filename": reg_name,
            "caption": src["caption"],   # clinical caption, NO style prefix
            "duration": 60.0,
        })
        n_reg += 1

    out = {**ds, "samples": new_samples}
    (staging / "dataset.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[assemble] {len(samples)} real (style-prefixed) + {n_reg} reg "
          f"(plain) = {len(new_samples)} samples")
    print(f"[assemble] staged in {staging} -- upload audio/*.flac and dataset.json")


if __name__ == "__main__":
    main()
