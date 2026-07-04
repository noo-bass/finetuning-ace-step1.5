#!/usr/bin/env python3
"""Assemble a Side-Step-compatible dataset JSON from per-track sidecar files
(.lyrics.txt + .json) in a folder, matching the schema expected by
acestep.training_v2.preprocess_discovery.load_sample_metadata:

    {"metadata": {...}, "samples": [{filename, caption, lyrics, genre, bpm,
     keyscale, timesignature, duration, is_instrumental}, ...]}

Usage:
    uv run python3 build_dataset_json.py <dataset_dir> <output_json>
"""
import json
import sys
from pathlib import Path

import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def main():
    if len(sys.argv) != 3:
        print("Usage: build_dataset_json.py <dataset_dir> <output_json>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    output_json = Path(sys.argv[2])

    audio_files = sorted(
        f for f in dataset_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )

    samples = []
    for audio_path in audio_files:
        stem = audio_path.stem
        lyrics_path = dataset_dir / f"{stem}.lyrics.txt"
        json_path = dataset_dir / f"{stem}.json"

        lyrics = lyrics_path.read_text(encoding="utf-8").strip() if lyrics_path.exists() else "[Instrumental]"
        meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}

        try:
            info = sf.info(str(audio_path))
            duration = round(info.frames / info.samplerate, 2)
        except Exception as e:
            print(f"[WARN] Could not read duration for {audio_path.name}: {e}")
            duration = 0

        samples.append({
            "filename": audio_path.name,
            "caption": meta.get("caption", ""),
            "lyrics": lyrics,
            "genre": meta.get("genre", ""),
            "bpm": meta.get("bpm"),
            "keyscale": meta.get("keyscale", ""),
            "timesignature": meta.get("timesignature", ""),
            "language": meta.get("language", "unknown"),
            "duration": duration,
            "is_instrumental": (lyrics.strip() == "[Instrumental]"),
        })
        print(f"[OK] {audio_path.name}: duration={duration}s bpm={meta.get('bpm')} lang={meta.get('language')}")

    dataset = {
        "metadata": {
            "tag_position": "prepend",
            "genre_ratio": 0,
            "custom_tag": "",
        },
        "samples": samples,
    }

    output_json.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Wrote {len(samples)} samples to {output_json}")


if __name__ == "__main__":
    main()
