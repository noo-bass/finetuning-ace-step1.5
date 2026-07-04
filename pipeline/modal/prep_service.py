"""ACE-Step 1.5 data-prep service on Modal: whisper transcription + LM auto-labelling.

Wraps the manual workflow developed locally (whisper CLI -> acestep-5Hz-lm
format_sample -> coverage validation) into a single Modal job, so the
training service always receives clean, validated lyrics/captions.

Input (uploaded ahead of time, e.g. by the web frontend or `modal volume put`):
    acestep-lora-jobs/<job_id>/audio/*.{wav,mp3,flac,ogg,opus,m4a}
    acestep-lora-jobs/<job_id>/audio/*.lyrics.txt   (optional -- skips whisper if present)

Output:
    acestep-lora-jobs/<job_id>/audio/*.raw.lyrics.txt   (whisper transcript, kept for audit)
    acestep-lora-jobs/<job_id>/audio/*.lyrics.txt       (LM-formatted, structure-tagged)
    acestep-lora-jobs/<job_id>/audio/*.json             (caption/bpm/keyscale/timesignature/language)
    acestep-lora-jobs/<job_id>/dataset.json             (merged manifest for train_service.py)

Coverage gate (the important part): a naive LM format pass can silently discard
real lyrics (calling a vocal track "instrumental") or hallucinate fabricated/
gibberish replacement lyrics -- confirmed directly on this project's own dataset.
Every formatting attempt is checked for word-overlap against the raw whisper
transcript (>=70% retained, <=35% invented vocabulary) before being accepted;
otherwise it retries with a different seed/temperature, up to MAX_ATTEMPTS times.

Run directly:
    modal run modal/prep_service.py::prep_job --job-id <job_id>

Deploy as an HTTP API for a frontend:
    modal deploy modal/prep_service.py
    # then POST /submit {"job_id": "..."} and GET /status/{call_id}
"""

import re

import modal

app = modal.App("acestep-lora-prep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.10.0",
        "torchaudio==2.10.0",
        "transformers>=4.51.0,<4.58.0",
        "accelerate>=1.12.0",
        "soundfile",
        "loguru",
        "einops",
        "openai-whisper",
        "fastapi[standard]",
    )
    .add_local_dir("../acestep", "/root/acestep", copy=True,
                    ignore=["**/__pycache__", "**/*_test.py"])
)

checkpoints_vol = modal.Volume.from_name("acestep-checkpoints", create_if_missing=True)
jobs_vol = modal.Volume.from_name("acestep-lora-jobs", create_if_missing=True)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
MAX_ATTEMPTS = 4
MIN_COVERAGE = 0.7
MAX_INVENTED = 0.35
SEED_TEMP_ATTEMPTS = [
    (42, 0.85, 1.0),
    (123, 0.7, 1.0),
    (7, 0.5, 1.05),
    (999, 0.85, 1.0),
]


def _normalize_words(text: str) -> set:
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
    return {w for w in text.split() if w}


def _coverage_ok(raw_words: set, formatted_words: set) -> tuple[bool, float, float]:
    if not raw_words:
        return True, 1.0, 0.0
    covered = raw_words & formatted_words
    coverage = len(covered) / len(raw_words)
    invented = len(formatted_words - raw_words) / len(formatted_words) if formatted_words else 0.0
    return (coverage >= MIN_COVERAGE and invented <= MAX_INVENTED), coverage, invented


@app.function(
    image=image,
    gpu="T4",  # whisper-medium + a 1.7B LM don't need A10G-class compute; T4 is ~half the cost
    volumes={"/root/checkpoints": checkpoints_vol, "/root/jobs": jobs_vol},
    timeout=3600,
)
def prep_job(job_id: str) -> dict:
    import json
    import sys
    import torch
    import whisper as openai_whisper
    from pathlib import Path

    sys.path.insert(0, "/root")
    from acestep.llm_inference import LLMHandler
    from acestep.inference import format_sample

    audio_dir = Path(f"/root/jobs/{job_id}/audio")
    if not audio_dir.is_dir():
        return {"job_id": job_id, "success": False, "error": f"No audio dir at {audio_dir}"}

    audio_files = sorted(f for f in audio_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
    if not audio_files:
        return {"job_id": job_id, "success": False, "error": "No audio files found"}

    # -- Pass 1: transcription (skip files that already have lyrics) --------
    print(f"[prep] Transcribing {len(audio_files)} files...")
    whisper_model = openai_whisper.load_model("medium")

    for audio_path in audio_files:
        lyrics_path = audio_dir / f"{audio_path.stem}.lyrics.txt"
        raw_path = audio_dir / f"{audio_path.stem}.raw.lyrics.txt"
        if lyrics_path.exists() or raw_path.exists():
            continue
        result = whisper_model.transcribe(str(audio_path), language="en")
        raw_path.write_text(result["text"].strip() + "\n", encoding="utf-8")
        print(f"  [OK] {audio_path.name}")

    del whisper_model
    torch.cuda.empty_cache()

    # -- Pass 2: LM formatting with coverage-gated retries -------------------
    print("[prep] Loading acestep-5Hz-lm for formatting...")
    llm_handler = LLMHandler()
    status, ok = llm_handler.initialize(
        checkpoint_dir="/root/checkpoints/checkpoints",
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )
    if not ok:
        return {"job_id": job_id, "success": False, "error": f"LLM init failed: {status}"}

    samples = []
    for audio_path in audio_files:
        stem = audio_path.stem
        raw_path = audio_dir / f"{stem}.raw.lyrics.txt"
        lyrics_path = audio_dir / f"{stem}.lyrics.txt"
        json_path = audio_dir / f"{stem}.json"

        if lyrics_path.exists() and json_path.exists():
            print(f"[prep] {stem}: already labelled, skipping")
        else:
            raw_lyrics = raw_path.read_text(encoding="utf-8") if raw_path.exists() else lyrics_path.read_text(encoding="utf-8")
            raw_words = _normalize_words(raw_lyrics)

            best = None  # (invented, -coverage, result) -- lower invented wins, ties broken by higher coverage
            for attempt, (seed, temperature, rep_penalty) in enumerate(SEED_TEMP_ATTEMPTS, 1):
                torch.manual_seed(seed)
                result = format_sample(
                    llm_handler=llm_handler, caption="", lyrics=raw_lyrics,
                    user_metadata=None, temperature=temperature,
                    repetition_penalty=rep_penalty, use_constrained_decoding=True,
                )
                if not result.success:
                    continue
                formatted_words = _normalize_words(result.lyrics or "")
                passed, coverage, invented = _coverage_ok(raw_words, formatted_words)
                print(f"  [{stem}] attempt {attempt}: coverage={coverage:.0%} invented={invented:.0%} {'PASS' if passed else 'FAIL'}")
                if passed:
                    key = (invented, -coverage)
                    if best is None or key < best[0]:
                        best = (key, result)
                    # Near-perfect result -- stop early rather than burning 3 more
                    # attempts for marginal gains (this was previously gated on
                    # invented == 0.0, which the LM almost never hits exactly).
                    if coverage >= 0.95 and invented <= 0.05:
                        break

            if best is None:
                print(f"  [{stem}] FAILED all {MAX_ATTEMPTS} attempts -- keeping raw transcript untagged")
                lyrics_path.write_text(raw_lyrics, encoding="utf-8")
                json_path.write_text(json.dumps({
                    "caption": "", "bpm": None, "keyscale": "", "timesignature": "", "language": "unknown",
                }, indent=2), encoding="utf-8")
            else:
                _, result = best
                lyrics_path.write_text(result.lyrics or raw_lyrics, encoding="utf-8")
                json_path.write_text(json.dumps({
                    "caption": result.caption or "",
                    "bpm": result.bpm,
                    "keyscale": result.keyscale or "",
                    "timesignature": result.timesignature or "",
                    "language": result.language or "unknown",
                }, indent=2, ensure_ascii=False), encoding="utf-8")

        meta = json.loads(json_path.read_text(encoding="utf-8"))
        samples.append({
            # Relative to dataset.json's own directory (one level above audio/) --
            # discover_audio_files() resolves bare filenames against the JSON's
            # parent dir, so a bare name here would silently resolve to nothing.
            "filename": f"audio/{audio_path.name}",
            "caption": meta.get("caption", ""),
            "lyrics": lyrics_path.read_text(encoding="utf-8").strip(),
            "genre": meta.get("genre", ""),
            "bpm": meta.get("bpm"),
            "keyscale": meta.get("keyscale", ""),
            "timesignature": meta.get("timesignature", ""),
            "language": meta.get("language", "unknown"),
            "is_instrumental": lyrics_path.read_text(encoding="utf-8").strip() == "[Instrumental]",
        })

    dataset = {
        "metadata": {"tag_position": "prepend", "genre_ratio": 0, "custom_tag": ""},
        "samples": samples,
    }
    dataset_path = Path(f"/root/jobs/{job_id}/dataset.json")
    dataset_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    jobs_vol.commit()

    print(f"[prep] Done: {len(samples)} samples -> {dataset_path}")
    return {"job_id": job_id, "success": True, "num_samples": len(samples), "dataset_path": str(dataset_path)}


# ---------------------------------------------------------------------------
# HTTP API for a frontend (deploy with `modal deploy modal/prep_service.py`)
# ---------------------------------------------------------------------------

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def submit(job_id: str):
    """Kick off prep asynchronously; returns a call_id to poll via /status."""
    call = prep_job.spawn(job_id)
    return {"call_id": call.object_id}


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def status(call_id: str):
    """Poll a prep job. Returns {state: 'pending'|'done'|'error', result?}."""
    from modal.functions import FunctionCall

    call = FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
        return {"state": "done", "result": result}
    except TimeoutError:
        return {"state": "pending"}
    except Exception as exc:
        return {"state": "error", "error": str(exc)}
