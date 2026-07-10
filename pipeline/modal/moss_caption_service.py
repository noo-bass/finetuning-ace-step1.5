"""MOSS-Music captioning service on Modal (L40S).

Re-captions an existing prepped job with MOSS-Music-8B-Instruct
(OpenMOSS, Apache 2.0) and materializes the result as a NEW job on the
same volume: audio is copied volume-side, dataset.json is rewritten with
ONLY the caption field swapped -- lyrics/bpm/keyscale/genre stay exactly
as the source job's prep produced them, so caption quality is the single
variable between the two jobs.

Model notes:
- Classes (MossMusicModel/MossMusicProcessor) live in the GitHub repo's
  src/ package, not in the HF weights repo -- the image clones the repo
  and imports from it, mirroring upstream's own infer.py.
- ~9B params, ~18GB in bf16: fits an A10G only barely, so this runs on an
  L40S (48GB) -- captioning 12 tracks is minutes of GPU time, the bigger
  card costs pennies more and removes the OOM risk.
- ACE-Step hard-truncates captions at 256 tokens including its prompt
  template, so we ask for <=110 words and hard-clamp as a backstop.

Run:
    modal run modal/moss_caption_service.py::caption_job \\
        --src-job-id mkgee-e2e-test --dst-job-id mkgee-moss
"""

import modal

app = modal.App("moss-music-captioner")

MOSS_REPO = "https://github.com/OpenMOSS/MOSS-Music"
MODEL_ID = "OpenMOSS-Team/MOSS-Music-8B-Instruct"

CAPTION_PROMPT = (
    "Listen to this song and write a detailed English caption describing "
    "its genre and subgenres, overall mood, instrumentation, production "
    "style and textures, and vocal character. Write one dense paragraph "
    "of no more than 110 words. Do not mention tempo, BPM, musical key, "
    "or any artist or song names."
)

# Clinical/structured mode: labeled-line text out, prose rendered by code.
# Flowery free-text captions ("luminous yet wistful atmosphere") proved both
# hard to audit and far from ACE-Step's taggy native conditioning register;
# schema fields leave no room for poetry, and downstream the per-field lists
# enable programmatic cross-track aggregation (style-prompt derivation with
# evidence counts).
#
# JSON was tried first and abandoned: at temp 0.3 the model loops
# ("synthwave", "retrowave", ...) and never closes the object; at temp 0.6+
# repetition_penalty the penalty avoids repeating JSON's structural tokens
# (quotes/braces/colons) and mangles the syntax itself (e.g. '"vocals"'
# becomes '" v o c e l s "'). repetition_penalty is fundamentally
# incompatible with a format built from repeated punctuation. Semicolon-
# separated labeled lines carry almost no repeated structural punctuation,
# so they tolerate both sampling noise and repetition_penalty far better.
SCHEMA_PROMPT = (
    "Listen to this song and describe it using EXACTLY five labeled lines "
    "in this format, one line per label, with short phrases separated by "
    "semicolons. Do not use JSON, markdown, or any other format -- plain "
    "text lines only.\n"
    "GENRES: <phrase>; <phrase>\n"
    "INSTRUMENTATION: <phrase>; <phrase>; ...\n"
    "VOCALS: <phrase>; <phrase>; ...  (or the single word none if instrumental)\n"
    "PRODUCTION: <phrase>; <phrase>; ...\n"
    "AESTHETIC: <phrase>; <phrase>; ...\n"
    "Here is an example describing a DIFFERENT song (a techno track -- do "
    "NOT copy its content, describe THIS song):\n"
    "GENRES: techno; minimal techno\n"
    "INSTRUMENTATION: analog kick drum; modular synth sequence; hi-hat "
    "machine; sub bass\n"
    "VOCALS: none\n"
    "PRODUCTION: heavy sidechain compression; narrow mono low end; long "
    "delay tails; club-oriented loudness\n"
    "AESTHETIC: dark warehouse\n"
    "Rules: 2-4 genre tags; 4-8 concrete sound sources in instrumentation; "
    "2-5 short technical vocal descriptors (or none if instrumental); 4-8 "
    "technical mixing/production descriptors; 1-3 recording-character "
    "phrases in aesthetic. Each phrase is 2-5 words of plain studio-engineer "
    "vocabulary -- short list items, not full sentences. No metaphors, no "
    "storytelling, no tempo, no BPM, no musical key, no artist or song "
    "names. All five lines are required: stay within the item counts above "
    "on the earlier lines so you have room to write PRODUCTION and "
    "AESTHETIC too -- do not run long on genres/instrumentation/vocals and "
    "skip the later lines. Output ONLY the five labeled lines, nothing "
    "else."
)

# Canonical schema key -> regex matching the label word the model might use.
# The model drifts to singular/alternate spellings ("GENRE:", "INSTRUMENTS:",
# "VOCAL:") despite the prompt spelling out the plural form, so each pattern
# accepts the natural singular/plural variants. Order matters where one
# alternative is a prefix of another (INSTRUMENTATION contains INSTRUMENT).
_LABEL_ALTS = {
    "GENRES": r"GENRES?",
    "INSTRUMENTATION": r"INSTRUMENTATION|INSTRUMENTS?",
    "VOCALS": r"VOCALS?",
    "PRODUCTION": r"PRODUCTIONS?",
    "AESTHETIC": r"AESTHETICS?",
}
# AESTHETIC is the last line and occasionally gets truncated by the token
# budget or dropped by the model; _render_caption already tolerates an
# empty aesthetic list, so only the other four are load-bearing.
_REQUIRED_LABELS = ("GENRES", "INSTRUMENTATION", "VOCALS", "PRODUCTION")


def _split_phrases(text: str) -> list:
    """Split a labeled line's value into short phrases, tolerating the
    model falling back to newline/bullet/markdown separators instead of
    semicolons."""
    import re
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"\n\s*[-*]\s*", "; ", text)
    parts = re.split(r"[;\n]", text)
    parts = [p.strip(" .\t*#") for p in parts]
    return [p for p in parts if p and p.lower() not in ("none", "n/a", "instrumental")]


def _parse_schema(text: str) -> dict:
    """Extract and validate the labeled lines from a model response.
    Tolerates markdown bold/bullets around labels (e.g. "**GENRES:**") and
    singular/plural label drift. Raises ValueError on anything unusable."""
    import re
    pattern = re.compile(
        r"(?im)^[\s\-*#>]*(?:"
        + "|".join(f"(?P<{k}>{v})" for k, v in _LABEL_ALTS.items())
        + r")[\s*]*:[\s*]*"
    )
    matches = []
    for m in pattern.finditer(text):
        label = next(k for k, v in m.groupdict().items() if v is not None)
        matches.append((label, m))
    found = {label for label, _ in matches}
    missing = [label for label in _REQUIRED_LABELS if label not in found]
    if missing:
        raise ValueError(f"missing labeled lines {missing}: {text[:200]!r}")
    fields = {}
    for idx, (label, m) in enumerate(matches):
        start = m.end()
        end = matches[idx + 1][1].start() if idx + 1 < len(matches) else len(text)
        fields[label] = text[start:end].strip()

    vocals_chars = _split_phrases(fields["VOCALS"])
    schema = {
        "genres": _split_phrases(fields["GENRES"]),
        "instrumentation": _split_phrases(fields["INSTRUMENTATION"]),
        "vocals": {"present": bool(vocals_chars), "character": vocals_chars},
        "production": _split_phrases(fields["PRODUCTION"]),
        "aesthetic": _split_phrases(fields.get("AESTHETIC", "")),
    }
    for key in ("genres", "instrumentation", "production"):
        if not schema[key]:
            raise ValueError(f"schema key {key!r} missing/empty")
    return schema


def _render_caption(schema: dict) -> str:
    """Deterministic clinical caption from a track schema."""
    parts = [", ".join(schema["genres"]).capitalize()]
    parts.append("Instrumentation: " + ", ".join(schema["instrumentation"]))
    vocals = schema.get("vocals") or {}
    if vocals.get("present") and vocals.get("character"):
        parts.append("Vocals: " + ", ".join(vocals["character"]))
    else:
        parts.append("Instrumental")
    parts.append("Production: " + ", ".join(schema["production"]))
    if schema.get("aesthetic"):
        parts.append(", ".join(schema["aesthetic"]).capitalize())
    return ". ".join(parts) + "."

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        # Same +cu128 pins as train_service.py, for the same reason: the
        # default-index torch wheel's bundled CUDA libs don't match what
        # torchcodec expects (OSError: libnvrtc.so.13).
        "torch==2.10.0+cu128",
        "torchaudio==2.10.0+cu128",  # MOSS src/audio_io.py imports it
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        # Exact-pinned: newer torchcodec wheels link against CUDA 13
        # (libnvrtc.so.13) which the cu128 torch wheel doesn't ship.
        # 0.9.1 is what the (working) train images resolved.
        "torchcodec==0.9.1",   # torchaudio 2.10's load() delegates to it
        # transformers 5.x breaks 4.x-era trust_remote_code modeling files;
        # MOSS-Music's custom code is from the 4.x era (May 2026).
        "transformers>=4.57.0,<5",
        "accelerate>=1.12.0",
        "soundfile",
        "librosa",
        "einops",
        "huggingface_hub",
    )
    .run_commands(f"git clone --depth 1 {MOSS_REPO} /root/MOSS-Music")
)

jobs_vol = modal.Volume.from_name("acestep-lora-jobs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/root/jobs": jobs_vol, "/root/hf_cache": hf_cache_vol},
    timeout=2 * 3600,
)
def caption_job(src_job_id: str, dst_job_id: str, max_words: int = 110,
                max_retries: int = 2, clinical: bool = False):
    """Caption every sample of <src_job_id> and write <dst_job_id> with
    swapped captions. Idempotent-ish: re-running overwrites dst outputs.

    clinical=True switches to schema mode: MOSS returns semicolon-separated
    labeled lines (GENRES:/INSTRUMENTATION:/VOCALS:/PRODUCTION:/AESTHETIC:),
    the caption is rendered from the parsed schema by code, and the
    per-track schemas are saved to moss_schemas.json for downstream
    aggregation (style-prompt derivation)."""
    import json
    import os
    import shutil
    import sys
    import time
    from pathlib import Path

    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/MOSS-Music")

    import torch
    from src.audio_io import load_audio
    from src.modeling_moss_music import MossMusicModel
    from src.processing_moss_music import MossMusicProcessor

    src_dir = Path(f"/root/jobs/{src_job_id}")
    dst_dir = Path(f"/root/jobs/{dst_job_id}")
    src_dataset = json.loads((src_dir / "dataset.json").read_text())
    samples = src_dataset["samples"]
    print(f"[moss] {len(samples)} samples from {src_job_id}", flush=True)

    print("[moss] loading model (first run downloads ~18GB)...", flush=True)
    t0 = time.time()
    model = MossMusicModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype="auto",
        device_map="cuda:0",
    )
    processor = MossMusicProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, enable_time_marker=True,
    )
    hf_cache_vol.commit()
    print(f"[moss] model ready in {time.time()-t0:.0f}s", flush=True)

    def load_audio_robust(path: str, sample_rate: int):
        """MOSS's load_audio via torchaudio->torchcodec, with a librosa
        fallback -- torchcodec's CUDA-lib linkage has broken twice now and
        captioning only needs resampled mono float32 anyway."""
        try:
            return load_audio(path, sample_rate=sample_rate)
        except Exception as exc:
            print(f"[moss] load_audio failed ({type(exc).__name__}), "
                  f"falling back to librosa: {path}", flush=True)
            import librosa
            wav, _ = librosa.load(path, sr=sample_rate, mono=True)
            return torch.from_numpy(wav)

    prompt = SCHEMA_PROMPT if clinical else CAPTION_PROMPT
    # Clinical mode: moderate temperature + a light repetition_penalty. Low
    # temp (0.3) degenerated into word-repetition loops; a heavier penalty
    # (1.15) was fine for suppressing that but also suppressed JSON's
    # necessarily-repeated punctuation, corrupting the format itself. The
    # labeled-line format has far less repeated structural punctuation than
    # JSON, so a light penalty is safe here.
    temperature = 0.7 if clinical else 1.0
    repetition_penalty = 1.1 if clinical else 1.0
    # Clinical runs kept dying with PRODUCTION/AESTHETIC dropped: the model
    # doesn't reliably hold to the 2-5-word/4-8-item budget on the earlier
    # lines (one run produced a 439-word GENRES+INSTRUMENTATION+VOCALS
    # before hitting its cap), so 500 tokens -- fine for the 110-word prose
    # mode -- runs out before reaching the later labeled lines. Give
    # clinical mode much more headroom; max_words still clamps the
    # rendered caption afterwards, so this doesn't affect output length.
    max_new_tokens = 900 if clinical else 500

    def generate_caption(audio_path: Path, seed: int) -> str:
        torch.manual_seed(seed)
        raw_audio = load_audio_robust(str(audio_path), sample_rate=processor.config.mel_sr)
        inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
        inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, num_beams=1,
            temperature=temperature, top_p=0.8, top_k=50, use_cache=True,
            repetition_penalty=repetition_penalty,
        )
        input_len = inputs["input_ids"].shape[1]
        return processor.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()

    raw_outputs = {}
    schemas = {}
    new_samples = []
    for i, sample in enumerate(samples):
        # sample["filename"] already carries the "audio/" prefix (see
        # prep_service.py: "filename": f"audio/{audio_path.name}") --
        # don't re-add it or the path doubles to audio/audio/....
        audio_path = src_dir / sample["filename"]
        caption, schema, raw, last_err = "", None, "", None
        for attempt in range(1 + max_retries):
            raw = generate_caption(audio_path, seed=42 + attempt)
            if clinical:
                try:
                    schema = _parse_schema(raw)
                    caption = _render_caption(schema)
                    break
                except Exception as exc:
                    last_err = exc
                    print(f"[moss] {sample['filename']}: bad schema on attempt "
                          f"{attempt+1} ({exc}), retrying\n"
                          f"        raw: {raw[:300]!r}", flush=True)
            else:
                caption = raw
                if len(caption.split()) >= 30:
                    break
                print(f"[moss] {sample['filename']}: short/empty caption on "
                      f"attempt {attempt+1}, retrying", flush=True)
        words = caption.split()
        if (clinical and schema is None) or len(words) < (10 if clinical else 30):
            raise RuntimeError(
                f"MOSS produced no usable output for {sample['filename']} "
                f"after {1+max_retries} attempts (last error: {last_err})"
            )
        raw_outputs[sample["filename"]] = raw
        if schema is not None:
            schemas[sample["filename"]] = schema
        clamped = " ".join(words[:max_words])
        new_samples.append({**sample, "caption": clamped})
        print(f"[moss] {i+1}/{len(samples)} {sample['filename']}: "
              f"{len(words)} words{' (clamped)' if len(words) > max_words else ''}\n"
              f"        {clamped}", flush=True)

    dst_dir.mkdir(parents=True, exist_ok=True)
    if not (dst_dir / "audio").is_dir():
        shutil.copytree(src_dir / "audio", dst_dir / "audio")
        print(f"[moss] copied audio/ -> {dst_job_id}", flush=True)

    dst_dataset = {**src_dataset, "samples": new_samples}
    (dst_dir / "dataset.json").write_text(json.dumps(dst_dataset, indent=2, ensure_ascii=False))
    (dst_dir / "moss_captions_raw.json").write_text(
        json.dumps(raw_outputs, indent=2, ensure_ascii=False))
    if schemas:
        (dst_dir / "moss_schemas.json").write_text(
            json.dumps(schemas, indent=2, ensure_ascii=False))
    jobs_vol.commit()

    print(f"[moss] wrote {dst_dir}/dataset.json ({len(new_samples)} samples)"
          f"{' + moss_schemas.json' if schemas else ''}", flush=True)
    return {"dst_job_id": dst_job_id, "samples": len(new_samples),
            "captions": raw_outputs, "schemas": schemas}
