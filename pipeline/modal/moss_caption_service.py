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

# Clinical/structured mode: strict JSON out, prose rendered by code. Flowery
# free-text captions ("luminous yet wistful atmosphere") proved both hard to
# audit and far from ACE-Step's taggy native conditioning register; schema
# fields leave no room for poetry, and downstream the per-field lists enable
# programmatic cross-track aggregation (style-prompt derivation with
# evidence counts).
SCHEMA_PROMPT = (
    "Listen to this song and describe it as a single JSON object with keys "
    "genres, instrumentation, vocals, production, aesthetic. Output ONLY the "
    "JSON -- no markdown fences, no commentary. Here is an example of the "
    "required format describing a DIFFERENT song (a techno track -- do NOT "
    "copy its content, describe THIS song):\n"
    '{"genres": ["techno", "minimal techno"], '
    '"instrumentation": ["analog kick drum", "modular synth sequence", '
    '"hi-hat machine", "sub bass"], '
    '"vocals": {"present": false, "character": []}, '
    '"production": ["heavy sidechain compression", "narrow mono low end", '
    '"long delay tails", "club-oriented loudness"], '
    '"aesthetic": ["dark warehouse"]}\n'
    "Rules: 2-4 genre tags; 4-8 concrete sound sources in instrumentation; "
    "2-5 short technical vocal descriptors (empty list and present=false if "
    "instrumental); 4-8 technical mixing/production descriptors; 1-3 "
    "recording-character phrases in aesthetic. IMPORTANT: every array must "
    "contain only short strings of 2-5 words -- no nested objects, no "
    "key-value structures inside arrays. Use plain studio-engineer "
    "vocabulary. No metaphors, no storytelling, no tempo, no BPM, no musical "
    "key, no artist or song names."
)


def _repair_json(text: str) -> str:
    """Fix the model's most common JSON malformation: Python-style set
    literals ({"a", "b"}) -- brace groups with no colon are arrays."""
    import re
    pattern = re.compile(r"\{[^{}:]*\}")
    while True:
        repaired = pattern.sub(lambda m: "[" + m.group(0)[1:-1] + "]", text)
        if repaired == text:
            return text
        text = repaired


def _coerce_str_list(value) -> list:
    """Flatten the model's over-structured values into a list of strings:
    dicts contribute their keys (e.g. {"electric guitar": {...}} -> the
    instrument names), nested lists flatten, strings pass through."""
    out = []
    if isinstance(value, dict):
        out.extend(str(k) for k in value.keys())
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.extend(str(k) for k in item.keys())
            elif isinstance(item, list):
                out.extend(str(x) for x in item if isinstance(x, str))
    elif isinstance(value, str):
        out.append(value)
    return [s.strip() for s in out if s and s.strip()]


def _parse_schema(text: str) -> dict:
    """Extract and validate the JSON object from a model response.
    Raises ValueError on anything unusable."""
    import json
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    schema = json.loads(_repair_json(text[start:end + 1]))
    for key in ("genres", "instrumentation", "production", "aesthetic"):
        schema[key] = _coerce_str_list(schema.get(key))
    vocals = schema.get("vocals") or {}
    if not isinstance(vocals, dict):
        vocals = {"present": bool(vocals), "character": _coerce_str_list(vocals)}
    vocals["character"] = _coerce_str_list(vocals.get("character"))
    schema["vocals"] = vocals
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

    clinical=True switches to schema mode: MOSS returns strict JSON at
    low temperature, the caption is rendered from it by code, and the
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
    # Low temperature in clinical mode: infer.py's temperature=1.0 actively
    # encourages florid prose and JSON drift; 0.3 keeps output terse/parseable.
    temperature = 0.3 if clinical else 1.0

    def generate_caption(audio_path: Path, seed: int) -> str:
        torch.manual_seed(seed)
        raw_audio = load_audio_robust(str(audio_path), sample_rate=processor.config.mel_sr)
        inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
        inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        generated_ids = model.generate(
            **inputs, max_new_tokens=500, do_sample=True, num_beams=1,
            temperature=temperature, top_p=0.8, top_k=50, use_cache=True,
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
