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
                max_retries: int = 2):
    """Caption every sample of <src_job_id> and write <dst_job_id> with
    swapped captions. Idempotent-ish: re-running overwrites dst outputs."""
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

    def generate_caption(audio_path: Path, seed: int) -> str:
        torch.manual_seed(seed)
        raw_audio = load_audio_robust(str(audio_path), sample_rate=processor.config.mel_sr)
        inputs = processor(text=CAPTION_PROMPT, audios=[raw_audio], return_tensors="pt")
        inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        generated_ids = model.generate(
            **inputs, max_new_tokens=400, do_sample=True, num_beams=1,
            temperature=1.0, top_p=0.8, top_k=50, use_cache=True,
        )
        input_len = inputs["input_ids"].shape[1]
        return processor.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()

    raw_outputs = {}
    new_samples = []
    for i, sample in enumerate(samples):
        audio_path = src_dir / "audio" / sample["filename"]
        caption = ""
        for attempt in range(1 + max_retries):
            caption = generate_caption(audio_path, seed=42 + attempt)
            words = caption.split()
            if len(words) >= 30:
                break
            print(f"[moss] {sample['filename']}: short/empty caption on "
                  f"attempt {attempt+1} ({len(words)} words), retrying", flush=True)
        words = caption.split()
        if len(words) < 30:
            raise RuntimeError(
                f"MOSS produced no usable caption for {sample['filename']} "
                f"after {1+max_retries} attempts: {caption!r}"
            )
        raw_outputs[sample["filename"]] = caption
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
    jobs_vol.commit()

    print(f"[moss] wrote {dst_dir}/dataset.json ({len(new_samples)} samples) "
          f"+ moss_captions_raw.json", flush=True)
    return {"dst_job_id": dst_job_id, "samples": len(new_samples),
            "captions": raw_outputs}
