"""One-off A/B test: does forcing float32 for the 5Hz LM on CUDA eliminate the
garbled non-English token injection observed with the default bfloat16?

Not part of the shipped pipeline -- delete after the question is answered.
"""

import modal

app = modal.App("acestep-dtype-ab-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.10.0", "torchaudio==2.10.0",
        "transformers>=4.51.0,<4.58.0", "accelerate>=1.12.0",
        "soundfile", "loguru", "einops",
    )
    .add_local_dir("../acestep", "/root/acestep", copy=True,
                    ignore=["**/__pycache__", "**/*_test.py"])
)

checkpoints_vol = modal.Volume.from_name("acestep-checkpoints", create_if_missing=True)
jobs_vol = modal.Volume.from_name("acestep-lora-jobs", create_if_missing=True)

RAW_LYRICS = """I'm in another party, losing somebody else
Both of the headless and heartless, dancing with themselves
You're not a boy, you're a liar
I'm not a liar, I'm just high"""


@app.function(
    image=image, gpu="T4",
    volumes={"/root/checkpoints": checkpoints_vol, "/root/jobs": jobs_vol},
    timeout=600,
)
def compare_dtypes():
    import sys
    import torch

    sys.path.insert(0, "/root")
    from acestep.llm_inference import LLMHandler
    from acestep.inference import format_sample

    results = {}
    for label, dtype in [("bf16_default", None), ("fp32_forced", torch.float32)]:
        llm_handler = LLMHandler()
        status, ok = llm_handler.initialize(
            checkpoint_dir="/root/checkpoints/checkpoints",
            lm_model_path="acestep-5Hz-lm-1.7B",
            backend="pt", device="cuda", dtype=dtype,
        )
        print(f"[{label}] init: {status}")
        if not ok:
            results[label] = {"error": status}
            continue

        outputs = []
        for seed in (42, 123, 7):
            torch.manual_seed(seed)
            result = format_sample(
                llm_handler=llm_handler, caption="", lyrics=RAW_LYRICS,
                user_metadata=None, temperature=0.85, use_constrained_decoding=True,
            )
            outputs.append({"seed": seed, "success": result.success, "lyrics": result.lyrics})
            print(f"[{label}] seed={seed}:\n{result.lyrics}\n")

        results[label] = {"outputs": outputs}
        del llm_handler
        torch.cuda.empty_cache()

    return results
