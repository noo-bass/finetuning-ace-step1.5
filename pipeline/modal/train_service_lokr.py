"""ACE-Step 1.5 LoKr training service on Modal (A10G) -- duplicate of
train_service.py with LoRA swapped for LoKr, per the project's own docs
claim of ~10x faster training (docs/en/LoRA_Training_Tutorial.md).

Writes to a SEPARATE output dir (lokr_output, not lora_output) so this can
run concurrently against the same job_id as an ongoing LoRA training run
without clobbering it -- that's the point: a direct, same-dataset,
same-hardware timing/quality comparison.

Run:
    modal run --detach modal/train_service_lokr.py::train_lokr \\
        --job-id <job_id> --epochs 200
"""

import os
import time

import modal

app = modal.App("acestep-lokr-trainer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "torchaudio==2.10.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers>=4.51.0,<4.58.0",
        "diffusers>=0.37.0",
        "accelerate>=1.12.0",
        "peft>=0.18.0",
        "lycoris-lora",
        "lightning>=2.0.0",
        "tensorboard>=2.20.0",
        "soundfile",
        "loguru",
        "einops",
        "numba",
        "vector-quantize-pytorch",
        "torchao>=0.16.0,<0.17.0",
        "toml",
        "modelscope",
        "pytorch-wavelets",
        "pywavelets",
        "torchcodec>=0.9.1",
        "fastapi[standard]",
    )
    .add_local_dir("../acestep", "/root/acestep", copy=True,
                    ignore=["**/__pycache__", "**/*_test.py"])
    .add_local_file("../train.py", "/root/train.py", copy=True)
)

checkpoints_vol = modal.Volume.from_name("acestep-checkpoints", create_if_missing=True)
jobs_vol = modal.Volume.from_name("acestep-lora-jobs", create_if_missing=True)

_SUBPROCESS_ENV = {
    **os.environ,
    "ACESTEP_SAFE_ROOT": os.path.commonpath([
        os.path.realpath("/root/checkpoints"),
        os.path.realpath("/root/jobs"),
    ]),
}


def _ensure_preprocessed(job_id: str, precision: str) -> str:
    """Same as train_service.py's version -- tensors are adapter-agnostic,
    so this reuses whatever train_service.py already preprocessed."""
    import subprocess
    import sys
    from pathlib import Path

    dataset_json = f"/root/jobs/{job_id}/dataset.json"
    tensor_dir = f"/root/jobs/{job_id}/tensors"

    if Path(tensor_dir).is_dir() and any(Path(tensor_dir).iterdir()):
        return tensor_dir

    if not Path(dataset_json).is_file():
        raise FileNotFoundError(
            f"No preprocessed tensors and no dataset.json at {dataset_json} -- "
            f"run prep_service.py::prep_job first."
        )

    cmd = [
        sys.executable, "/root/train.py", "fixed", "--preprocess",
        "--dataset-json", dataset_json,
        "--tensor-output", tensor_dir,
        "--checkpoint-dir", "/root/checkpoints/checkpoints",
        "--model-variant", "turbo",
        "--precision", precision,
        "--dataset-dir", tensor_dir,
        "--output-dir", f"/root/jobs/{job_id}/lokr_output",
    ]
    result = subprocess.run(cmd, cwd="/root", capture_output=True, text=True, env=_SUBPROCESS_ENV)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        raise RuntimeError(f"Preprocessing failed (returncode={result.returncode}):\n{result.stderr[-4000:]}")

    produced = list(Path(tensor_dir).glob("*.pt")) if Path(tensor_dir).is_dir() else []
    if not produced:
        raise RuntimeError(
            f"Preprocessing reported success but produced 0 tensor files in {tensor_dir}.\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )
    jobs_vol.commit()
    return tensor_dir


def _run_lokr_training(
    job_id: str,
    epochs: int = 500,
    save_every: int = 200,
    precision: str = "bf16",
    lr: float = 0.03,
    lokr_linear_dim: int = 64,
    lokr_linear_alpha: int = 128,
    lokr_factor: int = -1,
    lokr_weight_decompose: bool = True,  # DoRA, on by default per project docs
    output_subdir: str = "lokr_output",
    target_modules: str = "",
    resume_from: str = "",
):
    """LoKr training run. Writes to <job_id>/<output_subdir>/ (default
    lokr_output/), separate from train_service.py's <job_id>/lora_output/,
    so this can run alongside an ongoing LoRA job against the same
    preprocessed tensors. Pass a distinct output_subdir per run when
    sweeping several LoKr configs concurrently on one job.

    target_modules: space-separated suffix list forwarded to train.py's
    --target-modules (empty = the four attention projections). The LyCORIS
    injection suffix-matches too, so "... gate_proj up_proj down_proj"
    extends into the feed-forward layers."""
    import subprocess
    import sys
    from pathlib import Path

    dataset_dir = _ensure_preprocessed(job_id, precision)
    output_dir = f"/root/jobs/{job_id}/{output_subdir}"

    cmd = [
        sys.executable, "/root/train.py", "--yes", "fixed",
        "--checkpoint-dir", "/root/checkpoints/checkpoints",
        "--model-variant", "turbo",
        "--dataset-dir", dataset_dir,
        "--output-dir", output_dir,
        "--precision", precision,
        "--epochs", str(epochs),
        "--save-every", str(save_every),
        "--batch-size", "1",
        "--gradient-accumulation", "4",
        "--log-dir", f"{output_dir}/runs",
        "--lr", str(lr),
        "--adapter-type", "lokr",
        "--lokr-linear-dim", str(lokr_linear_dim),
        "--lokr-linear-alpha", str(lokr_linear_alpha),
        "--lokr-factor", str(lokr_factor),
    ]
    if lokr_weight_decompose:
        cmd.append("--lokr-weight-decompose")
    if target_modules.strip():
        cmd += ["--target-modules", *target_modules.split()]
    if resume_from.strip():
        cmd += ["--resume-from", f"/root/jobs/{job_id}/{resume_from.strip()}"]

    start = time.time()
    result = subprocess.run(cmd, cwd="/root", capture_output=True, text=True, env=_SUBPROCESS_ENV)
    elapsed = time.time() - start

    jobs_vol.commit()

    final_adapter_safetensors = Path(output_dir) / "final" / "lokr_weights.safetensors"
    final_adapter_dir = Path(output_dir) / "final"
    adapter_produced = final_adapter_safetensors.is_file() or (
        final_adapter_dir.is_dir() and any(final_adapter_dir.iterdir())
    )

    result_dict = {
        "job_id": job_id,
        "epochs": epochs,
        "elapsed_seconds": elapsed,
        "seconds_per_epoch": elapsed / epochs if epochs else None,
        "returncode": result.returncode,
        "adapter_produced": adapter_produced,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }

    print(f"[train_lokr] {epochs} epochs in {elapsed/60:.1f} min "
          f"({result_dict['seconds_per_epoch']:.2f}s/epoch), returncode={result.returncode}, "
          f"adapter_produced={adapter_produced}")
    if result.returncode != 0 or not adapter_produced:
        print("STDOUT TAIL:\n", result.stdout[-4000:])
        print("STDERR TAIL:\n", result.stderr[-4000:])

    return result_dict


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/checkpoints": checkpoints_vol, "/root/jobs": jobs_vol},
    timeout=6 * 3600,
)
def train_lokr(job_id: str, epochs: int = 500, save_every: int = 200, precision: str = "bf16",
               lr: float = 0.03, lokr_linear_dim: int = 64, lokr_linear_alpha: int = 128,
               lokr_factor: int = -1, lokr_weight_decompose: bool = True,
               output_subdir: str = "lokr_output", target_modules: str = "",
               resume_from: str = ""):
    return _run_lokr_training(job_id, epochs, save_every, precision, lr, lokr_linear_dim,
                              lokr_linear_alpha, lokr_factor, lokr_weight_decompose,
                              output_subdir, target_modules, resume_from)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/checkpoints": checkpoints_vol, "/root/jobs": jobs_vol},
    timeout=6 * 3600,
)
def train_lokr_h100(job_id: str, epochs: int = 500, save_every: int = 200, precision: str = "bf16",
                    lr: float = 0.03, lokr_linear_dim: int = 64, lokr_linear_alpha: int = 128,
                    lokr_factor: int = -1, lokr_weight_decompose: bool = True,
                    output_subdir: str = "lokr_output", target_modules: str = "",
                    resume_from: str = ""):
    """Same run on H100: ~4x A10G wall-clock at similar per-epoch cost."""
    return _run_lokr_training(job_id, epochs, save_every, precision, lr, lokr_linear_dim,
                              lokr_linear_alpha, lokr_factor, lokr_weight_decompose,
                              output_subdir, target_modules, resume_from)
