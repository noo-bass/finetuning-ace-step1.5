"""ACE-Step 1.5 LoRA training service on Modal (A10G).

Wraps the same acestep.training_v2 pipeline used locally (train.py fixed)
so a job that works on this Mac produces the same result on Modal, just
much faster on a real CUDA GPU with flash-attention available.

Storage layout (two persistent Volumes):
    acestep-checkpoints/checkpoints/          -- base model (shared across all users/jobs)
    acestep-lora-jobs/<job_id>/tensors/      -- preprocessed .pt tensors (input)
    acestep-lora-jobs/<job_id>/lora_output/  -- LoRA checkpoints + final weights (output)

Setup (one-time, from this machine):
    modal volume create acestep-checkpoints
    modal volume create acestep-lora-jobs
    modal volume put acestep-checkpoints ../checkpoints checkpoints
    modal volume put acestep-lora-jobs ../local_data/mk_gee_tensors <job_id>/tensors

Run a calibration benchmark (measures real per-epoch time + cost on A10G):
    modal run modal/train_service.py::calibrate --job-id <job_id> --epochs 20

Run a full training job:
    modal run modal/train_service.py::train_lora --job-id <job_id> --epochs 800
"""

import time

import modal

app = modal.App("acestep-lora-trainer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        # Pinned to the exact +cu128 build this repo's own pyproject.toml uses for
        # linux x86_64 -- a bare "torch==2.10.0" pulls in a default-index wheel
        # whose bundled CUDA runtime libs don't match what torchcodec expects,
        # causing `OSError: libnvrtc.so.13: cannot open shared object file`.
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
        "torchcodec>=0.9.1",  # required by acestep's audio loading path (audio_io.py)
        "fastapi[standard]",
        # flash-attn omitted for v1: needs a source build against a matching CUDA
        # toolkit and would make the image fragile. A10G supports it (SM 8.6) --
        # add later as a speed optimization once the base pipeline is proven.
    )
    .add_local_dir("../acestep", "/root/acestep", copy=True,
                    ignore=["**/__pycache__", "**/*_test.py"])
    .add_local_file("../train.py", "/root/train.py", copy=True)
)

checkpoints_vol = modal.Volume.from_name("acestep-checkpoints", create_if_missing=True)
jobs_vol = modal.Volume.from_name("acestep-lora-jobs", create_if_missing=True)

# acestep/training/path_safety.py resolves symlinks (os.path.realpath) on both
# the safe root and any user-provided path before comparing them. Modal
# Volumes mounted under /root/{checkpoints,jobs} each resolve to their own
# distinct canonical path (/__modal/volumes/vo-<id>/...) once symlinks are
# followed, which otherwise trips "Path escapes safe root" even though the
# path is legitimate. Compute the real common ancestor of both mounts at
# runtime (rather than hardcoding Modal's internal /__modal/volumes naming,
# or widening the safe root to "/", which would disable path-traversal
# protection entirely) and use exactly that as the safe root -- narrow,
# correct, and not dependent on Modal's internal path scheme.
import os
_SUBPROCESS_ENV = {
    **os.environ,
    "ACESTEP_SAFE_ROOT": os.path.commonpath([
        os.path.realpath("/root/checkpoints"),
        os.path.realpath("/root/jobs"),
    ]),
}


def _ensure_preprocessed(job_id: str, precision: str) -> str:
    """Run preprocessing if `<job_id>/tensors` doesn't exist yet.

    Expects `<job_id>/dataset.json` and `<job_id>/audio/` to already be
    populated (that's prep_service.py's job). Returns the tensor dir path.
    """
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
        "--dataset-dir", tensor_dir,       # unused placeholder, required by argparse
        "--output-dir", f"/root/jobs/{job_id}/lora_output",  # unused placeholder
    ]
    result = subprocess.run(cmd, cwd="/root", capture_output=True, text=True, env=_SUBPROCESS_ENV)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        raise RuntimeError(f"Preprocessing failed (returncode={result.returncode}):\n{result.stderr[-4000:]}")

    produced = list(Path(tensor_dir).glob("*.pt")) if Path(tensor_dir).is_dir() else []
    if not produced:
        # returncode==0 does NOT mean samples were processed -- train.py's
        # preprocess path exits 0 even at "Processed: 0/N" (e.g. every audio
        # path in dataset.json failed to resolve). Caught the hard way once
        # already: a silently-empty tensor dir let training "succeed" by
        # racing through 50 epochs of nothing in under a second.
        raise RuntimeError(
            f"Preprocessing reported success but produced 0 tensor files in {tensor_dir}.\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )
    jobs_vol.commit()
    return tensor_dir


def _run_training(job_id: str, epochs: int, save_every: int, precision: str,
                   lr: float = 1e-4, output_subdir: str = "lora_output",
                   target_modules: str = "", rank: int = 0, alpha: int = 0) -> dict:
    """Shared implementation for both calibrate() and train_lora().

    lr defaults to train.py's own default (1e-4) -- pass explicitly to
    override. output_subdir lets a differently-configured run (e.g. a
    higher-LR comparison) land in its own directory instead of clobbering
    an existing lora_output/.

    target_modules is a space-separated suffix list forwarded to train.py's
    --target-modules (empty = train.py's default: the four attention
    projections). PEFT suffix-matches module names, so adding
    "gate_proj up_proj down_proj" extends the adapter into the Qwen3MLP
    feed-forward layers.
    """
    import subprocess
    import sys

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
    ]
    if target_modules.strip():
        cmd += ["--target-modules", *target_modules.split()]
    # rank/alpha: 0 = keep train.py's defaults (64/128). Community consensus
    # for ~12-song datasets is rank 16-32 with alpha = 2x rank -- high rank
    # gives the adapter enough capacity to memorize the songs outright.
    if rank > 0:
        cmd += ["--rank", str(rank)]
    if alpha > 0:
        cmd += ["--alpha", str(alpha)]

    start = time.time()
    result = subprocess.run(cmd, cwd="/root", capture_output=True, text=True, env=_SUBPROCESS_ENV)
    elapsed = time.time() - start

    jobs_vol.commit()

    from pathlib import Path
    final_adapter = Path(output_dir) / "final" / "adapter_model.safetensors"
    adapter_produced = final_adapter.is_file()
    if result.returncode == 0 and not adapter_produced:
        # Same class of silent-success bug as the empty-tensor-dir case:
        # a training subprocess that iterates over 0 real batches (e.g. an
        # empty/broken dataset) can exit 0 having done nothing.
        result_returncode_note = "returncode=0 but no final adapter was written -- treat as failed"
    else:
        result_returncode_note = None

    return {
        "job_id": job_id,
        "epochs": epochs,
        "elapsed_seconds": elapsed,
        "seconds_per_epoch": elapsed / epochs if epochs else None,
        "returncode": result.returncode,
        "adapter_produced": adapter_produced,
        "warning": result_returncode_note,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/checkpoints": checkpoints_vol, "/root/jobs": jobs_vol},
    timeout=3600,
)
def calibrate(job_id: str, epochs: int = 20, save_every: int = 20, precision: str = "bf16",
              lr: float = 1e-4, output_subdir: str = "lora_output", target_modules: str = ""):
    """Short run to measure real per-epoch time + cost on A10G before committing to a full run."""
    result = _run_training(job_id, epochs, save_every, precision, lr, output_subdir, target_modules)

    A10G_PER_SECOND = 0.000306
    if result["seconds_per_epoch"]:
        per_epoch = result["seconds_per_epoch"]
        for target_epochs in (100, 200, 500, 800):
            est_seconds = per_epoch * target_epochs
            est_cost = est_seconds * A10G_PER_SECOND
            print(f"[estimate] {target_epochs} epochs -> {est_seconds/60:.1f} min, ${est_cost:.2f}")

    print(f"[calibrate] {result['epochs']} epochs in {result['elapsed_seconds']:.1f}s "
          f"({result['seconds_per_epoch']:.2f}s/epoch), returncode={result['returncode']}, "
          f"adapter_produced={result['adapter_produced']}")
    if result["warning"]:
        print(f"[WARNING] {result['warning']}")
    if result["returncode"] != 0 or not result["adapter_produced"]:
        print("STDOUT TAIL:\n", result["stdout_tail"])
        print("STDERR TAIL:\n", result["stderr_tail"])

    return result


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/checkpoints": checkpoints_vol, "/root/jobs": jobs_vol},
    timeout=6 * 3600,
)
def train_lora(job_id: str, epochs: int = 800, save_every: int = 25, precision: str = "bf16",
               lr: float = 1e-4, output_subdir: str = "lora_output", target_modules: str = "",
               rank: int = 0, alpha: int = 0):
    """Full training run for one job. Call with `.spawn()` from a web endpoint for async use."""
    result = _run_training(job_id, epochs, save_every, precision, lr, output_subdir, target_modules,
                           rank, alpha)
    print(f"[train_lora] {result['epochs']} epochs in {result['elapsed_seconds']/3600:.2f}h, "
          f"returncode={result['returncode']}, adapter_produced={result['adapter_produced']}")
    if result["warning"]:
        print(f"[WARNING] {result['warning']}")
    if result["returncode"] != 0 or not result["adapter_produced"]:
        print("STDOUT TAIL:\n", result["stdout_tail"])
        print("STDERR TAIL:\n", result["stderr_tail"])
    return result


# ---------------------------------------------------------------------------
# HTTP API for a frontend (deploy with `modal deploy modal/train_service.py`)
# ---------------------------------------------------------------------------

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def submit(job_id: str, epochs: int = 800, save_every: int = 25, precision: str = "bf16"):
    """Kick off training asynchronously; returns a call_id to poll via /status."""
    call = train_lora.spawn(job_id, epochs, save_every, precision)
    return {"call_id": call.object_id}


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def status(call_id: str):
    """Poll a training job. Returns {state: 'pending'|'done'|'error', result?}."""
    from modal.functions import FunctionCall

    call = FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
        return {"state": "done", "result": result}
    except TimeoutError:
        return {"state": "pending"}
    except Exception as exc:
        return {"state": "error", "error": str(exc)}
