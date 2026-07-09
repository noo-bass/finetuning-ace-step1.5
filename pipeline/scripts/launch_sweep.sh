#!/usr/bin/env bash
# Hyperparameter sweep for ACE-Step adapters on a prepped job.
#
# Matrix (see STATUS.md "Key findings" for the rationale):
#   - LoKr LR ladder 1e-3 / 3e-3 / 1e-2 @ 100 epochs -- bracket between the
#     sane-but-subtle 3e-4 and the accidental-but-audible 0.03
#   - LoRA @ 1e-3, 100 epochs -- control: is LoKr's effect just the hot LR?
#   - FFN-extended target modules (attention + Qwen3MLP gate/up/down) for
#     LoRA at both matched-baseline (1e-4/200ep) and hot (1e-3/100ep)
#     settings, plus one LoKr FFN run -- tests the "attention-only adapters
#     are structurally capped on timbre" hypothesis
#   - Anchor: LoRA 1e-4/200ep with default targets -- the exact old-baseline
#     config, so old-captions vs. new-captions is itself measurable
#
# ~1000 total epochs @ ~13.5s/epoch on A10G ~= 3.8 GPU-hours ~= $4.20.
#
# Usage: ./launch_sweep.sh <job_id> [path-to-ACE-Step-1.5/modal]
set -euo pipefail

JOB_ID="${1:?usage: launch_sweep.sh <job_id> [modal_dir]}"
MODAL_DIR="${2:-$(dirname "$0")/../../ACE-Step-1.5/modal}"
FULL_TARGETS="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

cd "$MODAL_DIR"

echo "[sweep] warming up preprocessing for $JOB_ID (foreground, ~5 min)..."
modal run train_service.py::calibrate --job-id "$JOB_ID" --epochs 2 \
    --output-subdir calibrate_warmup

echo "[sweep] tensors ready -- launching detached runs..."

# LoRA runs
modal run --detach train_service.py::train_lora --job-id "$JOB_ID" \
    --epochs 200 --lr 1e-4 --output-subdir lora_lr1e-4_anchor
modal run --detach train_service.py::train_lora --job-id "$JOB_ID" \
    --epochs 100 --lr 1e-3 --output-subdir lora_lr1e-3
modal run --detach train_service.py::train_lora --job-id "$JOB_ID" \
    --epochs 200 --lr 1e-4 --output-subdir lora_ffn_lr1e-4 \
    --target-modules "$FULL_TARGETS"
modal run --detach train_service.py::train_lora --job-id "$JOB_ID" \
    --epochs 100 --lr 1e-3 --output-subdir lora_ffn_lr1e-3 \
    --target-modules "$FULL_TARGETS"

# LoKr runs
modal run --detach train_service_lokr.py::train_lokr --job-id "$JOB_ID" \
    --epochs 100 --lr 1e-3 --save-every 50 --output-subdir lokr_lr1e-3
modal run --detach train_service_lokr.py::train_lokr --job-id "$JOB_ID" \
    --epochs 100 --lr 3e-3 --save-every 50 --output-subdir lokr_lr3e-3
modal run --detach train_service_lokr.py::train_lokr --job-id "$JOB_ID" \
    --epochs 100 --lr 1e-2 --save-every 50 --output-subdir lokr_lr1e-2
modal run --detach train_service_lokr.py::train_lokr --job-id "$JOB_ID" \
    --epochs 100 --lr 3e-3 --save-every 50 --output-subdir lokr_ffn_lr3e-3 \
    --target-modules "$FULL_TARGETS"

echo "[sweep] 8 runs detached. Watch: modal app list / modal app logs <id>"
