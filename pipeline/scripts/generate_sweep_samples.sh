#!/usr/bin/env bash
# Batch A/B evaluation samples for a sweep: same tracks + same seed across
# every adapter config plus the base model, each into its own save_subdir
# (LoKr checkpoints hash-collide on output filenames in a shared dir).
#
# Usage:
#   ./generate_sweep_samples.sh launch <job_id> [modal_dir]   # detach all runs
#   ./generate_sweep_samples.sh fetch  <job_id> [dest_dir]    # download results
set -euo pipefail

PHASE="${1:?usage: generate_sweep_samples.sh launch|fetch <job_id>}"
JOB_ID="${2:?usage: generate_sweep_samples.sh launch|fetch <job_id>}"

# Sweep output_subdirs (from launch_sweep.sh) -- keep in sync.
CONFIGS=(
    lora_lr1e-4_anchor
    lora_lr1e-3
    lora_ffn_lr1e-4
    lora_ffn_lr1e-3
    lokr_lr1e-3
    lokr_lr3e-3
    lokr_lr1e-2
    lokr_ffn_lr3e-3
)
TRACKS=(0 5)   # two contrasting tracks; same seed everywhere
SEED=42

if [[ "$PHASE" == "launch" ]]; then
    MODAL_DIR="${3:-$(dirname "$0")/../../ACE-Step-1.5/modal}"
    cd "$MODAL_DIR"
    for t in "${TRACKS[@]}"; do
        # Base model reference (no adapter)
        modal run --detach generate_service.py::generate --job-id "$JOB_ID" \
            --track-index "$t" --no-use-lora --seed "$SEED" \
            --save-subdir "eval_base_t${t}"
        for cfg in "${CONFIGS[@]}"; do
            modal run --detach generate_service.py::generate --job-id "$JOB_ID" \
                --track-index "$t" --use-lora --seed "$SEED" \
                --lora-subpath "${cfg}/final" \
                --save-subdir "eval_${cfg}_t${t}"
        done
    done
    echo "[eval] $(( ${#TRACKS[@]} * (1 + ${#CONFIGS[@]}) )) generations detached."
elif [[ "$PHASE" == "fetch" ]]; then
    DEST="${3:-generations/$JOB_ID}"
    mkdir -p "$DEST"
    modal volume get --force acestep-lora-jobs "$JOB_ID/generations" "$DEST"
    echo "[eval] downloaded to $DEST"
else
    echo "unknown phase: $PHASE (want launch|fetch)" >&2
    exit 1
fi
