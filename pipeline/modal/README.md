# ACE-Step 1.5 LoRA-as-a-service: backend architecture

Two Modal apps take a user's uploaded songs to a trained, listenable LoRA. A
third (inference) is not built yet -- see [Status](#status) at the bottom.

```
 upload audio          prep_service.py            train_service.py
 ----------->  acestep-lora-jobs/<job_id>/audio/*
                       |
                       |  whisper transcribe -> LM format (coverage-gated)
                       v
               acestep-lora-jobs/<job_id>/dataset.json
                       |
                       |  train_service auto-preprocesses if no tensors yet
                       v
               acestep-lora-jobs/<job_id>/tensors/*.pt
                       |
                       |  train.py fixed (LoRA, bf16, A10G)
                       v
               acestep-lora-jobs/<job_id>/lora_output/final/
```

Both services are plain Modal apps (`modal/prep_service.py`,
`modal/train_service.py`) that shell out to this repo's own
`acestep.training_v2` pipeline and `train.py` CLI -- the same code path
proven to work locally, just running on a real CUDA GPU instead of Apple
MPS. Nothing about the training/preprocessing logic itself was changed for
Modal; only the environment around it.

## Storage (two persistent Modal Volumes)

| Volume | Contents |
|---|---|
| `acestep-checkpoints` | Base model, shared by every job. Layout: `checkpoints/checkpoints/{acestep-v15-turbo,acestep-5Hz-lm-1.7B,vae,Qwen3-Embedding-0.6B}` (nested `checkpoints/checkpoints/` because the upload command's trailing-slash semantics put the source dir *itself* under the destination -- see gotcha below). |
| `acestep-lora-jobs` | Per-job data, keyed by `<job_id>`: `audio/` (input + labels), `tensors/` (preprocessed), `lora_output/` (checkpoints + final weights). |

One-time setup:
```bash
modal volume create acestep-checkpoints
modal volume create acestep-lora-jobs
modal volume put acestep-checkpoints ../checkpoints checkpoints
```

Per-job input (until a real upload endpoint exists -- see Status):
```bash
modal volume put acestep-lora-jobs /path/to/songs <job_id>/audio
```
Each audio file may optionally be paired with a `{name}.lyrics.txt` sidecar
to skip whisper transcription for that track.

## Running a job

```bash
# 1. Transcribe + label (whisper -> LM format, coverage-validated, writes dataset.json)
modal run modal/prep_service.py::prep_job --job-id my-job

# 2. Train (auto-preprocesses from dataset.json if tensors don't exist yet)
modal run modal/train_service.py::train_lora --job-id my-job --epochs 800

# Or calibrate first to get a real per-epoch timing/cost estimate before committing:
modal run modal/train_service.py::calibrate --job-id my-job --epochs 20
```

Measured on A10G against the 12-track demo dataset used to build this:
**0.86s/epoch**, i.e. ~11 minutes and ~$0.21 for a full 800-epoch run (vs.
~107s/epoch, ~24 hours on an M-series Mac's MPS backend for the same job --
see [Known gotchas](#known-gotchas)).

## HTTP API (for a frontend)

Deploy to get persistent HTTP endpoints instead of one-off `modal run` calls:
```bash
modal deploy modal/prep_service.py
modal deploy modal/train_service.py
```

Both services expose the same two-endpoint async pattern:

| Endpoint | Method | Params | Returns |
|---|---|---|---|
| `/submit` | POST | `job_id` (+ `epochs`/`save_every`/`precision` for train) | `{"call_id": "..."}` |
| `/status` | GET | `call_id` | `{"state": "pending"}` or `{"state": "done", "result": {...}}` or `{"state": "error", "error": "..."}` |

A frontend flow is: upload audio to the jobs volume -> POST prep `/submit` ->
poll prep `/status` until done -> POST train `/submit` -> poll train
`/status` until done -> LoRA weights are at
`acestep-lora-jobs/<job_id>/lora_output/final/`.

`result` for a training call_id includes `elapsed_seconds`,
`seconds_per_epoch`, and `returncode` -- useful for showing progress/ETA in
a UI even though it's a single blocking call under the hood (no
mid-training progress events yet; see Status).

## Known gotchas

These cost real debugging time locally -- documented here so they don't
resurface silently in the cloud version.

- **fp16 auto-precision silently produces NaN.** This repo's `detect_gpu()`
  (`acestep/training_v2/gpu_utils.py`) auto-selects `fp16` for MPS and
  `bf16` for CUDA/XPU when `--precision auto` is used. On MPS, fp16
  overflowed the Qwen3 text encoder into NaN on every single preprocessed
  tensor, with no error raised -- training would proceed on garbage data.
  Both services here always pass `--precision bf16` explicitly. If you add
  a precision knob to a frontend, don't expose "auto" for MPS-adjacent
  paths without re-verifying this.
- **fp16 GradScaler crash is a separate bug from the above.** Lightning's
  AMP `GradScaler` cannot unscale true fp16 gradients
  (`ValueError: Attempting to unscale FP16 gradients`) -- another reason
  `--precision bf16` is hardcoded rather than left to the trainer's own
  default.
- **Path-safety sandbox.** `acestep/training/path_safety.py` restricts
  `--dataset-dir`/`--output-dir` to resolve under the process's cwd at
  import time. Inside the Modal container this is `/root`, which is why
  everything lives under `/root/jobs`/`/root/checkpoints` -- don't mount
  volumes elsewhere without checking this.
- **`--preprocess` requires a real subcommand.** `train.py --preprocess
  ...` alone launches the interactive wizard (and hangs/aborts under a
  non-interactive shell) because `_has_subcommand()` only recognizes
  `vanilla`/`fixed`/`estimate`/`--help`. Always pair it:
  `train.py fixed --preprocess ...`. argparse also still requires
  `--dataset-dir`/`--output-dir` even though the preprocess path never
  touches them -- `_ensure_preprocessed()` in `train_service.py` passes
  harmless placeholders.
- **`modal volume put <vol> <src> /` does not flatten.** A trailing slash
  on the remote path uploads the source directory *by its own name* under
  that path, not its contents. This is why the checkpoint volume has an
  extra `checkpoints/` nesting level (see Storage table above) -- fixing it
  means either re-uploading with a different remote path or (what we did)
  just pointing `--checkpoint-dir` at the actual resulting path.
- **`dataset.json` filenames must be relative to the JSON's own directory,
  not the audio directory.** `discover_audio_files()` resolves a bare
  `"filename"` entry against `dataset.json`'s parent directory. Since
  `prep_service.py` writes `dataset.json` at `<job_id>/dataset.json` but
  audio lives at `<job_id>/audio/`, storing just the bare filename made
  every path resolve to nothing -- preprocessing silently processed 0/12
  files and still exited 0, and training then raced through 50 "epochs" of
  an empty dataset in under a second, also exiting 0. Nothing looked
  wrong until the tensor dir was checked by hand and found empty. Fixed by
  storing `"audio/<filename>"`. **This bit us twice** (once locally, once
  on Modal) because a bare filename looks correct at a glance -- if you
  write another dataset.json producer, sanity-check the resolved paths,
  don't trust returncode 0.
- **Corollary: `train.py`'s preprocess/train subprocess exiting 0 does not
  mean it did anything.** Both `_ensure_preprocessed()` and `_run_training()`
  in `train_service.py` now explicitly verify real output exists (tensor
  `.pt` files produced; `lora_output/final/adapter_model.safetensors`
  written) rather than trusting `returncode == 0`, and print stdout on
  success too, not just on failure -- the original code only surfaced
  stdout/stderr when returncode was non-zero, which is exactly how the bug
  above went unnoticed on the first end-to-end test.
- **`torchcodec` is a real (missed) dependency for training-side audio
  loading**, not just a local-env quirk -- `acestep/training/dataset_builder_modules/preprocess_audio.py`
  calls `torchaudio.load()` directly, and since torchaudio 2.9 that
  function is *hard-wired* to TorchCodec's `AudioDecoder` internally --
  its own `backend=` parameter is accepted but silently ignored, so
  there is no way to fall back to soundfile/sox from the caller's side
  (unlike `acestep/training/dataset_builder_modules/audio_io.py`, which
  does implement its own try/except fallback and isn't affected). Added
  `torchcodec` to `train_service.py`'s image -- it's listed in this repo's
  own `pyproject.toml`/`requirements.txt` but wasn't carried over.
- **`torchcodec`'s prebuilt wheel hard-requires CUDA 13's `libnvrtc.so.13`,
  independent of whatever CUDA version torch itself uses.** Adding
  torchcodec produced `OSError: libnvrtc.so.13: cannot open shared object
  file`. The obvious fix -- pin torch to the exact `+cu128` build this
  repo's own `pyproject.toml` uses for linux x86_64 -- did **not** help;
  the error was identical afterward, since it's torchcodec's own compiled
  extension that's linked against nvrtc, not something inherited from
  torch's bundled libs. No CUDA-13-nvrtc pip package exists to install
  standalone either (`nvidia-cuda-nvrtc-cu13` is an unpublished placeholder
  as of this writing). Real fix: `torchaudio.load()` has been hard-wired to
  TorchCodec's `AudioDecoder` since torchaudio 2.9 -- its own `backend=`
  parameter is accepted but silently ignored -- so there's no way to
  configure around a broken torchcodec install from the calling code. Added
  a soundfile fallback directly in
  `acestep/training/dataset_builder_modules/preprocess_audio.py`'s
  `load_audio_stereo()`, mirroring the fallback pattern the sibling
  `audio_io.py` module already had for its own (different) audio-loading
  functions. The `+cu128` torch pin was kept anyway since it doesn't hurt
  and matches the project's own pyproject.toml, but it was not the fix.
- **Modal Volumes trip the path-safety sandbox because they resolve to a
  different real path once symlinks are followed.** `path_safety.py`
  compares `os.path.realpath()` of both the safe root and any user-provided
  path. `/root/jobs/<job_id>/tensors` looks like it's under `/root`, but
  its realpath is actually `/__modal/volumes/vo-<id>/...` -- and the
  checkpoints and jobs volumes each resolve to their *own* distinct
  `vo-<id>` path, so there's no single shared prefix under `/root` either.
  `set_safe_root()` already existed for exactly this kind of override but
  had no way to invoke it short of importing the module and calling it
  before `train.py`'s own code runs -- not practical when `train.py` is
  launched as a fresh subprocess. Added an `ACESTEP_SAFE_ROOT` env var read
  at import time as a minimal opt-in hook. **First attempt at using it was
  wrong and got (correctly) blocked**: setting it to `"/"` "to be safe
  since the container is already isolated" both disabled path-traversal
  protection entirely *and* exposed a real bug in `safe_path()` itself
  (`root + os.sep` becomes `"//"` when `root == "/"`, which no normal path
  ever starts with, so it rejected everything regardless). Fixed properly
  instead: `train_service.py` computes `os.path.commonpath()` over both
  mounted volumes' *actual resolved real paths* at runtime and uses that
  as `ACESTEP_SAFE_ROOT` -- narrow, correct, and doesn't touch
  `path_safety.py`'s logic at all. Don't reach for "disable the check
  because the outer environment is already sandboxed" as a shortcut; compute
  the real boundary instead.
- **This whole class of bug (path resolution, missing torchcodec, CUDA
  mismatch) only surfaced because the training service's own audio-loading
  path had never actually been exercised on Modal before this end-to-end
  test.** Every prior calibration run reused a tensor dataset that had
  been preprocessed *locally* and uploaded directly, skipping
  `_ensure_preprocessed()` entirely. A calibration/smoke test that reuses
  cached artifacts is not the same as an end-to-end test from raw audio --
  don't mistake one for the other again.
- **`modal run --detach` survives *local* client disconnection, including a
  `kill -INT` on the local `modal` CLI process.** That's the entire point
  of `--detach` -- but it means stopping a long training run for real
  requires `modal app stop <app-id> --yes` against the remote app, not just
  killing the local process. Confirm via `modal container list` (should go
  empty) or `modal app list` (state should read `stopped`), not just "the
  local `ps` entry is gone".
- **`modal app stop` is a hard kill, not a graceful interrupt -- the
  training process never gets a chance to run its own
  `except KeyboardInterrupt` handler.** Locally, sending the training
  process a real `SIGINT` lets it catch the interrupt and write a
  `final/` export before exiting. `modal app stop` just terminates the
  container outright, so `final/` is left containing whatever the *last
  run* wrote there (stale!), while the interrupted run's actual progress
  only exists as periodic `checkpoints/epoch_N_loss_X/` directories from
  `--save-every`. Concretely: after stopping an 800-epoch run at epoch 200,
  `lora_output/final/adapter_model.safetensors` still held the *previous*
  50-epoch run's weights (same content hash, same generated audio) --
  caught by the `lora_weights_hash` field in generation metadata not
  changing between runs. **Load the specific
  `checkpoints/epoch_N_loss_X/` directory directly (same file layout as
  `final/`) when generating from a run you stopped via `modal app stop`,
  not `final/`.**
- **Loading a LoRA from a `checkpoints/epoch_N_loss_X.XXXX/` directory
  fails with `KeyError: module name can't contain "."`.** `add_lora()`
  derives the PEFT adapter name from the checkpoint directory's basename
  when no explicit name is passed, and torch's `nn.Module.add_module`
  rejects any name containing a `.` -- checkpoint dirs are named like
  `epoch_200_loss_0.9026` (a literal decimal point in the loss value).
  `generate_service.py` now always passes an explicit
  `adapter_name="loaded_adapter"` to `add_lora()` rather than relying on
  the auto-derived name, sidestepping this entirely regardless of
  directory naming.
- **`get_lora_weights_hash()` always returns `""` for LyCORIS (LoKr/LoHA)
  adapters, unlike PEFT LoRA where it correctly hashes the weight file
  bytes.** Since the generated audio's output filename is a deterministic
  hash of the full generation params dict (including this weights hash),
  two different LoKr checkpoints generated with identical seed/prompt
  collide on the same output filename and **overwrite each other in the
  volume** (though each run's own returned path always points at what it
  itself just wrote, so no result is silently served stale -- just be
  careful not to assume an old path is still what you think it is once a
  second LoKr generation has run). Root cause is upstream in
  `acestep/core/generation/handler/lora/lifecycle.py`'s registry-population
  path for LyCORIS adapters, not traced further since it doesn't affect
  correctness of the loaded weights themselves (verified independently via
  `decoder._lycoris_net` + non-zero adapter param counts + measurably
  different output audio). Workaround if you need to keep multiple LoKr
  generations from colliding: pass a distinct `save_dir` per checkpoint
  you're comparing, don't rely on the auto UUID for uniqueness.
- **LoRA and LoKr, as configured in this project, are not an
  apples-to-apples comparison -- expect a real difference in audible
  character even at matching loss values.** Two concrete, measured
  asymmetries: (1) our LoKr runs enable `lokr_weight_decompose=True`
  (DoRA-style magnitude decomposition) by default per the project's own
  docs, while the LoRA runs use plain LoRA with no DoRA -- DoRA changes the
  update mechanism, not just its parameterization. (2) LoKr's actual
  trainable parameter count came out to 1,419,648 vs LoRA's 44,040,192 at
  our chosen rank/dim settings -- **~31x fewer parameters** for a similar
  final loss. Both adapters were structurally and behaviorally verified as
  correctly applied at inference (PEFT/`PeftModel` wrapping vs
  `decoder._lycoris_net` presence, non-zero adapter parameters matching
  the trained counts, and measurably non-identical output audio vs the
  base model in both cases) -- there is no evidence of an inference-time
  loading bug for either adapter type. A visibly "bigger" or "wilder" LoKr
  effect for similar loss is consistent with it distributing far fewer
  parameters across the same Kronecker-tiled footprint of each weight
  matrix (coarser, more global perturbations) versus LoRA's larger
  low-rank space enabling smoother, more localized corrections -- not
  proof that one is broken and the other isn't.
- **MLX DiT silently breaks LoRA inference on Apple Silicon (local Gradio
  UI only, doesn't apply to Modal).** On MPS, this repo auto-converts the
  DiT to a native MLX decoder by default, and the LoRA-loading code has no
  awareness of that path -- it only patches the PyTorch model object. A
  LoRA loads with no error but has zero audible effect if MLX DiT stays
  active. Uncheck "Use MLX DiT" in the model-loading panel, or don't
  pre-init via `--init_service True` if you want to control it before
  load. N/A on Modal since these services never touch MLX.
- **A Metal command-buffer assertion crash was observed locally**
  (`-[IOGPUMetalCommandBuffer validate]:214: failed assertion 'commit an
  already committed command buffer'`) while generating with the 5Hz LM and
  loading a LoRA in the same session on MPS. Not investigated further
  since it's an MPS/Gradio-only path unrelated to the Modal services; flag
  if it recurs.
- **Dataset size guidance** (from this project's own docs, not Modal-
  specific): 5+ songs minimum recommended; epoch count should scale
  inversely with dataset size (~100 songs -> ~500 epochs, 10-20 songs ->
  ~800 epochs); very small datasets benefit from higher LoRA dropout
  (0.2-0.3 vs default 0.1).
- **Coverage-gated retry loop: best-attempt selection must rank by
  invented-word ratio first, not coverage.** An earlier version of both
  `prep_service.py` and `scripts/lora_data_prepare/format_lyrics_multiseed.py`
  picked whichever passing attempt had the *highest coverage*, with no
  regard for its invented-word ratio -- so an attempt with 100% coverage
  but 7% fabricated content could beat one with 99% coverage and 1%
  fabricated content. Fixed to rank `(invented, -coverage)` ascending
  (lowest fabrication wins, coverage as tiebreaker only). Caught during
  the first cold end-to-end test on real Modal infra -- if you copy this
  retry pattern elsewhere, copy the fixed comparator, not the original.
- **Early-stop threshold was unreachable in practice.** The retry loop
  originally stopped as soon as an attempt hit `invented == 0.0`, but the
  LM almost never returns literally zero fabricated words -- so every
  track burned all `MAX_ATTEMPTS` (4x the necessary GPU time/cost) even
  when attempt 1 was already excellent. Changed to stop once
  `coverage >= 0.95 and invented <= 0.05`.
- **Garbled non-English token injection observed on the CUDA `pt` LM
  backend, not seen locally on MLX.** Running `format_sample` on Modal
  (`backend="pt"`, A10G) produced stray fragments of Korean/Chinese/Polish
  characters spliced into otherwise-English generated lyrics (e.g.
  `losing somebody else즉 of the headless`), across most attempts and
  multiple tracks. Not seen at all when running the same LM locally via
  the MLX backend on Apple Silicon. Not yet root-caused -- possibly a
  dtype, sampling, or attention-implementation difference specific to the
  PyTorch-native generation path in `acestep/llm_inference.py`. The
  coverage gate still catches the worst cases (an attempt riddled with
  garbage would fail the invented-ratio check), but don't treat `pt`-backend
  LM output as trustworthy without spot-checking until this is understood.
  Worth trying `backend="vllm"` (needs Triton in the image) as a
  comparison point before assuming it's unfixable.

## Status

Built and verified end-to-end on the 12-track demo job:
- [x] `prep_service.py` -- whisper + LM labelling with coverage-gated retries
- [x] `train_service.py` -- LoRA training, calibrated at 0.86s/epoch on A10G
- [x] HTTP submit/status endpoints on both, ready for a frontend to call
- [ ] **Inference service** -- not built. Needs product decisions before
      building for real: how the frozen base model stays warm across
      requests, how per-user LoRA swapping works safely (multi-tenancy --
      don't let user A's request see user B's LoRA), and where generated
      audio gets stored/served back.
- [ ] **Real upload endpoint** -- jobs currently take audio via `modal
      volume put`; a frontend needs an actual upload API (presigned URL or
      direct multipart to a Modal endpoint).
- [ ] **Auth / rate limiting / abuse controls** -- required before this is
      public-facing; not addressed here since costs are currently
      near-zero and low-volume.
- [ ] **Job metadata / listing** -- there's no way yet to list a user's
      jobs or their status without already knowing the `call_id`; a
      frontend will likely want a small database (or a Modal Dict) mapping
      `user_id -> [job_id, ...]` and persisting `call_id`s past the
      lifetime of the submitting process.
