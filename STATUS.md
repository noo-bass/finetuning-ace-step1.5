# Project Status

Fine-tuning [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (a music
generation model) on artist-specific datasets using LoRA/DoRA/LoKr adapters,
with a full data-prep -> train -> generate pipeline running on Modal (A10G
GPUs). Two artists trained so far: Mk.gee (12 tracks) and Overmono (13
tracks, 11 after quality filtering).

Repo layout: see [README.md](README.md). This file is the running log of
what's been tried, what worked, what broke, and what's still open.

## Pipeline

`pipeline/modal/`:
- `prep_service.py` -- whisper transcription + LM auto-captioning, with a
  coverage gate (>=70% word overlap with the raw transcript, <=35% invented
  vocabulary) that retries with different seed/temperature before accepting
  a caption/lyrics pair.
- `train_service.py` -- LoRA/DoRA training (PEFT-based).
- `train_service_lokr.py` -- LoKr training (LyCORIS-based), separate output
  dir so it can run alongside a LoRA job on the same dataset.
- `generate_service.py` -- inference: loads the frozen turbo base model,
  optionally attaches a trained adapter, generates from caption+lyrics+bpm+
  keyscale. Supports a `caption_override` for testing alternate captions
  (e.g. Music Flamingo's) without touching lyrics/bpm/keyscale.

`pipeline/scripts/lora_data_prepare/` -- dataset-building helpers we added
on top of upstream's existing transcription scripts (`build_dataset_json.py`,
`format_lyrics_batch.py`, `format_lyrics_multiseed.py`,
`validate_lyrics_coverage.py`).

`pipeline/patches/upstream-acestep.patch` -- two small upstream fixes we
needed (torchaudio.load() soundfile fallback when torchcodec is broken;
`ACESTEP_SAFE_ROOT` env override for path-safety checks under Modal Volumes).

## Adapters trained (Mk.gee dataset, `mkgee-e2e-test`)

| Adapter | LR | Epochs | Final loss | Notes |
|---|---|---|---|---|
| LoRA | 1e-4 (default) | 200 | 0.9026 | Baseline. Audible effect vs. base model is real but subtle. |
| LoRA | 3e-4 | 200 | 0.8813 | Noisier, non-monotonic loss trajectory (bounced at epoch 100, 175) vs. the smooth 1e-4 descent. Modest gain, tested to see if a higher (but still sane) LR alone explains LoKr's bigger audible effect. |
| LoKr | 0.03 (buggy, see below) | 50 | 0.9694 | Loss plateaued/ticked back up epoch 40->50 -- classic sign of LR too high to converge cleanly. |

## Adapters trained (Overmono dataset, `overmono-e2e-test`)

| Adapter | LR | Epochs | Final loss | Notes |
|---|---|---|---|---|
| LoRA | 1e-4 | 200 | 0.8704 | Trained on 11/13 tracks after dropping 2 with blank captions/bpm/keyscale from a failed prep pass (see below). |

## Session 2026-07-09: captions-first pivot, MOSS-Music, sweep setup

**Decision: re-caption BEFORE the hyperparameter sweep, not after.** The
sweep's dialed-in settings should be tuned against the data pipeline the
product will actually ship, and the LM captioner is a known repeated
failure (see below). Captions are the text conditioning the adapter trains
against; tuning on known-bad captions optimizes for the wrong distribution.
Trade-off accepted: old mkgee baselines are no longer strictly controlled
comparisons, so the sweep includes an anchor run (LoRA 1e-4/200ep, default
targets -- the exact old-baseline config) to measure the caption swap itself.

**MOSS-Music (OpenMOSS, May 2026) replaces Music Flamingo as captioner
candidate.** Apache 2.0 (commercially usable, unlike Flamingo's OneWay
Noncommercial), ~9.05B params (~18GB bf16 -- runs on L40S; 8-bit would fit
A10G), and also does lyrics ASR + BPM + key + structure, so it could
eventually replace the whole whisper+LM prep stage. Self-reported SOTA on
MusicCaps/Song Describer with an LLM judge; no independent evals yet.
`modal/moss_caption_service.py` re-captions an existing job into a new job
(`mkgee-e2e-test` -> `mkgee-moss`) with ONLY the caption field swapped --
lyrics/bpm/keyscale/genre kept -- so caption quality stays a single
controlled variable. Dependency lessons baked into its image: needs
torchaudio + torchcodec==0.9.1 exactly (newer torchcodec wheels link CUDA 13
/ libnvrtc.so.13, absent from cu128 torch wheels), +cu128 torch wheels, and
transformers<5 (MOSS trust_remote_code files are 4.x-era); librosa fallback
around their load_audio as a backstop.

**Sweep infrastructure** (`pipeline/scripts/launch_sweep.sh`, 8 runs,
~$4.20): LoKr LR ladder 1e-3/3e-3/1e-2 @100ep (bracketing between sane
3e-4 and the accidental-but-audible 0.03), LoRA 1e-3 @100ep control
(is LoKr's effect just the hot LR?), FFN-extended `--target-modules`
runs testing the attention-only-cap hypothesis (the FFN blocks are
`Qwen3MLP`: gate_proj/up_proj/down_proj, suffix-matched by both PEFT and
LyCORIS -- upstream train.py already had the flag; our services now thread
it through), and the anchor run. `train_service_lokr.py` gained
`--output-subdir` so concurrent LoKr runs don't clobber; `generate_service.py`
gained `--save-subdir` so sweep-wide A/B generation doesn't hit the
LoKr empty-weights-hash filename collision
(`pipeline/scripts/generate_sweep_samples.sh` batches base+8 configs x 2
tracks at a fixed seed).

**Product note (from user's prior art-gen experience):** dataset captions
define the style; the user's prompt gets aligned to the caption vocabulary
at inference time. Maps to our stack as: MOSS captions at training time +
a prompt-rewrite layer in front of generation (building on
`generate_service.py`'s `caption_override`). Fold into UI design.

## Key findings this session (2026-07-06)

**LoKr's "wilder"/less-subtle character vs. LoRA is very likely explained
by a learning-rate bug, not a fundamental property of the technique.** Our
`train_service_lokr.py` hardcoded `lr=0.03`; `train_service.py` (LoRA) never
overrides `--lr` and falls back to `train.py`'s own default of `1e-4` --
which is also what the third-party Side-Step docs recommend for LoKr/DoRA
(1e-4 to 3e-4). That's a 300x LR discrepancy between the two adapters we were
comparing, not a controlled comparison. A same-LR LoRA run at 3e-4 showed a
real but much smaller behavioral shift (loss 0.8813, mildly noisier
trajectory) than LoKr's dramatic audible difference at 0.03 -- consistent
with the LR gap being the dominant factor.

**LoRA/LoKr both only target `q_proj`/`k_proj`/`v_proj`/`o_proj`** (attention
projections), never the FFN/MLP layers (`gate_proj`/`up_proj`/`down_proj`).
Untested hypothesis: if this architecture's timbral character lives partly
in the FFN blocks, attention-only adapters may be structurally capped in how
much "voice" they can shift regardless of LR or epoch count. Worth testing
by extending `target_modules`.

**With only 12-13 songs and `batch_size=1, gradient_accumulation=4`, 200
"epochs" is only ~600 real optimizer steps** (12 songs / 4 = 3 steps/epoch).
Training loss (flow-matching velocity-prediction MSE on latents) is not a
perceptual metric -- a real, measurable loss improvement doesn't have to
translate into an audible timbre shift; they're different distances in
different spaces.

**LoKr structural/inference verification required adapter-family-aware
code.** Our original verification (checking for a PEFT `PeftModel` wrapper
with non-zero `lora_`-named params) correctly caught real bugs elsewhere,
but incorrectly flagged genuinely-successful LoKr loads as failures, since
LoKr wraps via LyCORIS (`decoder._lycoris_net`), not PEFT. Fixed to detect
both families (`adapter_family: "peft_lora" | "lycoris"`).

**LoKr's internal weights-hash is always empty (`""`)**, unlike LoRA's which
hashes correctly. Since generated audio filenames are a deterministic hash
of the full generation-params dict (including this value), two different
LoKr checkpoints generated with the same seed/prompt collide on the same
output filename in the volume. Not yet root-caused upstream; downloads are
safe as long as each run's own freshly-returned path is used immediately,
not assumed stable across separate runs.

**LM auto-captioning has real, repeated quality problems**, not a one-off:
- Mk.gee "New Low" was captioned with the wrong vocalist gender (Mk.gee is
  male; caption said "female vocalist"). Caught and manually corrected early
  in the project.
- The Overmono prep run produced 2/13 tracks (`Calling Out`, `Vermonly`)
  with entirely blank caption/bpm/keyscale despite `is_instrumental=False`,
  and several more with clearly hallucinated lyrics (Turkish word fragments,
  repeated-digit garbage) or genre labels that don't match the artist at all
  ("J-core/gabber", "Latin dance-pop", "cloud rap" for a UK electronic/bass
  duo). Filtered dataset down to 11 tracks before training on it.
- Whisper+LM struggles more on production styles built from heavily
  chopped/pitched/looped vocal samples (Overmono) than on straightforward
  verse/chorus singing (Mk.gee) -- "more traditional genre" doesn't
  necessarily mean "easier for the labelling pipeline."

**NVIDIA Music Flamingo (8B, self-hosted) tested as an alternative
captioner** via its public Gradio Space
(`nvidia/music-flamingo`, `/infer` endpoint -- no formal HF Inference API
exists for this model, only the Space). Genuinely more accurate/detailed
genre and instrumentation calls for Mk.gee (correctly ID'd "indie
pop/dream pop" vs. our pipeline's "pop-punk"). For Overmono, Flamingo is
still wrong, just wrong differently ("dark trap/phonk" vs. our "J-core/
gabber") -- the user's own listening judgment favored Flamingo's read
here even though neither nails Overmono's actual genre. Raw + condensed
captions saved in `music_flamingo_captions/` (gitignored source captions
live outside the repo under that path locally -- see note below).
**License note:** Music Flamingo is NVIDIA OneWay Noncommercial -- research
use only, not for commercial use.

**ACE-Step's caption field is hard-truncated to 256 tokens** (including the
instruction template and metadata) in
`acestep/core/generation/handler/conditioning_text.py`. Flamingo's raw
output (2000-3000+ characters) would be silently truncated if passed
through as-is; captions were manually condensed to ~100-130 words (BPM/key
mentions stripped out, since those are already passed as separate fields
from the dataset and Flamingo's own tempo/key estimates sometimes
contradicted our dataset's values for the same track -- e.g. 107 BPM vs.
158 BPM for "Alesis").

## Open threads / not yet done

- **`flow_edit_morph` (reference-audio + text-guided steering) combined with
  a trained LoRA/DoRA adapter** -- discussed as a way to combine durable
  style (adapter weights) with structural grounding from a real reference
  track (flow-edit) and specific target description (caption), analogous to
  how strong reference conditioning has reduced LoRA dependence in image
  generation. Not yet implemented: `generate_service.py` only exposes
  `task_type="text2music"` today: no `src_audio`, `flow_edit_source_caption`,
  or `flow_edit_morph` params wired through.
- **DoRA is not yet supported in our training code.** PEFT supports it
  natively (`use_dora=True` on `LoraConfig`), but our `LoraConfig` dataclass
  and `lora_injection.py`'s `LoraConfig(...)` call don't thread it through
  yet. Discussed as the likely better choice for style fidelity specifically
  (decouples per-channel magnitude from direction), not yet tested head to
  head against LoRA at matched hyperparameters.
- **FFN/MLP target modules untested** -- current adapters only touch
  attention projections; extending to `gate_proj/up_proj/down_proj` could
  test whether that's why LoRA's audible effect stays subtle even at higher
  LR.
- Side-Step docs' Fisher-Information adaptive rank ("Preprocessing++") is a
  real technique but explicitly documented as capable of destabilizing
  *turbo*-variant training specifically -- and we only train turbo (to
  match our own inference schedule). Untested, flagged as higher-risk.
- Chunk-duration augmentation (`--chunk-duration 60`) and EMA
  (`--ema-decay 0.9999`) from Side-Step's docs are real, unused-by-us
  features worth adopting; not yet integrated.

## What's deliberately excluded from this repo

- `songs/` -- source audio, commercial/copyrighted tracks. Never pushed.
- `generations/` -- generated/derived audio outputs. Kept local-only.
- `ACE-Step-1.5/`, `Side-Step/` -- separate upstream git clones (see
  README.md for how to set them up locally). Our additions to ACE-Step-1.5
  are extracted into `pipeline/` above rather than committed into its own
  history, since we don't have (and don't need) push access to that
  upstream repo.
