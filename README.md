# music-model-finetunes

Artist-specific fine-tuning of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
(music generation model) using LoRA/DoRA/LoKr adapters, with a data-prep ->
train -> generate pipeline running on [Modal](https://modal.com) (A10G GPUs).

See [STATUS.md](STATUS.md) for the running log of experiments, findings, and
open threads.

## Repo layout

```
pipeline/
  modal/                     Our Modal services: prep, train (LoRA/DoRA),
                             train (LoKr), generate
  scripts/lora_data_prepare/ Dataset-building helpers (on top of upstream's
                             own transcription scripts)
  patches/                  Small fixes applied on top of upstream ACE-Step
music_flamingo_captions/    Captions generated via NVIDIA Music Flamingo,
                             used to compare against our own LM captioner
STATUS.md                  Experiment log, findings, open threads
```

This repo does **not** vendor ACE-Step-1.5 or Side-Step (a third-party
ACE-Step training tool) -- both are separate upstream projects. Set them up
alongside this repo:

```bash
git clone git@github.com:ace-step/ACE-Step-1.5.git
git clone <side-step-repo-url> Side-Step   # if you use Side-Step

# apply our small upstream fixes
cd ACE-Step-1.5 && git apply ../pipeline/patches/upstream-acestep.patch

# drop our services into place
cp ../pipeline/modal/*.py modal/
cp ../pipeline/scripts/lora_data_prepare/*.py scripts/lora_data_prepare/
```

## Pipeline usage

All three stages run as Modal apps against two persistent Volumes
(`acestep-checkpoints` for the shared base model, `acestep-lora-jobs` for
per-job data/adapters/generations):

```bash
# 1. Upload raw audio for a new job
modal volume put acestep-lora-jobs ./my_songs <job_id>/audio

# 2. Prep: whisper transcription + LM captioning (coverage-gated)
modal run modal/prep_service.py::prep_job --job-id <job_id>

# 3. Train a LoRA (or DoRA via train_service.py's dora flag once added --
#    see STATUS.md open threads)
modal run --detach modal/train_service.py::train_lora \
    --job-id <job_id> --epochs 200 --lr 0.0001

# ...or a LoKr, into a separate output dir so it can run alongside the LoRA job
modal run --detach modal/train_service_lokr.py::train_lokr \
    --job-id <job_id> --epochs 50 --lr 0.0001

# 4. Generate with the trained adapter
modal run modal/generate_service.py::generate \
    --job-id <job_id> --track-index 0 --use-lora --lora-subpath lora_output/final
```

Always sanity-check a new dataset's `dataset.json` before training on it --
this project has repeatedly hit LM-captioning failures (blank
caption/bpm/keyscale, hallucinated lyrics, wrong genre/vocalist-gender
labels) that silently degrade training data quality. See STATUS.md for
specifics.

## What's not in this repo

Source audio (`songs/`) and generated outputs (`generations/`) are
gitignored -- the former is commercial/copyrighted material, the latter is
large derived binary data. Re-source your own audio and regenerate as
needed via the pipeline above.
