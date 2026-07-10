# Emu: quality-tuning via tiny curated SFT (Meta, 2023)

Paper: https://arxiv.org/abs/2309.15807 ("Emu: Enhancing Image Generation
Models Using Photogenic Needles in a Haystack")

## The claim

A strongly pre-trained generative model already *can* produce exceptional
outputs -- it just isn't guided to do so consistently. Supervised
fine-tuning on a *tiny, ruthlessly curated* set of exceptional examples
("quality-tuning") restricts generation to the high-quality subset without
losing generality. No preference pairs, no reward model, no RL.

## The curation funnel (the actual work is here, not in training)

billions of images
  -> automatic filters (size/aspect/domain/etc.)     -> ~200K
  -> generalist human annotators                     -> 20K
  -> specialist annotators applying photography
     principles (composition, lighting, color/contrast,
     subject/background separation, subjective
     "compelling story")                             -> 2,000 final

Captions for the final 2,000 were manually written.

## Training numbers

- 2,000 images, batch 64, <=15K iterations, noise offset 0.1
- "Early stopping is important -- fine-tuning on a small dataset for too
  long results in significant overfitting"

## Results

- vs pre-trained: 82.9% visual-appeal win rate (91.2% on their prompt set)
- vs SDXL v1.0: 68.4-81.7% preferred
- Dataset-size ablation (win rate vs SDXL): 100 imgs -> 60.3%,
  1,000 -> 63.2%, 2,000 -> 67.0%. **Already winning at 100 examples.**
- Text faithfulness *improved*; no observed loss of generality
- Replicated on two other architectures (pixel diffusion, masked
  transformer) -- the effect is architecture-independent

## Mapping to this project

- The "100 curated examples already works" line is the load-bearing fact
  for the user-curation loop: one musician keeping 10-20 clips per round
  reaches Emu-scale curation within a few rounds.
- Emu curates *real* photos; our loop would curate *model generations* --
  closer to FluxAudio-S's expert iteration (see fluxaudio-s note). Mixing
  the 12 real tracks into every round approximates Emu's "ground in real
  quality" property and guards against self-training artifact drift.
- Their early-stopping warning matches our own observed
  best-checkpoint!=final behavior; dense checkpoints remain mandatory.
- Product-shape implication: the curation UI IS the training data factory.
  Emu spent specialist-annotator budget once; each musician is their own
  specialist annotator for their own style.
