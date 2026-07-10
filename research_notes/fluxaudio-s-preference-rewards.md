# FluxAudio-S: reward conditioning + expert iteration for text-to-music (2026)

Paper: https://arxiv.org/abs/2606.21670 ("Improving Text-to-Music
Generation with Human Preference Rewards")

Small model (120M Flux-style flow-matching transformer, mel-VAE latents,
44.1kHz, ~10s clips, 25 Euler steps), which makes its numbers a lower
bound for what a bigger base like ACE-Step could get from the same recipe.

## Three stages, with the ablation verdict up front

**Expert iteration was the dominant contributor** (-0.0362 FAD-CLAP);
the 2K-pair DPO/CRPO stage added gains "within paired-t noise" (p>0.05).
Reward conditioning was the second useful piece. At small preference-data
scale: curate > condition > DPO.

## Stage 1: reward-conditioned SFT ("quality as a second CFG axis")

- Every training example gets a scalar quality score (from TuneJury, an
  open CLAP+MERT twin pairwise ranker: 70.3% held-out accuracy on ~22K
  public preference pairs).
- Score injected via Fourier features -> MLP (zero-init final proj),
  null-dropped at p=0.1 exactly like text CFG.
- Swept 5 injection architectures; interestingly train-time and
  inference-time winners DIFFERED (GlobalAdaLN for training forward,
  InputAdd for inference forward).
- At inference: guide jointly on text + high score. Their settings:
  s=5.0, w=4.0. "Quality" becomes a slider.
- SFT scale: 200K updates, lr 1e-4, batch 64, bf16, ~32h on ONE RTX A5000.

## Stage 2: expert iteration (the winner)

- Generate ~630 clips from the SFT checkpoint (s=2.0).
- Rank by equal-weight z-blend of ranker reward + CLAP-text similarity.
- Keep top decile (64 clips, mean reward +1.05).
- Fine-tune 30K steps @ lr 1e-5, then 5K-step polish on the same subset
  @ lr 1e-6. Single round.

## Stage 3: CRPO preference pass (the noise)

- ~2,000 pairs (high-CLAP vs low-CLAP per prompt), beta=2000,
  lambda_FM=1.0 (plain flow-matching loss on winners kept in the mix),
  lr 1e-6, 5K updates. Not statistically significant over stage 2.

## Mapping to this project

- Validates the user's proposed loop almost verbatim, in music, on a
  flow-matching model: generate a batch -> keep the top slice -> fine-tune
  on keepers. 64 keepers moved the headline metric; a musician keeping
  10-20 clips/round for 3-4 rounds is the same order of magnitude.
- Their ranker-based selection can be replaced by the musician's ears
  (strictly better signal for a personal-style task; TuneJury usable as a
  pre-filter so the human only auditions plausible candidates).
- Reward conditioning port for us: a "style-match: high/low" caption token
  (discrete version) is a days-scale trainer change and extracts signal
  from REJECTED clips too -- worth pairing with the curation loop since
  the same ratings feed both.
- Their beta=2000 / lr 1e-6 / lambda_FM=1.0 numbers are the starting
  hyperparameters if we ever do the DPO stage -- but their own ablation
  says don't start there.
- Caveat: no human eval in the paper (TuneJury proxy only), and clips are
  10s -- short-window curation may transfer imperfectly to 60s+ music.
