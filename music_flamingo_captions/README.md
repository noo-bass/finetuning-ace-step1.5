# Music Flamingo Captions

Generated via NVIDIA's `nvidia/music-flamingo` Gradio Space
(https://huggingface.co/spaces/nvidia/music-flamingo), called through its
`/infer` API (not a hosted Inference API -- Music Flamingo has none; see
`raw_captions.json` for the exact request/response mechanics).

Prompt sent to Flamingo for every track:
> "Describe this track in full detail - tell me the genre, tempo, and key,
> then dive into the instruments, production style, and overall mood it
> creates."

Raw, full responses are in `raw_captions.json`. Below are the condensed
versions actually used to prompt ACE-Step's base model (BPM/key stripped
out -- those are passed as separate fields already sourced from our own
dataset.json, so repeating Flamingo's own tempo/key estimate in the caption
text would just contradict them).

## Mk.gee -- Alesis

**Flamingo genre call:** Indie Pop / Dream Pop (vs. our pipeline's "energetic
pop-punk")

**Condensed caption used for generation:**
> A dreamy indie pop/dream pop track with hazy, reverb-drenched synths, clean
> electric guitar arpeggios processed with chorus and delay, and understated
> electronic drums. Male lead vocals sit in a smooth, breathy baritone/tenor
> range, heavily treated with reverb and delay. Structure follows verse ->
> pre-chorus -> chorus with instrumental breaks. Reflective, cinematic,
> melancholic mood typical of early-2020s indie dream pop.

## Mk.gee -- New Low

**Flamingo genre call:** experimental electronic / glitch-hop / avant-garde
electronica / downtempo hip-hop

**Condensed caption used for generation:**
> An experimental electronic track fusing glitch-hop rhythms with ambient
> textures, between avant-garde electronica and downtempo hip-hop. Spacious
> reverb-drenched synth pads, deep sub-bass, glitchy stuttered hi-hats and
> syncopated drum loops panned widely. Heavily processed, pitch-shifted male
> vocal fragments used as atmospheric texture rather than clear lyrics.
> Mysterious, slightly unsettling, late-night mood blending lo-fi hip-hop
> grooves with immersive ambient design.

## Overmono -- Cold Blooded

**Flamingo genre call:** dark trap / phonk-infused trap (vs. our pipeline's
hallucinated "J-core and happy hardcore, gabber-style kick") -- note neither
of these is actually correct for Overmono's real genre identity (UK
electronic/bass/techno); Flamingo is just wrong in a different way here.

**Condensed caption used for generation:**
> A dark-toned trap track leaning into phonk aesthetics with dark ambient
> textures. Deep sub-bass 808 kicks, crisp snares, rapid half-time hi-hat
> rolls, distorted gritty synth bass, and eerie floating pad synths.
> Pitch-shifted, reverb-drenched male vocal snippets used as rhythmic stabs
> rather than melodic leads. Brooding, cinematic, paranoid mood suited to
> underground/late-night atmosphere.

## Overmono -- Feelings Plain

**Flamingo genre call:** R&B-pop with trap rhythms (vs. our pipeline's
hallucinated "Latin dance-pop" with garbled Turkish-language lyric artifacts)

**Condensed caption used for generation:**
> A dreamy R&B-pop track blending contemporary trap rhythms with smooth
> electronic textures. Lush reverberant synth pads, melodic arpeggiated
> leads, deep sub-bass, and trap-style hi-hats with side-chain compression.
> Male and female vocals heavily treated with auto-tune, reverb and delay,
> doubling in unison during choruses. Reflective, yearning, late-night mood
> typical of genre-fluid contemporary pop.

## Caveats

- Flamingo produces long, technically fluent descriptions (specific chord
  progressions, exact BPM figures) regardless of whether they're accurate --
  same "fluency isn't accuracy" trap as our own LM-based captioning. Treat
  confident-sounding detail with the same skepticism either way.
- ACE-Step's text encoder truncates the caption+instruction+metadata prompt
  to 256 tokens (`acestep/core/generation/handler/conditioning_text.py:131`).
  Flamingo's raw output is 2000-3000+ characters and would be silently
  truncated if passed through as-is -- the condensed versions above were
  written to fit comfortably within that budget while preserving genre,
  instrumentation, vocal character, and mood.
- Lyrics, BPM, keyscale, and timesignature were left untouched from the
  original dataset.json for all four generations -- caption is the only
  variable that changed, to isolate its effect cleanly.
