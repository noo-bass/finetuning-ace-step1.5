# Rating app

Taste-teaching loop for fine-tuning a music model: rounds of blinded audio
clips get rated (Top pick / Keep / Pass + an independent Artifacts flag), and
the ratings drive training. Supersedes the static `rate.html` prototype while
staying compatible with `pipeline/scripts/deal_round.py` output.

## Layout

```
app/
  backend/     FastAPI API (Python 3.11, uv)
  frontend/    Vite + React + TypeScript UI
  data/rounds/ round dirs in deal_round.py format:
               <roundName>/audio/*.flac|wav + mapping.json (+ ratings.json once rated)
```

`mapping.json` (the blind-id -> true-config mapping) is never exposed through
the API. The UI only ever sees opaque clip ids and audio URLs.

## Run

Backend (from `app/backend/`):

```sh
uv sync                # first time only; creates .venv with Python 3.11
uv run uvicorn main:app --reload --port 8000
```

Frontend dev server (from `app/frontend/`):

```sh
npm install            # first time only
npm run dev            # http://localhost:5173, proxies /api to :8000
```

Open http://localhost:5173. The left rail lists every round found in
`app/data/rounds/`; a committed `fixture` round (three 2 s silent wavs) is
included so the app works out of the box.

## Adding a real round

Deal a round directly into the data dir:

```sh
python3 pipeline/scripts/deal_round.py \
  --manifest my_manifest.json \
  --round-dir app/data/rounds/round0
```

Refresh the app; the round appears in the rail. Ratings autosave to
`app/data/rounds/<roundName>/ratings.json`.

## Rating modes

The round screen has two modes (the choice persists in the ratings doc):

- **Tiers** — one card per clip; Top pick / Keep / Pass plus an independent
  Artifacts flag. Best when clips already spread across quality levels.
- **Compare** — A/B forced choice for early rounds where absolute tiers
  saturate. A deterministic Elo pairing engine (K=32) picks each next pair
  adaptively (fewest-compared clip vs its closest-scored rival, unplayed
  pairs first). After ~2n comparisons the ranking is called stable; finish
  anytime to see the ranked list with the top 4 highlighted as keepers.
  Raw comparisons and the derived ranking both autosave.

## API

- `GET  /api/rounds` — `[{name, clipCount, rated}]`
- `GET  /api/rounds/{name}` — `{name, clips: [{id, url}]}` (blind ids only)
- `GET  /api/rounds/{name}/ratings` — saved ratings doc or `{}`
- `PUT  /api/rounds/{name}/ratings` — atomically persists
  `{round, updated, mode, ratings: {blindId: {tier, artifact, ts}},
  comparisons: [{a, b, winner, ts}], ranking: [blindId...]}` —
  `mode` defaults to `"tiers"`; tier-era docs without it stay valid
- `GET  /api/audio/{name}/{file}` — audio stream with HTTP Range support

## Keyboard

Tiers: `space` play/pause current clip, `1`/`2`/`3` top/keep/pass, `a`
artifact flag, `j`/`k` next/previous card (scrolls into view and autoplays).

Compare: `left`/`right` arrow picks the winner, `space` toggles playback
between A and B, `u` undoes the last comparison.
