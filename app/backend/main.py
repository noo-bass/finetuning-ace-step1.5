"""FastAPI backend for the taste-teaching rating app.

Serves rounds dealt by pipeline/scripts/deal_round.py from app/data/rounds/.
A round dir contains audio/*.flac|wav, mapping.json ({seed, clips: {blindId:
{...true config...}}}) and, once rated, ratings.json.

Blinding is a core requirement: mapping.json's configs are NEVER exposed
through the API -- clients only ever see opaque blind ids and audio URLs.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

DATA_ROOT = (Path(__file__).resolve().parent.parent / "data" / "rounds").resolve()
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
MEDIA_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}
CHUNK_SIZE = 64 * 1024
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

app = FastAPI(title="finetune-rating-app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_round_dir(name: str) -> Path:
    """Resolve a round name to a dir under DATA_ROOT, rejecting traversal."""
    if not NAME_RE.match(name) or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid round name")
    round_dir = (DATA_ROOT / name).resolve()
    if round_dir.parent != DATA_ROOT or not round_dir.is_dir():
        raise HTTPException(status_code=404, detail="round not found")
    return round_dir


def _round_clips(round_dir: Path) -> list[Path]:
    audio_dir = round_dir / "audio"
    if not audio_dir.is_dir():
        return []
    return sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)


@app.get("/api/rounds")
def list_rounds() -> list[dict]:
    rounds = []
    if DATA_ROOT.is_dir():
        for d in sorted(DATA_ROOT.iterdir()):
            if not d.is_dir() or not (d / "mapping.json").exists():
                continue
            rounds.append(
                {
                    "name": d.name,
                    "clipCount": len(_round_clips(d)),
                    "rated": (d / "ratings.json").exists(),
                }
            )
    return rounds


@app.get("/api/rounds/{name}")
def get_round(name: str) -> dict:
    round_dir = _safe_round_dir(name)
    clips = [
        {"id": p.stem, "url": f"/api/audio/{name}/{p.name}"}
        for p in _round_clips(round_dir)
    ]
    return {"name": name, "clips": clips}


class Rating(BaseModel):
    tier: str | None = Field(default=None, pattern="^(top|keep|pass)$")
    artifact: bool = False
    ts: str | None = None


class Comparison(BaseModel):
    a: str
    b: str
    winner: str
    ts: str | None = None


class RatingsPayload(BaseModel):
    """Ratings doc. mode "tiers" (implied when absent) uses `ratings` only;
    mode "pairwise" additionally carries raw `comparisons` and the derived
    `ranking`. Tier-era clients that send just {ratings} remain valid."""

    mode: Literal["tiers", "pairwise"] = "tiers"
    ratings: dict[str, Rating] = Field(default_factory=dict)
    comparisons: list[Comparison] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)


@app.get("/api/rounds/{name}/ratings")
def get_ratings(name: str) -> dict:
    round_dir = _safe_round_dir(name)
    ratings_file = round_dir / "ratings.json"
    if not ratings_file.exists():
        return {}
    return json.loads(ratings_file.read_text())


@app.put("/api/rounds/{name}/ratings")
def put_ratings(name: str, payload: RatingsPayload) -> dict:
    round_dir = _safe_round_dir(name)
    known_ids = {p.stem for p in _round_clips(round_dir)}

    unknown = set(payload.ratings) - known_ids
    unknown |= set(payload.ranking) - known_ids
    for cmp in payload.comparisons:
        if cmp.a == cmp.b:
            raise HTTPException(status_code=400, detail=f"comparison pairs a clip with itself: {cmp.a}")
        if cmp.winner not in (cmp.a, cmp.b):
            raise HTTPException(status_code=400, detail=f"winner {cmp.winner!r} is not in pair ({cmp.a}, {cmp.b})")
        unknown |= {cmp.a, cmp.b} - known_ids
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown clip ids: {sorted(unknown)}")

    doc = {
        "round": name,
        "updated": datetime.now(timezone.utc).isoformat(),
        "mode": payload.mode,
        "ratings": {
            cid: r.model_dump(exclude_none=True) for cid, r in payload.ratings.items()
        },
    }
    if payload.mode == "pairwise" or payload.comparisons or payload.ranking:
        doc["comparisons"] = [c.model_dump(exclude_none=True) for c in payload.comparisons]
        doc["ranking"] = payload.ranking
    # Atomic write: temp file in the same dir, then rename.
    fd, tmp_path = tempfile.mkstemp(dir=round_dir, prefix=".ratings-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp_path, round_dir / "ratings.json")
    except BaseException:
        os.unlink(tmp_path)
        raise
    return doc


def _iter_file(path: Path, start: int, end: int) -> Iterator[bytes]:
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@app.get("/api/audio/{name}/{file}")
def stream_audio(name: str, file: str, request: Request) -> Response:
    round_dir = _safe_round_dir(name)
    if not NAME_RE.match(file) or file.startswith("."):
        raise HTTPException(status_code=400, detail="invalid file name")
    path = (round_dir / "audio" / file).resolve()
    if path.parent != (round_dir / "audio").resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if path.suffix.lower() not in AUDIO_EXTS:
        raise HTTPException(status_code=404, detail="not an audio file")

    size = path.stat().st_size
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    range_header = request.headers.get("range")

    if range_header:
        m = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
        if not m or (not m.group(1) and not m.group(2)):
            raise HTTPException(status_code=416, detail="invalid range")
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        else:  # suffix range: bytes=-N
            start = max(size - int(m.group(2)), 0)
            end = size - 1
        if start >= size or end < start:
            return Response(
                status_code=416, headers={"Content-Range": f"bytes */{size}"}
            )
        end = min(end, size - 1)
        return StreamingResponse(
            _iter_file(path, start, end),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(end - start + 1),
                "Accept-Ranges": "bytes",
            },
        )

    return StreamingResponse(
        _iter_file(path, 0, size - 1),
        media_type=media_type,
        headers={"Content-Length": str(size), "Accept-Ranges": "bytes"},
    )
