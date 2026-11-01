"""
TTS proxy: OpenAI-compatible /v1/audio/speech → Chatterbox native /tts.

Adds per-message emotional expressiveness by estimating exaggeration (0.2–0.85)
from text content — punctuation, sentiment keywords, and tone markers — then
forwarding to the Chatterbox /tts endpoint which exposes the parameter.
The OpenAI-compatible /v1/audio/speech endpoint Chatterbox exposes does not
accept exaggeration, hence the proxy.
"""

import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHATTERBOX_URL = os.getenv("CHATTERBOX_URL", "http://chatterbox:8004")

app = FastAPI(title="Chatterbox TTS Proxy")


# ---------------------------------------------------------------------------
# Sentiment → exaggeration
# ---------------------------------------------------------------------------

_EXCITEMENT = {
    "amazing", "wonderful", "fantastic", "incredible", "awesome", "love",
    "excited", "thrilled", "brilliant", "excellent", "perfect", "great",
    "beautiful", "spectacular", "outstanding", "magnificent", "hooray",
    "congratulations", "fantastic",
}
_EMPATHY = {
    "sorry", "unfortunately", "apolog", "concern", "difficult", "worried",
    "sad", "regret", "problem", "trouble", "unfortunate", "hard time",
    "my condolences", "i understand",
}


def estimate_exaggeration(text: str) -> float:
    score = 0.50
    lower = text.lower()

    # Exclamation intensity
    bangs = text.count("!")
    if bangs >= 2:
        score += 0.20
    elif bangs == 1:
        score += 0.10

    # Excitement vocabulary
    if any(w in lower for w in _EXCITEMENT):
        score += 0.10

    # Empathetic / sombre content
    if any(w in lower for w in _EMPATHY):
        score -= 0.12

    # Pure question — more measured, curious delivery
    if text.strip().endswith("?") and bangs == 0:
        score -= 0.10

    result = round(max(0.20, min(0.85, score)), 2)
    return result


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "Abigail.wav"
    response_format: str = "mp3"
    speed: float = 1.0
    seed: Optional[int] = None
    language: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    exaggeration = estimate_exaggeration(req.input)
    preview = req.input[:60].replace("\n", " ")
    log.info("TTS voice=%s exaggeration=%.2f text=%r", req.voice, exaggeration, preview)

    fmt_map = {"mp3": "mp3", "wav": "wav", "opus": "wav", "aac": "mp3", "flac": "wav"}
    out_fmt = fmt_map.get(req.response_format, "mp3")

    payload: dict = {
        "text": req.input,
        "voice_mode": "predefined",
        "predefined_voice_id": req.voice,
        "output_format": out_fmt,
        "exaggeration": exaggeration,
        "split_text": True,
    }
    if req.seed is not None:
        payload["seed"] = req.seed
    if req.language:
        payload["language"] = req.language

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{CHATTERBOX_URL}/tts", json=payload)

    if resp.status_code != 200:
        log.error("Chatterbox error %d: %s", resp.status_code, resp.text[:200])
        raise HTTPException(status_code=502, detail=f"Chatterbox error {resp.status_code}")

    media_type = "audio/mpeg" if out_fmt == "mp3" else "audio/wav"
    return Response(content=resp.content, media_type=media_type)


@app.get("/v1/audio/voices")
async def list_voices():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{CHATTERBOX_URL}/v1/audio/voices")
    return resp.json()


@app.get("/health")
async def health():
    return {"status": "ok"}
