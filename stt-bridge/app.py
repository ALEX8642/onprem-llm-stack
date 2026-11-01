"""
OpenAI-compatible speech-to-text front end for OpenWebUI (no model here, no RAM/GPU on the Sparks).

Decodes any upload (browser webm/opus, mp3, wav, m4a) to 16 kHz mono PCM with ffmpeg, adds a short
lead-in of silence, then transcribes with:
  1. primary: an OpenAI-API Whisper server (STT_OPENAI_URL), i.e. OpenVINO Model Server running
     whisper-large-v3-turbo on NAS's Intel Arc A750 (~0.1-0.3 s per clip);
  2. fallback, if that fails or is unset: the Wyoming faster-whisper on NAS (CPU, also used by
     Home Assistant), so voice never goes fully dead.
"""
import io, json, os, socket, subprocess, time, logging, urllib.request, uuid, wave
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

HOST = os.environ.get("WYOMING_HOST", "nas")
PORT = int(os.environ.get("WYOMING_PORT", "10300"))
DEFAULT_LANG = os.environ.get("STT_LANGUAGE", "en") or None   # "" = let Whisper auto-detect
TIMEOUT = float(os.environ.get("WYOMING_TIMEOUT", "120"))
OPENAI_URL = os.environ.get("STT_OPENAI_URL", "").rstrip("/")   # e.g. http://nas:8100/v3
OPENAI_MODEL = os.environ.get("STT_OPENAI_MODEL", "whisper-large-v3-turbo")
OPENAI_TIMEOUT = float(os.environ.get("STT_OPENAI_TIMEOUT", "30"))
# Silence prepended before Whisper. Browser voice mode starts recording the instant it detects
# speech, so clips begin mid-word ("Hello" arrives as "-ello") and Whisper guesses wrong ("I'm in
# a row."). 0.5 s of lead-in fixed that on real recordings (2026-10-11) at ~no cost; 0 disables.
LEAD_SILENCE = float(os.environ.get("STT_LEAD_SILENCE_SEC", "0.5"))
RATE, WIDTH, CHANNELS, CHUNK = 16000, 2, 1, 32000
log = logging.getLogger("stt-bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = FastAPI(title="stt-bridge")


def _send(sock, typ, data=None, payload=b""):
    body = json.dumps(data or {}).encode()
    hdr = {"type": typ, "data_length": len(body)}
    if payload:
        hdr["payload_length"] = len(payload)
    sock.sendall(json.dumps(hdr).encode() + b"\n" + body + payload)


def _events(sock):
    f = sock.makefile("rb")
    while True:
        line = f.readline()
        if not line:
            raise ConnectionError("Wyoming server closed the connection")
        hdr = json.loads(line)
        dl = hdr.get("data_length", 0)
        data = json.loads(f.read(dl)) if dl else hdr.get("data") or {}
        if hdr.get("payload_length"):
            f.read(hdr["payload_length"])
        yield hdr["type"], data


def wyoming_transcribe(pcm: bytes, language: str | None) -> str:
    fmt = {"rate": RATE, "width": WIDTH, "channels": CHANNELS}
    with socket.create_connection((HOST, PORT), timeout=TIMEOUT) as s:
        _send(s, "transcribe", {"language": language} if language else {})
        _send(s, "audio-start", fmt)
        for i in range(0, len(pcm), CHUNK):
            _send(s, "audio-chunk", fmt, pcm[i:i + CHUNK])
        _send(s, "audio-stop", {})
        for typ, data in _events(s):
            if typ == "transcript":
                return (data.get("text") or "").strip()
            if typ == "error":
                raise HTTPException(502, f"Wyoming error: {data.get('text')}")


def openai_transcribe(pcm: bytes, language: str | None) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS); w.setsampwidth(WIDTH); w.setframerate(RATE); w.writeframes(pcm)
    b = uuid.uuid4().hex
    fields = [("model", OPENAI_MODEL)] + ([("language", language)] if language else [])
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields)
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
             f"Content-Type: audio/wav\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"{OPENAI_URL}/audio/transcriptions", body,
                                 {"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=OPENAI_TIMEOUT) as r:
        return (json.loads(r.read()).get("text") or "").strip()


def decode(data: bytes) -> bytes:
    p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                        "-ac", str(CHANNELS), "-ar", str(RATE), "pipe:1"], input=data, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise HTTPException(400, f"could not decode audio: {p.stderr.decode()[:200]}")
    return p.stdout


def wyoming_info():
    with socket.create_connection((HOST, PORT), timeout=5) as s:
        _send(s, "describe")
        for typ, data in _events(s):
            if typ == "info":
                asr = (data.get("asr") or [{}])[0]
                return {"backend": f"{HOST}:{PORT}", "asr": asr.get("name"),
                        "models": [m.get("name") for m in asr.get("models", [])]}


@app.get("/health")
def health():
    """ok = primary (OpenAI/OVMS) answers; degraded = only the Wyoming fallback; 503 = neither."""
    out = {}
    if OPENAI_URL:
        try:
            with urllib.request.urlopen(f"{OPENAI_URL.rsplit('/', 1)[0]}/v1/models", timeout=5) as r:
                ids = [m["id"] for m in json.loads(r.read()).get("data", [])]
            out["primary"] = {"url": OPENAI_URL, "model": OPENAI_MODEL, "loaded": OPENAI_MODEL in ids}
        except Exception as e:
            out["primary"] = {"url": OPENAI_URL, "error": str(e)[:120]}
    try:
        out["fallback"] = wyoming_info()
    except Exception as e:
        out["fallback"] = {"error": str(e)[:120]}
    primary_ok = bool(out.get("primary", {}).get("loaded"))
    out["status"] = "ok" if primary_ok else ("degraded" if "asr" in out["fallback"] else "down")
    # Kept for older callers (smoke-test.sh prints these two).
    out["backend"] = OPENAI_URL if primary_ok else out["fallback"].get("backend")
    out["asr"] = OPENAI_MODEL if primary_ok else out["fallback"].get("asr")
    if out["status"] == "down":
        raise HTTPException(503, out)
    return out


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "whisper-1", "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = Form("whisper-1"),
                         language: str | None = Form(None), response_format: str = Form("json"),
                         prompt: str | None = Form(None), temperature: float | None = Form(None)):
    pcm = decode(await file.read())
    dur = len(pcm) / (RATE * WIDTH)
    pcm = b"\0" * (int(RATE * LEAD_SILENCE) * WIDTH) + pcm
    t = time.time()
    lang = language or DEFAULT_LANG
    text, via = None, "wyoming"
    if OPENAI_URL:
        try:
            text, via = openai_transcribe(pcm, lang), "openai"
        except Exception as e:
            log.warning("primary STT %s failed (%s); falling back to Wyoming", OPENAI_URL, e)
    if text is None:
        try:
            text = wyoming_transcribe(pcm, lang)
        except (OSError, ConnectionError) as e:
            raise HTTPException(502, f"no STT backend answered (Wyoming {HOST}:{PORT}: {e})")
    log.info("transcribed %.1fs of audio in %.2fs via %s", dur, time.time() - t, via)
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return JSONResponse({"task": "transcribe", "language": language or DEFAULT_LANG or "", "duration": dur, "text": text})
    return JSONResponse({"text": text})
