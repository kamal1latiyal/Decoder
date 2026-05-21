"""
voice_loop.py — live voice-agent loop server (runs on the 5090 box).

Designed for the walkie-talkie demo: client sends a single utterance as raw
PCM, then a text "END" marker. Server runs Deepgram STT → Claude LLM →
megakernel TTS, streams the audio response back as raw PCM chunks, then a
text "DONE" marker. Loop on the same WebSocket connection.

Protocol (one turn = repeat as needed on the same connection):
  client → server : binary  raw PCM int16 mono @ 16 kHz   (mic chunks)
  client → server : text    "END"
  server → client : text    {"type":"transcript","text":"..."}
  server → client : text    {"type":"reply","text":"..."}
  server → client : binary  raw PCM int16 mono @ 24 kHz   (TTS chunks)
  server → client : text    {"type":"done","metrics":{...}}

Run on the box:
    python -m pipecat_integration.voice_loop \\
        --host 0.0.0.0 --port 8766 \\
        --tts-url ws://localhost:8765

Requires DEEPGRAM_API_KEY + ANTHROPIC_API_KEY in env or .env.
"""

import argparse
import asyncio
import io
import json
import logging
import os
import time
import wave
from pathlib import Path

import requests
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voice_loop")

REPO = Path(__file__).resolve().parent.parent


def _load_dotenv():
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

DEEPGRAM = os.environ["DEEPGRAM_API_KEY"]
ANTHROPIC = os.environ["ANTHROPIC_API_KEY"]
TTS_URL = os.environ.get("TTS_SERVER_URL", "ws://localhost:8765")

# Conversation system prompt — kept short on purpose; TTS is greedy & capped
# at ~10s so brief replies sound best.
SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Keep replies very short — "
    "one short sentence, ideally under 12 words. The user is hearing you "
    "through text-to-speech, so avoid URLs, long names, or bullet lists. "
    "Reply naturally as if you were chatting."
)


# ───────────────── STT ─────────────────
def stt_deepgram(pcm_16k_mono: bytes) -> tuple[str, float]:
    """Wrap raw PCM in a WAV header and send to Deepgram. Returns (text, wall_s)."""
    t0 = time.perf_counter()
    if not pcm_16k_mono:
        return "", 0.0
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_16k_mono)
    buf.seek(0)

    r = requests.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-3", "smart_format": "true", "punctuate": "true"},
        headers={
            "Authorization": f"Token {DEEPGRAM}",
            "Content-Type": "audio/wav",
        },
        data=buf.read(),
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    text = (
        j.get("results", {})
         .get("channels", [{}])[0]
         .get("alternatives", [{}])[0]
         .get("transcript", "")
    ).strip()
    return text, time.perf_counter() - t0


# ───────────────── LLM ─────────────────
def llm_claude(history: list[dict], user_text: str) -> tuple[str, float]:
    """Call Claude Haiku 4.5 with conversation history. Returns (reply, wall_s)."""
    t0 = time.perf_counter()
    messages = list(history) + [{"role": "user", "content": user_text}]
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 80,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        },
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text").strip()
    return text, time.perf_counter() - t0


# ───────────────── TTS ─────────────────
async def tts_stream(text: str, ws_client_send_bytes, tts_url: str) -> tuple[dict, float]:
    """Synthesize `text` via the megakernel TTS server. Stream PCM chunks
    directly to the client (via the provided async sender). Returns
    (server_metrics, wall_s)."""
    t0 = time.perf_counter()
    metrics = {}
    async with websockets.connect(f"{tts_url}/synthesize", max_size=10 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"text": text}))
        async for msg in ws:
            if isinstance(msg, bytes):
                await ws_client_send_bytes(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "done":
                    metrics = data.get("metrics", {})
                elif data.get("type") == "error":
                    raise RuntimeError(f"TTS server error: {data.get('message')}")
    return metrics, time.perf_counter() - t0


# ───────────────── FastAPI app ─────────────────
app = FastAPI(title="Voice Loop Agent")


@app.websocket("/voice")
async def voice(ws: WebSocket):
    await ws.accept()
    history: list[dict] = []
    log.info(f"Client connected: {ws.client}")

    try:
        while True:
            # ── 1. Receive audio until "END" ──
            mic_buf = bytearray()
            t_first_audio = None
            while True:
                msg = await ws.receive()
                # FastAPI WS gives us {"type":"websocket.disconnect"|"...receive", "bytes"|"text"}
                if msg["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect
                if "bytes" in msg and msg["bytes"] is not None:
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                    mic_buf.extend(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    if msg["text"].strip().upper() == "END":
                        break
                    # ignore other text messages

            t_end = time.perf_counter()
            recv_dur = (t_end - t_first_audio) if t_first_audio else 0.0
            log.info(f"Got {len(mic_buf)} bytes mic audio over {recv_dur:.2f} s")

            if len(mic_buf) < 16000 * 2 * 0.3:    # < 0.3 s of audio
                await ws.send_text(json.dumps({"type": "error", "message": "Audio too short"}))
                continue

            # ── 2. STT ──
            try:
                transcript, t_stt = stt_deepgram(bytes(mic_buf))
            except Exception as e:
                log.exception("STT failed")
                await ws.send_text(json.dumps({"type": "error", "message": f"STT failed: {e}"}))
                continue
            log.info(f"STT [{t_stt*1000:.0f} ms]: {transcript!r}")
            await ws.send_text(json.dumps({"type": "transcript", "text": transcript, "latency_ms": int(t_stt * 1000)}))

            if not transcript:
                await ws.send_text(json.dumps({"type": "error", "message": "Empty transcript"}))
                continue

            # ── 3. LLM ──
            try:
                reply, t_llm = llm_claude(history, transcript)
            except Exception as e:
                log.exception("LLM failed")
                await ws.send_text(json.dumps({"type": "error", "message": f"LLM failed: {e}"}))
                continue
            log.info(f"LLM [{t_llm*1000:.0f} ms]: {reply!r}")
            await ws.send_text(json.dumps({"type": "reply", "text": reply, "latency_ms": int(t_llm * 1000)}))

            # Update conversation history (lightweight)
            history.append({"role": "user", "content": transcript})
            history.append({"role": "assistant", "content": reply})
            # Cap history length so prompt doesn't grow unboundedly
            history[:] = history[-12:]

            # ── 4. TTS streamed back ──
            try:
                async def send_pcm(b: bytes):
                    await ws.send_bytes(b)
                tts_metrics, t_tts = await tts_stream(reply, send_pcm, TTS_URL)
            except Exception as e:
                log.exception("TTS failed")
                await ws.send_text(json.dumps({"type": "error", "message": f"TTS failed: {e}"}))
                continue

            log.info(f"TTS [{t_tts*1000:.0f} ms wall, server metrics: {tts_metrics}]")
            await ws.send_text(json.dumps({
                "type": "done",
                "metrics": {
                    "stt_ms":         int(t_stt * 1000),
                    "llm_ms":         int(t_llm * 1000),
                    "tts_wall_ms":    int(t_tts * 1000),
                    "tts_server":     tts_metrics,
                    "turn_total_ms":  int((t_stt + t_llm + t_tts) * 1000),
                },
            }))

    except WebSocketDisconnect:
        log.info("Client disconnected.")
    except Exception as e:
        log.exception(f"Voice loop error: {e}")


@app.get("/health")
async def health():
    return {"status": "ready", "tts_url": TTS_URL}


def main():
    global TTS_URL
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--tts-url", default=TTS_URL,
                   help="Megakernel TTS server URL (default: ws://localhost:8765)")
    args = p.parse_args()
    TTS_URL = args.tts_url

    import uvicorn
    uvicorn.run("pipecat_integration.voice_loop:app",
                host=args.host, port=args.port, log_level="info",
                ws_ping_interval=None)


if __name__ == "__main__":
    main()
