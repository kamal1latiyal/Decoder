#!/usr/bin/env python
"""
mock_tts_server.py — drop-in stand-in for the megakernel TTS server.

Same WebSocket protocol (ws://host:port/synthesize), but instead of running
the kernel it emits a short sine-tone whose duration scales with the input
text length. Used for end-to-end testing of the voice loop on a laptop
without a GPU.

Protocol exactly mirrors server/app.py:
    client → JSON  {"text": "...", "speaker": "default"}
    server → bytes int16 LE PCM @ 24 kHz mono   (several messages)
    server → JSON  {"type":"done", "metrics":{...}}

Run:
    python scripts/mock_tts_server.py --port 8765
"""

import argparse
import asyncio
import json
import math
import struct
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn


SAMPLE_RATE = 24_000
CHUNK_FRAMES = 4
SAMPLES_PER_FRAME = 1920          # 24 kHz / 12.5 Hz
CHUNK_SAMPLES = CHUNK_FRAMES * SAMPLES_PER_FRAME  # 7680
CHUNK_BYTES = CHUNK_SAMPLES * 2

app = FastAPI(title="Mock TTS Server")


def sine_chunk(freq_hz: float, n_samples: int, phase: float) -> tuple[bytes, float]:
    """Generate a low-amplitude sine chunk; return (bytes, next_phase)."""
    t = (np.arange(n_samples) + 0) / SAMPLE_RATE
    samples = 0.15 * np.sin(2 * math.pi * freq_hz * t + phase)
    pcm = (samples * 32767.0).astype(np.int16).tobytes()
    next_phase = (phase + 2 * math.pi * freq_hz * n_samples / SAMPLE_RATE) % (2 * math.pi)
    return pcm, next_phase


@app.get("/health")
async def health():
    return {"status": "ready", "kind": "mock"}


@app.websocket("/synthesize")
async def synthesize(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        text = req.get("text", "").strip()
        if not text:
            await ws.send_text(json.dumps({"type": "error", "message": "Empty text"}))
            await ws.close()
            return

        # Audio duration scales with text length: ~80 ms per character, capped at 8 s.
        target_duration_s = min(8.0, max(0.6, len(text) * 0.08))
        n_chunks = max(1, int(target_duration_s * 1000 / 320))  # ~320ms per chunk

        t_start = time.perf_counter()
        first_chunk_t = None
        phase = 0.0
        freq = 440.0  # gentle A4

        # Send chunks with realistic cadence (~50 ms between chunks; faster
        # than the real server so playback feels snappy in testing).
        for i in range(n_chunks):
            chunk, phase = sine_chunk(freq, CHUNK_SAMPLES, phase)
            if first_chunk_t is None:
                first_chunk_t = time.perf_counter()
            await ws.send_bytes(chunk)
            await asyncio.sleep(0.05)
            # Step pitch slightly each chunk so it doesn't sound monotone.
            freq += 5.0

        t_end = time.perf_counter()
        wall = t_end - t_start
        audio_dur = n_chunks * (CHUNK_SAMPLES / SAMPLE_RATE)
        await ws.send_text(json.dumps({
            "type": "done",
            "metrics": {
                "ttfc_ms":          round((first_chunk_t - t_start) * 1000, 1),
                "rtf":              round(wall / audio_dur, 3),
                "tokens_per_sec":   0,
                "total_tokens":     n_chunks * 4,
                "audio_duration_s": round(audio_dur, 3),
                "wall_time_s":      round(wall, 3),
                "mock":             True,
            },
        }))

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    print(f"Mock TTS server on http://{args.host}:{args.port}")
    print(f"  /health         → status check")
    print(f"  /synthesize WS  → sine-tone PCM (mock cloned voice)")
    # Pass the app object directly — `scripts/` isn't an importable package.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
