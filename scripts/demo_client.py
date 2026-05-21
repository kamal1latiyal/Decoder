#!/usr/bin/env python
"""
demo_client.py — laptop-side demo client for the megakernel TTS service.

Connects to the local TTS server (via SSH tunnel to the 5090), sends a
sequence of test prompts that simulate LLM responses in a voice-agent
conversation, plays back the cloned-voice audio in real time, and prints
per-utterance metrics (TTFC, RTF, throughput).

This is the demo recording driver — narrate over its output to show
the full pipeline:
  - LLM text (simulated here for determinism in the recording)
  - Megakernel-driven Qwen3-TTS streaming the response
  - Pipecat-style chunk-by-chunk dispatch (audio plays as it arrives)

Run (after `ssh -L 8765:localhost:8765 root@<vast>`):
    python scripts/demo_client.py
or:
    python scripts/demo_client.py --url ws://localhost:8765 --save-wav

Dependencies: websockets, sounddevice, numpy
"""

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets


# Sample prompts that read like a voice agent's responses.  Picked to
# showcase: short utterance, medium, long with technical content.
DEMO_TURNS = [
    ("Greeting",         "Hi there! How can I help you today?"),
    ("Medium reply",     "Sure, I can explain that. Machine learning is a branch of artificial intelligence focused on systems that improve from data."),
    ("Technical detail", "The megakernel achieves over a thousand tokens per second by launching a single persistent CUDA kernel that fuses the entire decode loop across one hundred twenty eight thread blocks."),
]

SAMPLE_RATE = 24_000   # server emits 24 kHz mono int16


async def synthesize_one(server_url: str, text: str, *,
                          save_path: Path | None = None) -> dict:
    """Send one text, stream audio chunks, play them live, return metrics."""
    ws_url = f"{server_url}/synthesize"

    # OutputStream pushes chunks as they arrive; we open it before the
    # first chunk arrives so playback starts at the moment of first byte.
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=0,            # let PortAudio pick
        latency="low",
    )
    stream.start()

    chunks: list[bytes] = []
    chunk_arrivals: list[float] = []
    first_chunk_t: float | None = None
    server_metrics: dict | None = None

    try:
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
            t_send = time.perf_counter()
            await ws.send(json.dumps({"text": text}))

            async for msg in ws:
                if isinstance(msg, bytes):
                    if first_chunk_t is None:
                        first_chunk_t = time.perf_counter()
                    chunk_arrivals.append(time.perf_counter() - t_send)
                    chunks.append(msg)
                    samples = np.frombuffer(msg, dtype=np.int16)
                    stream.write(samples)
                else:
                    data = json.loads(msg)
                    if data.get("type") == "done":
                        server_metrics = data.get("metrics", {})
                    elif data.get("type") == "error":
                        raise RuntimeError(f"server error: {data.get('message')}")

            t_done = time.perf_counter()
    finally:
        stream.stop()
        stream.close()

    pcm = b"".join(chunks)
    audio_dur = (len(pcm) // 2) / SAMPLE_RATE
    wall = t_done - t_send
    ttfc_ms = (first_chunk_t - t_send) * 1000 if first_chunk_t else 0.0
    rtf = wall / audio_dur if audio_dur > 0 else 0.0

    if save_path is not None:
        with wave.open(str(save_path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)

    return {
        "text_chars":   len(text),
        "ttfc_ms":      ttfc_ms,
        "rtf":          rtf,
        "audio_s":      audio_dur,
        "wall_s":       wall,
        "chunks":       len(chunks),
        "first_chunk_ms":   chunk_arrivals[0] * 1000 if chunk_arrivals else 0.0,
        "median_chunk_ms":  np.median(np.diff(chunk_arrivals)) * 1000 if len(chunk_arrivals) > 1 else 0.0,
        "server_metrics": server_metrics,
    }


def print_metrics(label: str, text: str, m: dict) -> None:
    print(f"\n── {label} ── ({m['text_chars']} chars)")
    print(f"   \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print(f"   TTFC          : {m['ttfc_ms']:>7.1f} ms   (audio starts playing here)")
    print(f"   RTF           : {m['rtf']:>7.3f}      (1.0 = realtime; lower = faster)")
    print(f"   Audio length  : {m['audio_s']:>7.2f} s")
    print(f"   Wall time     : {m['wall_s']:>7.2f} s")
    print(f"   PCM chunks    : {m['chunks']}   (streaming chunk-by-chunk if > 1)")
    if m['chunks'] > 1:
        print(f"   Chunk cadence : ~{m['median_chunk_ms']:.0f} ms between chunks")


async def main_async(server_url: str, save_dir: Path | None) -> None:
    print("=" * 68)
    print(" Decoder voice-agent demo — megakernel TTS streaming")
    print(f" Server: {server_url}")
    print("=" * 68)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for i, (label, text) in enumerate(DEMO_TURNS):
        save_path = save_dir / f"demo_{i+1:02d}_{label.lower().replace(' ', '_')}.wav" if save_dir else None
        m = await synthesize_one(server_url, text, save_path=save_path)
        print_metrics(label, text, m)
        all_metrics.append(m)
        await asyncio.sleep(0.5)   # natural conversational pause

    print("\n" + "=" * 68)
    print(" Summary")
    print("=" * 68)
    print(f"  Median TTFC : {np.median([m['ttfc_ms'] for m in all_metrics]):.0f} ms")
    print(f"  Median RTF  : {np.median([m['rtf'] for m in all_metrics]):.3f}")
    print(f"  Total audio : {sum(m['audio_s'] for m in all_metrics):.1f} s across {len(all_metrics)} utterances")
    if save_dir:
        print(f"  WAVs saved  : {save_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8765",
                   help="TTS server WebSocket URL (default: ws://localhost:8765)")
    p.add_argument("--save-wav", action="store_true",
                   help="Also write each utterance to a .wav file")
    p.add_argument("--save-dir", default="benchmarks/results/demo",
                   help="Where to save WAVs if --save-wav is set")
    args = p.parse_args()

    save_dir = Path(args.save_dir) if args.save_wav else None
    try:
        asyncio.run(main_async(args.url, save_dir))
    except KeyboardInterrupt:
        print("\n(interrupted)")
        sys.exit(130)


if __name__ == "__main__":
    main()
