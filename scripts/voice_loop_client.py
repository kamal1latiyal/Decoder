#!/usr/bin/env python
"""
voice_loop_client.py — laptop-side walkie-talkie client for the voice agent.

You press Enter, speak, press Enter again to send. The agent transcribes,
generates a reply, speaks it back through your speakers in your cloned voice.
Loops indefinitely until Ctrl+C.

Server side: `pipecat_integration/voice_loop.py` running on the 5090 box,
typically tunneled to laptop port 8766:
    ssh -p <PORT> -N -L 8766:localhost:8766 root@<HOST>

Run on laptop:
    python3 scripts/voice_loop_client.py
or:
    python3 scripts/voice_loop_client.py --url ws://localhost:8766/voice

Dependencies: websockets, sounddevice, numpy.
"""

import argparse
import asyncio
import json
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets


MIC_SR = 16000     # what we send (Deepgram-friendly)
SPK_SR = 24000     # what TTS sends back
MIC_CHUNK_MS = 50  # 50ms chunks


async def record_and_send(ws, stop_event: asyncio.Event):
    """Background: read mic stream, send chunks until stop_event."""
    loop = asyncio.get_running_loop()
    q: queue.Queue = queue.Queue()
    n_samples_per_chunk = int(MIC_SR * MIC_CHUNK_MS / 1000)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        q.put(indata.copy().tobytes())

    stream = sd.InputStream(
        samplerate=MIC_SR, channels=1, dtype="int16",
        blocksize=n_samples_per_chunk, callback=callback, latency="low",
    )
    stream.start()

    try:
        while not stop_event.is_set():
            try:
                chunk = await loop.run_in_executor(None, q.get, True, 0.1)
            except queue.Empty:
                continue
            except Exception:
                continue
            if chunk is None:
                continue
            await ws.send(chunk)
    finally:
        stream.stop()
        stream.close()


async def play_until_done(ws, on_meta) -> dict:
    """Receive text+binary frames from server until 'done'. Plays PCM live.
    Returns the final metrics dict."""
    spk = sd.OutputStream(samplerate=SPK_SR, channels=1, dtype="int16", latency="low")
    spk.start()
    metrics: dict = {}
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                samples = np.frombuffer(msg, dtype=np.int16)
                spk.write(samples)
            else:
                data = json.loads(msg)
                on_meta(data)
                if data.get("type") == "done":
                    metrics = data.get("metrics", {})
                    break
                if data.get("type") == "error":
                    print(f"   ⚠ server error: {data.get('message')}", file=sys.stderr)
                    break
    finally:
        spk.stop()
        spk.close()
    return metrics


def on_meta(data: dict):
    t = data.get("type")
    if t == "transcript":
        print(f"   you  : {data.get('text', '')!r}  ({data.get('latency_ms',0)} ms STT)")
    elif t == "reply":
        print(f"   agent: {data.get('text', '')!r}  ({data.get('latency_ms',0)} ms LLM)")
    elif t == "done":
        m = data.get("metrics", {})
        print(f"   ── turn: STT {m.get('stt_ms','-')} ms · "
              f"LLM {m.get('llm_ms','-')} ms · "
              f"TTS {m.get('tts_wall_ms','-')} ms (server TTFC "
              f"{m.get('tts_server',{}).get('ttfc_ms','-')} ms) · "
              f"total {m.get('turn_total_ms','-')} ms")


async def one_turn(url: str):
    print("\nPress Enter to start speaking …", end="", flush=True)
    await asyncio.get_running_loop().run_in_executor(None, input, "")
    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        print("   🎙  recording — press Enter again to stop", flush=True)
        stop_event = asyncio.Event()
        sender = asyncio.create_task(record_and_send(ws, stop_event))

        # Wait for user to press Enter (signals end of speaking)
        await asyncio.get_running_loop().run_in_executor(None, input, "")
        stop_event.set()
        await sender
        await ws.send("END")

        print("   …processing", flush=True)
        await play_until_done(ws, on_meta)


async def main_async(url: str):
    print("=" * 68)
    print(" Decoder voice loop — walkie-talkie client")
    print(f" Server: {url}")
    print(" Press Enter to speak, Enter again to send.  Ctrl+C to quit.")
    print("=" * 68)
    while True:
        try:
            await one_turn(url)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n   ⚠ {type(e).__name__}: {e}", file=sys.stderr)
            await asyncio.sleep(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8766/voice",
                   help="Voice-loop server WS URL (default: ws://localhost:8766/voice)")
    args = p.parse_args()
    try:
        asyncio.run(main_async(args.url))
    except KeyboardInterrupt:
        print("\nbye.")


if __name__ == "__main__":
    main()
