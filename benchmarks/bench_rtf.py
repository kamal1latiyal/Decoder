"""
RTF Benchmark — Real-Time Factor.

Measures: (total wall-clock synthesis time) / (total audio duration).
RTF < 1.0 means faster-than-real-time. Target: RTF < 0.3.

Also reports talker tok/s (from server metrics).

Run:
  python benchmarks/bench_rtf.py [--url ws://localhost:8765] [--runs 5]
"""

import argparse
import asyncio
import json
import statistics
import time

import websockets

# Sentences of varying lengths to stress-test sustained RTF
TEST_SENTENCES = [
    # Short (< 20 tokens)
    "The sky is blue.",
    # Medium (~40 tokens)
    "In machine learning, a neural network is a series of algorithms that attempt to "
    "recognize underlying relationships in a set of data.",
    # Long (~80 tokens)
    "The real-time factor measures how long it takes a system to generate one second of audio "
    "relative to the duration of that audio. A real-time factor below one means the system "
    "runs faster than real time, which is essential for low-latency voice applications where "
    "users expect immediate responses without noticeable delay.",
]


async def measure_rtf(server_url: str, text: str) -> dict:
    """Collect RTF and tok/s from one synthesis request."""
    ws_url = f"{server_url}/synthesize"
    total_audio_bytes = 0
    t_start = time.perf_counter()
    result = {}

    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"text": text}))

        async for msg in ws:
            if isinstance(msg, bytes):
                total_audio_bytes += len(msg)
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "done":
                    result = data.get("metrics", {})

    t_end = time.perf_counter()
    wall_time = t_end - t_start

    # Compute from audio bytes: int16 mono 24kHz → 2 bytes/sample
    audio_samples = total_audio_bytes // 2
    audio_duration = audio_samples / 24_000

    # Use server-reported RTF (more accurate, excludes network overhead)
    server_rtf = result.get("rtf", wall_time / audio_duration if audio_duration > 0 else 0)

    return {
        "wall_time_s": wall_time,
        "audio_duration_s": audio_duration,
        "audio_bytes": total_audio_bytes,
        "server_rtf": server_rtf,
        "client_rtf": wall_time / audio_duration if audio_duration > 0 else 0,
        "tokens_per_sec": result.get("tokens_per_sec", 0),
        "ttfc_ms": result.get("ttfc_ms", 0),
        "total_tokens": result.get("total_tokens", 0),
    }


async def run_benchmark(server_url: str, runs: int):
    print(f"\nRTF Benchmark — {runs} runs per sentence")
    print(f"Server: {server_url}")
    print("=" * 70)

    all_rtf = []
    all_tps = []

    for sentence in TEST_SENTENCES:
        run_results = []
        label = sentence[:50] + ("..." if len(sentence) > 50 else "")
        print(f"\nText ({len(sentence)} chars): \"{label}\"")
        print(f"  {'Run':>3}  {'RTF':>8}  {'tok/s':>8}  {'audio_s':>8}  {'wall_s':>8}  {'TTFC_ms':>8}")

        for i in range(runs):
            try:
                r = await measure_rtf(server_url, sentence)
                run_results.append(r)
                print(
                    f"  {i+1:3d}  "
                    f"{r['server_rtf']:8.4f}  "
                    f"{r['tokens_per_sec']:8.1f}  "
                    f"{r['audio_duration_s']:8.3f}  "
                    f"{r['wall_time_s']:8.3f}  "
                    f"{r['ttfc_ms']:8.1f}"
                )
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"  {i+1:3d}  ERROR: {e}")

        if run_results:
            rtfs = [r["server_rtf"] for r in run_results]
            tpss = [r["tokens_per_sec"] for r in run_results if r["tokens_per_sec"] > 0]
            print(f"  ── RTF mean={statistics.mean(rtfs):.4f}  "
                  f"min={min(rtfs):.4f}  max={max(rtfs):.4f}  |  "
                  f"tok/s mean={statistics.mean(tpss):.0f}")
            all_rtf.extend(rtfs)
            all_tps.extend(tpss)

    print("\n" + "=" * 70)
    print("Overall RTF Summary:")
    print(f"  mean   : {statistics.mean(all_rtf):.4f}")
    print(f"  median : {statistics.median(all_rtf):.4f}")
    print(f"  p95    : {sorted(all_rtf)[int(len(all_rtf)*0.95)]:.4f}")
    print(f"  min    : {min(all_rtf):.4f}")
    print(f"  max    : {max(all_rtf):.4f}")
    print(f"\nTalker Throughput:")
    print(f"  mean tok/s : {statistics.mean(all_tps):.1f}")
    print(f"\nTarget RTF: < 0.3")
    passing = sum(1 for r in all_rtf if r < 0.3)
    print(f"Pass rate: {passing}/{len(all_rtf)} ({100*passing/len(all_rtf):.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="RTF Benchmark")
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.url, args.runs))


if __name__ == "__main__":
    main()
