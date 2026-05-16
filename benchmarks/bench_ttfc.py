"""
TTFC Benchmark — Time To First audio Chunk.

Measures: text received → first PCM bytes pushed to client.

Definition used:
  t=0:     JSON request received by server
  t=TTFC:  first binary WebSocket message (PCM bytes) sent to client

Run:
  python benchmarks/bench_ttfc.py [--url ws://localhost:8765] [--runs 10]
"""

import argparse
import asyncio
import json
import statistics
import time

import websockets

TEST_SENTENCES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "This is a streaming text-to-speech system powered by a custom CUDA kernel.",
    "Tell me about the weather in San Francisco.",
    "Artificial intelligence is transforming the way we interact with computers.",
]


async def measure_ttfc(server_url: str, text: str) -> float:
    """Returns TTFC in milliseconds for one synthesis request."""
    ws_url = f"{server_url}/synthesize"
    async with websockets.connect(ws_url) as ws:
        t_send = time.perf_counter()
        await ws.send(json.dumps({"text": text}))

        async for msg in ws:
            if isinstance(msg, bytes) and len(msg) > 0:
                t_first = time.perf_counter()
                # Drain remaining messages without measuring
                async for _ in ws:
                    pass
                return (t_first - t_send) * 1000

    return float("inf")


async def run_benchmark(server_url: str, runs: int):
    print(f"\nTTFC Benchmark — {runs} runs per sentence")
    print(f"Server: {server_url}")
    print("=" * 60)

    all_ttfc = []

    for sentence in TEST_SENTENCES:
        sentence_ttfc = []
        print(f"\nText ({len(sentence)} chars): \"{sentence[:50]}\"")

        for i in range(runs):
            try:
                ttfc = await measure_ttfc(server_url, sentence)
                sentence_ttfc.append(ttfc)
                print(f"  Run {i+1:2d}: {ttfc:7.1f} ms")
                await asyncio.sleep(0.5)  # brief pause between requests
            except Exception as e:
                print(f"  Run {i+1:2d}: ERROR — {e}")

        if sentence_ttfc:
            print(f"  ── mean={statistics.mean(sentence_ttfc):.1f}ms  "
                  f"p50={statistics.median(sentence_ttfc):.1f}ms  "
                  f"min={min(sentence_ttfc):.1f}ms  "
                  f"max={max(sentence_ttfc):.1f}ms")
            all_ttfc.extend(sentence_ttfc)

    print("\n" + "=" * 60)
    print("Overall TTFC Summary:")
    print(f"  mean   : {statistics.mean(all_ttfc):.1f} ms")
    print(f"  median : {statistics.median(all_ttfc):.1f} ms")
    print(f"  p95    : {sorted(all_ttfc)[int(len(all_ttfc)*0.95)]:.1f} ms")
    print(f"  min    : {min(all_ttfc):.1f} ms")
    print(f"  max    : {max(all_ttfc):.1f} ms")
    print(f"\nTarget: < 90 ms")
    passing = sum(1 for t in all_ttfc if t < 90)
    print(f"Pass rate: {passing}/{len(all_ttfc)} ({100*passing/len(all_ttfc):.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="TTFC Benchmark")
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.url, args.runs))


if __name__ == "__main__":
    main()
